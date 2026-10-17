#!/usr/bin/env python3
"""Режет готовую модель по высоте на нижнюю и верхнюю части — отдельными примами, чтобы в Isaac Sim
ненужную (обычно верх: потолок и верх стеллажей лидар видел издалека и мало, гауссианы там мутные)
удалить одним действием: Stage -> правая кнопка на /World/Upper -> Delete (или скрыть глазом).

Из scenes/reconstructed/<имя>_rko.usda делает <имя>_rko_split.usda:
  /World/Lower, /World/Upper — в каждой меш своей части (цвет и коллайдер) и облако (скрыто):
  файлы <имя>_rko_mesh_lower.usdc / _upper.usdc и <имя>_rko_cloud_lower.usdc / _upper.usdc.
Если есть сцена гауссиан <имя>_gs.usda — и <имя>_gs_split.usda: гауссианы срезаны по той же высоте
обрезкой NuRec (omni:nurec:crop:maxBounds, файл .usdz тот же), а коллайдер и облако — по частям,
как выше. Гауссианы — одним объёмом: Isaac 5.0 рисует только один объём NuRec и не замечает
удаления его прима на лету (проверено), а обрезку одного объёма меняет сразу. Поэтому вернуть
или сдвинуть срез гауссиан можно прямо в Isaac: прим /World/Gaussians/.../gauss -> Property ->
omni:nurec:crop:maxBounds -> третье число (Z).
Исходные файлы не меняются; свет, физика, оболочка (если есть) и настройки рендера — как были.
Треугольник уходит в ту часть, где его центр; облако — по высоте точки.

Срез: --height метров над полом (пол — floor_z_m из out/mesh/<имя>_rko/report.json) или --z —
абсолютная высота в координатах модели.

Запуск: .venv/bin/python scripts/split_model.py sklad03 --height 3
"""

import argparse
import json
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "scenes" / "reconstructed"
PARTS = ("Lower", "Upper")


def copy_layer(src, dst):
    """Слой dst с содержимым src (метаданные и customLayerData — тоже); сохраняется один раз в
    конце — повторное сохранение .usdc дописывает новые массивы к старым, и файл раздувается."""
    dst.unlink(missing_ok=True)
    layer = Sdf.Layer.CreateNew(str(dst))
    layer.TransferContent(Sdf.Layer.FindOrOpen(str(src)))
    return Usd.Stage.Open(layer)


def split_geometry(src, z_cut):
    """Меш (треугольники по центру) или облако (точки) из src -> файлы _lower/_upper; число элементов."""
    counts = {}
    for part in PARTS:
        dst = src.with_name(f"{src.stem}_{part.lower()}.usdc")
        stage = copy_layer(src, dst)
        prim = next(p for p in stage.Traverse() if p.GetTypeName() in ("Mesh", "Points"))
        geo = UsdGeom.PointBased(prim)
        P = np.array(geo.GetPointsAttr().Get(), dtype=np.float64)
        pv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayColor")
        colors = np.array(pv.Get()) if pv and pv.HasValue() else None
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            F = np.array(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int64).reshape(-1, 3)
            zc = P[F, 2].mean(1)
            keep = zc <= z_cut if part == "Lower" else zc > z_cut
            used = np.unique(F[keep])
            remap = np.full(len(P), -1, np.int64)
            remap[used] = np.arange(len(used))
            mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(remap[F[keep]].ravel().astype(np.int32)))
            mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(int(keep.sum()), 3, np.int32)))
            normals = mesh.GetNormalsAttr().Get()
            if normals is not None and len(normals) == len(P):
                mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(np.array(normals, np.float32)[used]))
            counts[part] = int(keep.sum())
        else:
            used = np.flatnonzero(P[:, 2] <= z_cut if part == "Lower" else P[:, 2] > z_cut)
            widths = geo.GetPrim().GetAttribute("widths").Get()
            if widths is not None and len(widths) == len(P):
                geo.GetPrim().GetAttribute("widths").Set(Vt.FloatArray.FromNumpy(np.array(widths, np.float32)[used]))
            counts[part] = len(used)
        geo.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(P[used].astype(np.float32)))
        if colors is not None and len(colors) == len(P):
            pv.Set(Vt.Vec3fArray.FromNumpy(colors[used].astype(np.float32)))
        Q = P[used] if len(used) else np.zeros((1, 3))
        geo.GetExtentAttr().Set(Vt.Vec3fArray([Gf.Vec3f(*Q.min(0).tolist()), Gf.Vec3f(*Q.max(0).tolist())]))
        stage.GetRootLayer().Save()
    return counts


def add_part_geometry(stage, part, name, has_shell, cloud, invisible_mesh):
    """В /World/<part>: меш своей части (коллайдер; выключен, если коллайдер — оболочка) и облако (скрыто)."""
    rec = UsdGeom.Xform.Define(stage, f"/World/{part}/Reconstruction")
    rec.GetPrim().GetReferences().AddReference(f"./{name}_rko_mesh_{part.lower()}.usdc")
    if invisible_mesh:
        rec.MakeInvisible()
    if has_shell:
        UsdPhysics.CollisionAPI.Apply(stage.OverridePrim(f"/World/{part}/Reconstruction/mesh")).CreateCollisionEnabledAttr(False)
    if cloud:
        c = UsdGeom.Xform.Define(stage, f"/World/{part}/Cloud")
        c.GetPrim().GetReferences().AddReference(f"./{name}_rko_cloud_{part.lower()}.usdc")
        c.MakeInvisible()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", help="имя модели (сессия): scenes/reconstructed/<имя>_rko.usda")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--height", type=float, help="срез на этой высоте над полом, м")
    g.add_argument("--z", type=float, help="срез на этой абсолютной высоте, м")
    args = parser.parse_args()

    scene = OUT / f"{args.name}_rko.usda"
    mesh_asset = OUT / f"{args.name}_rko_mesh.usdc"
    if not scene.exists() or not mesh_asset.exists():
        raise SystemExit(f"нет {scene} или {mesh_asset} — сначала scripts/pipeline.sh {args.name}")
    floor = 0.0
    report = ROOT / "out" / "mesh" / f"{args.name}_rko" / "report.json"
    if report.exists():
        floor = json.loads(report.read_text(encoding="utf-8")).get("floor_z_m") or 0.0
    z_cut = args.z if args.z is not None else floor + args.height

    mesh_counts = split_geometry(mesh_asset, z_cut)
    cloud_asset = OUT / f"{args.name}_rko_cloud.usdc"
    cloud = cloud_asset.exists()
    cloud_counts = split_geometry(cloud_asset, z_cut) if cloud else {}

    # Сцена меша: части вместо целого.
    out_scene = OUT / f"{args.name}_rko_split.usda"
    stage = copy_layer(scene, out_scene)
    has_shell = stage.GetPrimAtPath("/World/Shell").IsValid()
    stage.RemovePrim("/World/Reconstruction")
    stage.RemovePrim("/World/Cloud")
    for part in PARTS:
        UsdGeom.Xform.Define(stage, f"/World/{part}")
        add_part_geometry(stage, part, args.name, has_shell, cloud, invisible_mesh=False)
    stage.GetRootLayer().Save()
    print(f"[split] {out_scene.name}: срез на z = {z_cut:.2f} м ({z_cut - floor:.2f} м над полом); "
          f"меш низ {mesh_counts['Lower']} / верх {mesh_counts['Upper']} треуг."
          + (f", облако {cloud_counts['Lower']} / {cloud_counts['Upper']} точек" if cloud else ""), flush=True)

    # Сцена гауссиан: та же резка, гауссианы — обрезкой NuRec по высоте.
    gs_scene = OUT / f"{args.name}_gs.usda"
    if gs_scene.exists():
        src = Usd.Stage.Open(str(gs_scene))
        refs = src.GetPrimAtPath("/World/Gaussians").GetMetadata("references")
        usdz = refs.GetAddedOrExplicitItems()[0].assetPath if refs else f"./{args.name}_gs.usdz"
        out_gs = OUT / f"{args.name}_gs_split.usda"
        stage = copy_layer(gs_scene, out_gs)
        has_shell = stage.GetPrimAtPath("/World/Shell").IsValid()
        for p in ("/World/Gaussians", "/World/Reconstruction", "/World/Cloud"):
            stage.RemovePrim(p)
        gs = UsdGeom.Xform.Define(stage, "/World/Gaussians")
        gs.GetPrim().GetReferences().AddReference(usdz)
        vol = next(p for p in Usd.PrimRange(gs.GetPrim()) if p.GetAttribute("omni:nurec:isNuRecVolume").IsValid())
        hi = Gf.Vec3f(vol.GetAttribute("omni:nurec:crop:maxBounds").Get())
        hi[2] = z_cut
        stage.OverridePrim(vol.GetPath()).CreateAttribute("omni:nurec:crop:maxBounds", Sdf.ValueTypeNames.Float3).Set(hi)
        for part in PARTS:
            UsdGeom.Xform.Define(stage, f"/World/{part}")
            add_part_geometry(stage, part, args.name, has_shell, cloud, invisible_mesh=True)
        stage.GetRootLayer().Save()
        print(f"[split] {out_gs.name}: гауссианы срезаны на z = {z_cut:.2f} м ({vol.GetPath()} -> "
              f"omni:nurec:crop:maxBounds), коллайдер и облако — по частям", flush=True)
    print("[split] в Isaac Sim: Stage -> /World/Upper -> правая кнопка -> Delete (или глаз — скрыть)", flush=True)


if __name__ == "__main__":
    main()
