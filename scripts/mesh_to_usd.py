#!/usr/bin/env python3
"""Упрощает меш (build_mesh.py) и записывает его в USD для Isaac Sim — с коллайдером.

Если рядом есть mesh_colored.ply (colorize_mesh.py), берётся он, и цвета вершин идут
в displayColor и в свечение материала. Перевод — по кривой рендера Isaac
(config/isaac_color_lut.json, scripts/isaac_color_lut.py): снимки уже прошли его тональную
кривую (ACES), и при показе модель должна дать тот же пиксель, а не пройти кривую второй раз
(иначе серые стены выходят белёсыми). Без файла кривой — просто sRGB -> линейные.

scenes/reconstructed/<name>_mesh.usdc — только меш (/Reconstruction), его можно
подключать в любые сцены; <name>.usda — готовая сцена: меш, свет, физика и, если
export_cloud.py уже записал <name>_cloud.usdc, скрытый слой облака точек.
Коллайдер — треугольная сетка без упрощения (approximation "none"): годится для
неподвижного окружения, по которому ходят роботы. Если fit_shell.py уже сделал
оболочку, коллайдер в сцене — она (ровные пол, стены, потолок) плюс предметы из меша:
<name>_shell.usdc, /World/Shell, скрыт; у цветного меша столкновения выключены.
--collider mesh — по-старому, сам меш.

Запуск: .venv/bin/python scripts/mesh_to_usd.py out/mesh/walk01_rko [--triangles 500000]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt

ROOT = Path(__file__).resolve().parents[1]


LUT = ROOT / "config" / "isaac_color_lut.json"


def srgb_to_linear(srgb):
    """Цвета снимков (sRGB 0..1) -> линейные."""
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def lut_render_settings():
    """Тональная кривая, при которой мерилась кривая цвета, — в корневой слой сцены: Isaac
    применит её при открытии (иначе после сцены гауссиан с её op 2 цвета меша поплывут)."""
    if not LUT.exists():
        return {}
    measured = json.loads(LUT.read_text(encoding="utf-8")).get("render_settings", {})
    keys = {"/rtx/post/tonemap/op": "rtx:post:tonemap:op", "/rtx/post/histogram/enabled": "rtx:post:histogram:enabled"}
    return {usd_key: measured[k] for k, usd_key in keys.items() if measured.get(k) is not None}


def emissive_for(srgb):
    """Цвет снимка (sRGB 0..1) -> свечение, которое Isaac покажет тем же пикселем."""
    if not LUT.exists():
        return srgb_to_linear(srgb)
    lut = json.loads(LUT.read_text(encoding="utf-8"))
    pix, em = np.asarray(lut["pixel_srgb8"]), np.asarray(lut["emissive"])
    keep = np.r_[True, np.diff(pix) > 1e-6]  # только строго растущие точки
    return np.interp(np.asarray(srgb) * 255.0, pix[keep], em[keep])


def bind_vertex_color_material(stage, geo, root="/Reconstruction"):
    """Цвет вершин уже содержит освещение со снимков: показываем его как есть — свечением,
    без бликов, чтобы свет сцены не освещал модель второй раз."""
    mat = UsdShade.Material.Define(stage, f"{root}/Looks/VertexColor")
    reader = UsdShade.Shader.Define(stage, f"{root}/Looks/VertexColor/displayColor")
    reader.CreateIdAttr("UsdPrimvarReader_float3")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("displayColor")
    reader_out = reader.CreateOutput("result", Sdf.ValueTypeNames.Float3)
    surface = UsdShade.Shader.Define(stage, f"{root}/Looks/VertexColor/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(reader_out)
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    UsdShade.MaterialBindingAPI.Apply(geo.GetPrim()).Bind(mat)


def define_mesh(stage, path, mesh, color):
    """Сетка-коллайдер (треугольники как есть) с одним цветом — чтобы её было видно, если включить."""
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    geo = UsdGeom.Mesh.Define(stage, path)
    geo.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    geo.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    geo.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.astype(np.int32).ravel()))
    geo.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    geo.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    geo.CreateDoubleSidedAttr(True)
    geo.CreateExtentAttr([Gf.Vec3f(*v.min(0).tolist()), Gf.Vec3f(*v.max(0).tolist())])
    UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geo.GetPrim()).CreateApproximationAttr("none")
    return len(f)


def write_shell_asset(path, mdir):
    """Коллайдер из fit_shell.py: оболочка комнаты (ровные пол и стены), плинтус и карниз,
    предметы из меша (стол, светильник)."""
    shell = o3d.io.read_triangle_mesh(str(mdir / "shell.ply"))
    objects = o3d.io.read_triangle_mesh(str(mdir / "objects.ply"))
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Shell")
    stage.SetDefaultPrim(root.GetPrim())
    n_room = define_mesh(stage, "/Shell/room", shell, (0.55, 0.75, 1.0))
    if (mdir / "details.ply").exists():  # плинтус и карниз (fit_shell.py, room_details.py)
        n_room += define_mesh(stage, "/Shell/details", o3d.io.read_triangle_mesh(str(mdir / "details.ply")), (0.8, 0.88, 1.0))
    n_obj = define_mesh(stage, "/Shell/objects", objects, (1.0, 0.6, 0.2))
    root.GetPrim().SetCustomDataByKey("source", str(mdir))
    stage.GetRootLayer().Save()
    return n_room, n_obj


def add_cloud_layer(scene, cloud_usd):
    """Облако точек (export_cloud.py), если оно есть, — в сцену скрытым слоем /World/Cloud:
    включается глазом в Stage, чтобы сверить меш с тем, что мерил лидар."""
    if not cloud_usd.exists():
        return
    prim = UsdGeom.Xform.Define(scene, "/World/Cloud")
    prim.GetPrim().GetReferences().ClearReferences()
    prim.GetPrim().GetReferences().AddReference(f"./{cloud_usd.name}")
    prim.MakeInvisible()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir")
    parser.add_argument("--triangles", type=int, default=500_000)
    parser.add_argument("--name", help="по умолчанию — имя папки меша")
    parser.add_argument("--collider", choices=("auto", "shell", "mesh"), default="auto",
                        help="auto — оболочка fit_shell.py, если она есть, иначе сам меш")
    args = parser.parse_args()
    mdir = Path(args.mesh_dir)
    name = args.name or mdir.name
    out = ROOT / "scenes" / "reconstructed"
    out.mkdir(parents=True, exist_ok=True)

    colored = (mdir / "mesh_colored.ply").exists()
    mesh = o3d.io.read_triangle_mesh(str(mdir / ("mesh_colored.ply" if colored else "mesh.ply")))
    if len(mesh.triangles) > args.triangles:
        mesh = mesh.simplify_quadric_decimation(args.triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.triangles)
    n = np.asarray(mesh.vertex_normals)

    asset = out / f"{name}_mesh.usdc"
    stage = Usd.Stage.CreateNew(str(asset))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Reconstruction")
    stage.SetDefaultPrim(root.GetPrim())
    geo = UsdGeom.Mesh.Define(stage, "/Reconstruction/mesh")
    geo.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(v.astype(np.float32)))
    geo.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(f), 3, np.int32)))
    geo.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(f.astype(np.int32).ravel()))
    geo.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32)))
    geo.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    geo.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    if colored:
        color = geo.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex)
        color.Set(Vt.Vec3fArray.FromNumpy(emissive_for(np.asarray(mesh.vertex_colors)).astype(np.float32)))
        bind_vertex_color_material(stage, geo)
    else:
        geo.CreateDisplayColorAttr([Gf.Vec3f(0.7, 0.7, 0.72)])
    geo.CreateDoubleSidedAttr(True)
    lo, hi = v.min(0), v.max(0)
    geo.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    UsdPhysics.CollisionAPI.Apply(geo.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(geo.GetPrim()).CreateApproximationAttr("none")
    root.GetPrim().SetCustomDataByKey("source", str(mdir))
    stage.GetRootLayer().Save()

    scene_path = out / f"{name}.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(scene, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(scene, 1.0)
    scene.SetTimeCodesPerSecond(60)
    scene.SetStartTimeCode(0)
    scene.SetEndTimeCode(1000000)
    world = UsdGeom.Xform.Define(scene, "/World")
    scene.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(scene, "/World/Reconstruction").GetPrim().GetReferences().AddReference(f"./{asset.name}")
    UsdPhysics.Scene.Define(scene, "/World/PhysicsScene")
    # Свет только внутри: без неба дыры в модели (например, неувиденный пол) видны чёрными.
    center = (lo + hi) / 2
    light = UsdLux.SphereLight.Define(scene, "/World/RoomLight")
    light.CreateIntensityAttr(30000.0)
    light.CreateRadiusAttr(0.3)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], hi[2] - 0.6))
    add_cloud_layer(scene, out / f"{name}_cloud.usdc")
    has_shell = (mdir / "shell.ply").exists() and (mdir / "objects.ply").exists()
    if args.collider == "shell" and not has_shell:
        raise SystemExit(f"нет {mdir}/shell.ply — сначала scripts/fit_shell.py {mdir}")
    collider = "меш"
    if args.collider == "shell" or (args.collider == "auto" and has_shell):
        shell_asset = out / f"{name}_shell.usdc"
        n_room, n_obj = write_shell_asset(shell_asset, mdir)
        shell = UsdGeom.Xform.Define(scene, "/World/Shell")
        shell.GetPrim().GetReferences().AddReference(f"./{shell_asset.name}")
        shell.MakeInvisible()
        # Цветной меш — только картинка: столкновения у него выключены поверх ссылки.
        UsdPhysics.CollisionAPI.Apply(scene.OverridePrim("/World/Reconstruction/mesh")).CreateCollisionEnabledAttr(False)
        collider = f"оболочка {n_room} + предметы {n_obj} треуг. (/World/Shell, скрыт)"
    else:
        (out / f"{name}_shell.usdc").unlink(missing_ok=True)  # не оставлять коллайдер прошлого прогона
    if colored and lut_render_settings():
        scene.GetRootLayer().customLayerData = {"renderSettings": lut_render_settings()}
    scene.GetRootLayer().Save()
    print(f"[usd] {asset.name}: вершин {len(v)}, треугольников {len(f)}, {asset.stat().st_size / 1e6:.1f} МБ, "
          f"{'с цветом' if colored else 'без цвета'}; коллайдер: {collider}; сцена {scene_path}")


if __name__ == "__main__":
    main()
