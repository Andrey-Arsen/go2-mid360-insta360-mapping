#!/usr/bin/env python3
"""Облако точек сессии — в LAS/LAZ (CloudCompare, QGIS, Potree) и в USD (Isaac Sim).

Берётся cloud.ply из build_mesh.py: только то, что реально мерил лидар (воксель 1 см,
без выбросов и без заплаток пола), в мировых координатах — тех же, что у меша.
Каждая точка красится кадрами обоих объективов (colorize_mesh.blend_colors, загороженность
проверяется по мешу); точки, которых не видела ни одна камера, берут цвет ближайшей
окрашенной. Пол — класс 2 (ground) по ASPRS, остальное — 1 (unclassified).

Isaac Sim LAS не открывает (в нём есть импорт E57/PTS через omni.kit.pointclouds), поэтому
рядом пишется готовый USD — UsdGeom.Points с тем же цветом, прорежённый до --usd-voxel;
в сцену <name>.usda он подключается скрытым слоем /World/Cloud.

Пишет <mesh_dir>/cloud.laz (или .las) и scenes/reconstructed/<name>_cloud.usdc.

Запуск: .venv/bin/python scripts/export_cloud.py out/mesh/tilt30_rko tilt30 [--tum путь] [--format las]
"""

import argparse
import json
import sys
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d
from pxr import Gf, Usd, UsdGeom, Vt
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colorize_mesh import blend_colors, session_cameras  # noqa: E402
from mesh_to_usd import add_cloud_layer, bind_vertex_color_material, emissive_for  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUND, UNCLASSIFIED = 2, 1


def write_las(path, points, colors, classes):
    header = laspy.LasHeader(point_format=2, version="1.2")  # xyz, интенсивность, класс, RGB — читают все
    header.offsets = np.floor(points.min(0))
    header.scales = np.full(3, 0.0001)  # 0.1 мм
    las = laspy.LasData(header)
    las.x, las.y, las.z = points.T
    las.red, las.green, las.blue = np.round(colors * 65535).astype(np.uint16).T
    las.classification = classes
    las.write(str(path))  # .laz сжимается сам (lazrs)


def write_usd(path, points, colors, width):
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Cloud")
    stage.SetDefaultPrim(root.GetPrim())
    geo = UsdGeom.Points.Define(stage, "/Cloud/points")
    geo.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    geo.CreateWidthsAttr(Vt.FloatArray([width]))
    geo.SetWidthsInterpolation(UsdGeom.Tokens.constant)
    geo.CreateDisplayColorPrimvar(UsdGeom.Tokens.vertex).Set(
        Vt.Vec3fArray.FromNumpy(emissive_for(colors).astype(np.float32)))
    lo, hi = points.min(0), points.max(0)
    geo.CreateExtentAttr([Gf.Vec3f(*lo.tolist()), Gf.Vec3f(*hi.tolist())])
    bind_vertex_color_material(stage, geo, root="/Cloud")
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir")
    parser.add_argument("session")
    parser.add_argument("--tum", help="траектория для поз камер; по умолчанию последняя RKO-LIO; 'gt' — эталон")
    parser.add_argument("--format", choices=("laz", "las"), default="laz")
    parser.add_argument("--usd-voxel", type=float, default=0.02, help="прореживание для Isaac Sim, м")
    parser.add_argument("--name", help="имя сцены в scenes/reconstructed; по умолчанию — имя папки меша")
    args = parser.parse_args()

    mdir = Path(args.mesh_dir)
    name = args.name or mdir.name
    pcd = o3d.io.read_point_cloud(str(mdir / "cloud.ply"))
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)

    # Загороженность — по упрощённому мешу (тот же, что красит colorize_mesh.py).
    colored_mesh = mdir / "mesh_colored.ply"
    mesh = o3d.io.read_triangle_mesh(str(colored_mesh if colored_mesh.exists() else mdir / "mesh.ply"))
    if len(mesh.triangles) > 500_000:
        mesh = mesh.simplify_quadric_decimation(500_000)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    sdir, meta, T_wl, tum = session_cameras(args.session, args.tum)
    colors, colored = blend_colors(P, N, scene, sdir, meta, T_wl)
    if colored.any() and not colored.all():
        nearest = cKDTree(P[colored]).query(P[~colored], workers=-1)[1]
        colors[~colored] = colors[colored][nearest]

    report = json.loads((mdir / "report.json").read_text(encoding="utf-8"))
    floor_z = report.get("floor_z_m")
    classes = np.full(len(P), UNCLASSIFIED, np.uint8)
    if floor_z is not None:
        classes[(np.abs(P[:, 2] - floor_z) < 0.05) & (np.abs(N[:, 2]) > 0.9)] = GROUND

    las_path = mdir / f"cloud.{args.format}"
    write_las(las_path, P, colors, classes)

    pcd.colors = o3d.utility.Vector3dVector(colors)
    small = pcd.voxel_down_sample(args.usd_voxel)
    usd_dir = ROOT / "scenes" / "reconstructed"
    usd_path = usd_dir / f"{name}_cloud.usdc"
    write_usd(usd_path, np.asarray(small.points), np.asarray(small.colors), args.usd_voxel)
    scene_path = usd_dir / f"{name}.usda"
    if scene_path.exists():
        stage = Usd.Stage.Open(str(scene_path))
        add_cloud_layer(stage, usd_path)
        stage.GetRootLayer().Save()

    print(f"[cloud] {las_path}: точек {len(P)}, видели камеры {colored.mean():.1%}, пол {np.mean(classes == GROUND):.1%}, "
          f"{las_path.stat().st_size / 1e6:.0f} МБ; {usd_path.name}: точек {len(small.points)} "
          f"(воксель {args.usd_voxel * 100:.0f} см), {usd_path.stat().st_size / 1e6:.0f} МБ; позы {tum or 'эталон'}")


if __name__ == "__main__":
    main()
