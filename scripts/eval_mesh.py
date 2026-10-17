#!/usr/bin/env python3
"""Сравнивает меш (build_mesh.py) с настоящей геометрией комнаты (export_scene_mesh.py).

Точность — расстояние от поверхности меша до эталона. Полнота — какая доля
эталонной поверхности покрыта мешем; считаем только то, что лидар вообще видел
(эталонные точки ближе 3 см к облаку), иначе в счёт попадут обратные стороны стен
и оболочка сцены. Отдельно — какая доля пола покрыта мешем (вместе с местами, где лидар
пол не видел) в пределах 5 м от пути: на складе весь пол — сотни м², а проехали малую часть.
Картинка — вид сверху и сбоку, цвет — ошибка.

Запуск: .venv/bin/python scripts/eval_mesh.py out/mesh/walk01_rko
"""

import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import session_info  # noqa: E402
from build_mesh import floor_level  # noqa: E402
SEEN_M = 0.03
FLOOR_REACH_M = 5.0


def gt_floor_level(points, normals, sensor_z, cell=0.2):
    """Уровень пола по эталону: в каждой клетке — верхняя горизонталь нормалью вверх (ниже
    датчика на 0.2–3 м), затем самый частый уровень. Скрытая плита под полом (в Simple Room
    на −0.05 м, площадью больше пола) так не побеждает: сверху её закрывает пол."""
    up = (normals[:, 2] > 0.9) & (points[:, 2] < sensor_z - 0.2) & (points[:, 2] > sensor_z - 3.0)
    P = points[up]
    if not len(P):
        return None
    key = np.floor(P[:, :2] / cell).astype(np.int64) @ np.array([1_000_003, 1])
    order = np.lexsort((-P[:, 2], key))  # в каждой клетке — сначала верхняя
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    tops = P[order][first, 2]
    hist, edges = np.histogram(tops, bins=np.arange(sensor_z - 3.0, sensor_z - 0.2, 0.01))
    peak = edges[np.argmax(hist)] + 0.005
    return float(np.median(tops[np.abs(tops - peak) < 0.02]))


def normal_errors(mesh, gt, n=300_000, near=0.05):
    """Угол между поверхностью меша и эталона, град: точки меша с нормалями его треугольников ->
    ближайший треугольник эталона. Знак нормали не важен. Только точки ближе near к эталону
    (дальние — уже ошибка положения, не неровность). Бугры на ровном полу/стене дают большой
    угол, даже когда в среднем поверхность стоит на месте."""
    mesh.compute_triangle_normals()
    smp = mesh.sample_points_uniformly(n, use_triangle_normal=True)
    pts, nrm = np.asarray(smp.points), np.asarray(smp.normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
    q = scene.compute_closest_points(o3d.core.Tensor(pts.astype(np.float32)))
    d = np.linalg.norm(q["points"].numpy() - pts, axis=1)
    cos = np.abs(np.einsum("ij,ij->i", q["primitive_normals"].numpy(), nrm))
    return np.degrees(np.arccos(np.clip(cos[d < near], 0.0, 1.0)))


def distances(mesh, points):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene.compute_distance(o3d.core.Tensor(points.astype(np.float32))).numpy()


def error_view(points, err, axes, size=700, vmax=0.05):
    """Проекция точек на две оси; цвет от зелёного (0) до красного (vmax и больше)."""
    a, b = points[:, axes[0]], points[:, axes[1]]
    lo, hi = np.array([a.min(), b.min()]), np.array([a.max(), b.max()])
    scale = size / max(hi - lo)
    w, h = int((hi[0] - lo[0]) * scale) + 1, int((hi[1] - lo[1]) * scale) + 1
    order = np.argsort(err)  # большие ошибки рисуем поверх
    u = ((a[order] - lo[0]) * scale).astype(int)
    v = ((hi[1] - b[order]) * scale).astype(int)
    x = np.clip(err[order] / vmax, 0, 1)
    rgb = np.stack([255 * x, 255 * (1 - x) * 0.8 + 40, 40 * np.ones_like(x)], 1).astype(np.uint8)
    img = np.full((h, w, 3), 24, np.uint8)
    img[np.clip(v, 0, h - 1), np.clip(u, 0, w - 1)] = rgb
    return Image.fromarray(img)


def main():
    mdir = Path(sys.argv[1])
    gt_path = Path(sys.argv[sys.argv.index("--gt") + 1]) if "--gt" in sys.argv else session_info.gt_mesh(session_info.session_of(mdir))
    if gt_path is None or not gt_path.exists():
        print(f"[eval] эталона нет ({gt_path}) — оценка пропущена; выгрузить: scripts/export_scene_mesh.py --session ...")
        return
    mesh = o3d.io.read_triangle_mesh(str(mdir / "mesh.ply"))
    gt = o3d.io.read_triangle_mesh(str(gt_path))
    pcd = o3d.io.read_point_cloud(str(mdir / "cloud.ply"))
    cloud, cloud_n = np.asarray(pcd.points), np.asarray(pcd.normals)

    rec_pts = np.asarray(mesh.sample_points_uniformly(1_000_000).points)
    acc = distances(gt, rec_pts)
    nerr = normal_errors(mesh, gt)

    gt.compute_triangle_normals()
    gt_sample = gt.sample_points_uniformly(2_000_000, use_triangle_normal=True)
    gt_pts, gt_n = np.asarray(gt_sample.points), np.asarray(gt_sample.normals)
    seen = cKDTree(cloud).query(gt_pts, distance_upper_bound=SEEN_M, workers=-1)[0] < SEEN_M
    comp = distances(mesh, gt_pts[seen])

    # Пол — весь, а не только увиденный: роботу нужен сплошной. Уровень пола — по эталону под
    # путём датчика (gt_floor_level). Без пути — по облаку, как floor_level в build_mesh.py.
    gt_traj = ROOT / "data" / "sessions" / session_info.session_of(mdir) / "gt.npz"
    lo, hi = cloud.min(0)[:2], cloud.max(0)[:2]
    region = np.all((gt_pts[:, :2] > lo) & (gt_pts[:, :2] < hi), 1)
    floor_z = None
    if gt_traj.exists():
        path = np.load(gt_traj)["pos"]
        near = cKDTree(path[:, :2]).query(gt_pts[:, :2], workers=-1)[0]
        floor_z = gt_floor_level(gt_pts[near < 1.0], gt_n[near < 1.0], float(np.percentile(path[:, 2], 5)))
        region &= near < FLOOR_REACH_M
    if floor_z is None and len(cloud_n):
        floor_z = floor_level(cloud, cloud_n, float(cloud[:, 2].min()) + 1.0)
    if floor_z is None:
        floor_z = float(np.percentile(cloud[:, 2], 1))
    floor = region & (np.abs(gt_n[:, 2]) > 0.9) & (np.abs(gt_pts[:, 2] - floor_z) < 0.02)
    floor_cov = distances(mesh, gt_pts[floor])

    def share(x, d):
        return round(float(np.mean(x < d)), 4)

    prec5, rec5 = share(acc, 0.05), share(comp, 0.05)
    report = {
        "mesh": str(mdir),
        "gt": str(gt_path),
        "accuracy_median_mm": round(float(np.median(acc)) * 1000, 1),
        "normal_error_median_deg": round(float(np.median(nerr)), 2),
        "normal_error_mean_deg": round(float(np.mean(nerr)), 2),
        "normal_share_within_5deg": share(nerr, 5.0),
        "accuracy_p90_mm": round(float(np.percentile(acc, 90)) * 1000, 1),
        "accuracy_share_within_2cm": share(acc, 0.02),
        "accuracy_share_within_5cm": prec5,
        "seen_gt_area_share": round(float(seen.mean()), 3),
        "completeness_share_within_2cm": share(comp, 0.02),
        "completeness_share_within_5cm": rec5,
        "fscore_5cm": round(2 * prec5 * rec5 / max(prec5 + rec5, 1e-9), 4),
        "floor_z_m": round(floor_z, 4) + 0.0,  # без «-0.0»
        "floor_covered_share_within_5cm": share(floor_cov, 0.05),
    }
    top = error_view(rec_pts, acc, (0, 1))
    side = error_view(rec_pts, acc, (0, 2))
    canvas = Image.new("RGB", (max(top.width, side.width), top.height + side.height + 30), (24, 24, 28))
    canvas.paste(top, (0, 0))
    canvas.paste(side, (0, top.height + 30))
    ImageDraw.Draw(canvas).text((8, top.height + 8), "error to ground truth: green 0 -> red 5 cm+ (top: XY, bottom: XZ)",
                                fill=(220, 220, 220))
    canvas.save(mdir / "error_views.png")
    (mdir / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
