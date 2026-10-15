#!/usr/bin/env python3
"""Строит меш комнаты из сессии: облако по позам RKO-LIO (или эталона) -> Poisson.

Каждая точка ставится в мир по позе на момент её выстрела (позы интерполируются
между отсчётами траектории) — это и есть deskew. Одометрия ведётся в odom; в мир её
переносит известная поза старта — тот же сдвиг, что /tf_static в session_to_bag.py,
коррекцией это не является.

Пол, которого лидар не видел (под столом, в тенях мебели), достраивается плоскостью —
см. fill_floor; --no-fill-floor отключает. У робота (в сессии есть legs.npz — высота лидара
над подошвой по кадрам) пол кладётся под путём и в просмотренном лидаром свободном
пространстве — см. floor_under_path: Mid-360 на спине стоит ровно и пола рядом не видит.

Пишет в out/mesh/<session>_<poses>/: cloud.ply (прорежено), floor_fill.ply (заплатки пола),
mesh.ply, report.json.

Запуск: .venv/bin/python scripts/build_mesh.py walk01 [--poses rko|gt] [--tum путь] [--no-fill-floor]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
BASE_S = 1_700_000_000


def latest_tum(session, refined=False):
    """Самая свежая траектория RKO-LIO: out/rko/<сессия>_N/<сессия>_tum_N[_refined].txt."""
    files = [f for f in (ROOT / "out" / "rko").glob(f"{session}_*/{session}_tum_*.txt")
             if f.stem.endswith("_refined") == refined]
    if not files:
        raise FileNotFoundError(f"нет траектории RKO-LIO для {session} в out/rko")
    return max(files, key=lambda f: f.stat().st_mtime)


def load_trajectory(session_dir, poses, tum):
    gt = np.load(session_dir / "gt.npz")
    if poses == "gt":
        return gt["t"], gt["pos"], Rotation.from_quat(gt["quat_xyzw"])
    est = np.loadtxt(tum)
    q0 = Rotation.from_quat(gt["quat_xyzw"][0])
    world_odom = Rotation.from_euler("z", q0.as_euler("ZYX")[0])
    return est[:, 0] - BASE_S, world_odom.apply(est[:, 1:4]) + gt["pos"][0], world_odom * Rotation.from_quat(est[:, 4:8])


def voxel_mean(points, origins, voxel):
    """Среднее по вокселю — и точек, и положений датчика, откуда их видели."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    out_p = np.zeros((len(counts), 3))
    out_o = np.zeros((len(counts), 3))
    for k in range(3):
        out_p[:, k] = np.bincount(inv, points[:, k]) / counts
        out_o[:, k] = np.bincount(inv, origins[:, k]) / counts
    return out_p, out_o, counts


def floor_level(points, normals, sensor_z):
    """Высота пола: самый населённый горизонтальный уровень под датчиком (ниже траектории
    датчика хотя бы на 0.2 м, но не глубже 3 м). Не «самый нижний слой»: на складе ниже пола
    бывают приямки и рампы."""
    flat = np.abs(normals[:, 2]) > 0.9
    z = points[flat, 2]
    top = sensor_z - 0.2
    hist, edges = np.histogram(z, bins=np.arange(sensor_z - 3.0, top, 0.01))
    if not hist.any():
        return None
    peak = edges[np.argmax(hist)] + 0.005
    on = flat & (np.abs(points[:, 2] - peak) < 0.05)
    return float(np.median(points[on, 2]))


def fill_floor(points, normals, voxel, z0, cell=0.05):
    """Точки-заплатки на плоскости пола там, где лидар его не видел (под столом, в тенях мебели).

    Пол — уровень z0 (floor_level). Дыра — пустая клетка
    сетки, со всех сторон окружённая увиденным полом: края комнаты и проёмы не трогаем.
    Без пола в коллайдере робот провалится, поэтому дыры закрываются допущением, что пол
    плоский; площадь заплаток пишется в отчёт.
    """
    on_floor = (np.abs(normals[:, 2]) > 0.9) & (np.abs(points[:, 2] - z0) < 0.05)
    xy = points[on_floor, :2]
    lo = xy.min(0)
    ij = np.floor((xy - lo) / cell).astype(int)
    seen = np.zeros(ij.max(0) + 1, bool)
    seen[ij[:, 0], ij[:, 1]] = True
    holes = binary_fill_holes(seen) & ~seen
    sub = (np.arange(round(cell / voxel)) + 0.5) * voxel  # заплатка той же плотности, что облако
    offsets = np.stack(np.meshgrid(sub, sub, indexing="ij"), -1).reshape(-1, 2)
    corners = np.argwhere(holes) * cell + lo
    fill_xy = (corners[:, None, :] + offsets[None]).reshape(-1, 2)
    return np.c_[fill_xy, np.full(len(fill_xy), z0)], float(holes.sum() * cell ** 2)


def path_floor_z(tt, pos, legs, window=1.0):
    """Высота пола под датчиком в каждой позе траектории: z датчика минус его высота над
    подошвой опорной ноги в тот же момент (legs.npz), медиана за window секунд — гасит
    рывки походки и запаздывание z одометрии."""
    raw = pos[:, 2] - np.interp(tt, legs["t"], legs["sensor_height_m"])
    lo = np.searchsorted(tt, tt - window / 2)
    hi = np.searchsorted(tt, tt + window / 2, side="right")
    return np.array([np.median(raw[a:b]) for a, b in zip(lo, hi)])


def floor_under_path(points, normals, ray_from, ray_to, path_xy, path_z, voxel, cell=0.1,
                     reach=12.0, body=0.4, seen_min=10):
    """Точки-заплатки пола для робота, у которого лидар пола почти не видит (Mid-360 на спине
    смотрит от −7° вниз и видит пол дальше ~4 м, да и то вскользь).

    Где пол: полоса шириной с робота вдоль пути (он по нему шёл) и всё, над чем прошли низкие
    лучи лидара (к точкам не выше 1 м над полом, не дальше reach м), плюс замкнутые этим дыры
    (под стеллажами и предметами). Высота клетки — пол ближайшей позы пути (path_z). Не кладём,
    где видна горизонталь заметно ниже пола (приямок) и где пол и так увиден (seen_min точек
    в клетке). Возвращает (точки, площадь м², высота пола — медиана)."""
    from scipy.spatial import cKDTree
    path = cKDTree(path_xy)
    lo = np.minimum(path_xy.min(0), ray_to[:, :2].min(0)) - cell
    hi = np.maximum(path_xy.max(0), ray_to[:, :2].max(0)) + cell
    shape = tuple(np.ceil((hi - lo) / cell).astype(int) + 1)

    def cells(xy):
        ij = np.floor((xy - lo) / cell).astype(int)
        return np.clip(ij[:, 0], 0, shape[0] - 1), np.clip(ij[:, 1], 0, shape[1] - 1)

    walked = np.zeros(shape, bool)
    walked[cells(path_xy)] = True
    walked = distance_transform_edt(~walked) * cell <= body

    # Лучи к низким точкам: клетки под ними — свободное пространство у пола.
    zf_end = path_z[path.query(ray_to[:, :2], workers=-1)[1]]
    low = (ray_to[:, 2] - zf_end < 1.0) & (ray_to[:, 2] - zf_end > -0.2)
    a, b = ray_from[low, :2], ray_to[low, :2]
    d = b - a
    length = np.linalg.norm(d, axis=1)
    keep = length > cell
    a, d, length = a[keep], d[keep] / length[keep, None], np.minimum(length[keep], reach)
    free = np.zeros(shape, bool)
    step = cell / 2
    for s in range(0, len(a), 20000):
        n = np.ceil(length[s:s + 20000] / step).astype(int)
        ray = np.repeat(np.arange(len(n)), n)
        along = (np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)) * step
        xy = a[s:s + 20000][ray] + d[s:s + 20000][ray] * along[:, None]
        free[cells(xy)] = True

    region = binary_fill_holes(walked | free)
    zf_pts = path_z[path.query(points[:, :2], workers=-1)[1]]
    flat = np.abs(normals[:, 2]) > 0.9
    # Приямок — много горизонтальных точек на 0.3 м ниже пола (меньше — это дальний пол, сдвинутый
    # дрейфом одометрии: 0.5° наклона на 20 м — уже 17 см).
    pit = np.zeros(shape, np.int32)
    np.add.at(pit, cells(points[flat & (points[:, 2] < zf_pts - 0.3), :2]), 1)
    region &= ~binary_dilation(pit >= 10, iterations=1)
    seen = np.zeros(shape, np.int32)
    np.add.at(seen, cells(points[flat & (np.abs(points[:, 2] - zf_pts) < 0.08), :2]), 1)
    region &= seen < seen_min

    corners = np.argwhere(region) * cell + lo
    sub = (np.arange(max(1, round(cell / 0.05))) + 0.5) * min(cell, 0.05)  # 5 см: мельче Poisson не различит
    offsets = np.stack(np.meshgrid(sub, sub, indexing="ij"), -1).reshape(-1, 2)
    fill_xy = (corners[:, None, :] + offsets[None]).reshape(-1, 2)
    z = path_z[path.query(fill_xy, workers=-1)[1]]
    return np.c_[fill_xy, z], float(region.sum() * cell ** 2), float(np.median(path_z))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--poses", choices=("rko", "gt"), default="rko")
    parser.add_argument("--tum", help="траектория RKO-LIO; по умолчанию последняя out/rko/<session>_*/")
    parser.add_argument("--voxel", type=float, default=0.01)
    parser.add_argument("--min-range", type=float, default=0.3)
    parser.add_argument("--max-range", type=float, default=40.0)
    parser.add_argument("--depth", type=int, default=10, help="глубина Poisson: 10 -> ячейка ~1 см на комнату 10 м")
    parser.add_argument("--normal-radius", type=float, default=0.10, help="радиус оценки нормалей, м (шире — устойчивее к толстым слоям)")
    parser.add_argument("--keep-dist", type=float, default=0.05, help="срезать поверхность дальше этого от точек облака, м")
    parser.add_argument("--no-fill-floor", action="store_true", help="не достраивать пол там, где его не видел лидар")
    args = parser.parse_args()

    sdir = ROOT / "data" / "sessions" / args.session
    tum = args.tum or latest_tum(args.session)
    refined = args.poses == "rko" and "refined" in Path(tum).name
    out = ROOT / "out" / "mesh" / (f"{args.session}_{args.poses}" + ("_refined" if refined else ""))
    out.mkdir(parents=True, exist_ok=True)

    lidar = np.load(sdir / "lidar.npz")
    p_s, t = lidar["points"].astype(np.float64), lidar["t"]
    tt, pos, rot = load_trajectory(sdir, args.poses, tum)
    rng = np.linalg.norm(p_s, axis=1)
    keep = (t >= tt[0]) & (t <= tt[-1]) & (rng >= args.min_range) & (rng <= args.max_range)
    p_s, t = p_s[keep], t[keep]

    slerp = Slerp(tt, rot)
    origins = np.c_[[np.interp(t, tt, pos[:, k]) for k in range(3)]].T
    world = np.empty_like(p_s)
    for a in range(0, len(t), 2_000_000):  # кусками, чтобы не держать 10 млн поворотов разом
        b = min(a + 2_000_000, len(t))
        world[a:b] = slerp(t[a:b]).apply(p_s[a:b]) + origins[a:b]

    # Лучи для пола под путём робота — выборка из всех (с настоящими точками выстрела).
    rays = np.random.default_rng(0).choice(len(t), size=min(len(t), 600_000), replace=False)
    ray_from, ray_to = origins[rays], world[rays]
    pts, org, counts = voxel_mean(world, origins, args.voxel)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, idx = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    org = org[idx]
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=args.normal_radius, max_nn=60))
    n = np.asarray(pcd.normals)
    flip = np.einsum("ij,ij->i", n, org - np.asarray(pcd.points)) < 0  # нормаль — к датчику
    n[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(n)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)  # только то, что видел лидар: по нему eval_mesh судит, что «увидено»

    floor_z = floor_filled = None
    floor_source = "lidar"
    fill_path = out / "floor_fill.ply"
    legs_path = sdir / "legs.npz"
    if not args.no_fill_floor and legs_path.exists():
        floor_source = "robot_path"
        zf = path_floor_z(tt, pos, np.load(legs_path))
        fill, floor_filled, floor_z = floor_under_path(np.asarray(pcd.points), n, ray_from, ray_to,
                                                       pos[:, :2], zf, args.voxel)
    elif not args.no_fill_floor:
        floor_z = floor_level(np.asarray(pcd.points), n, float(np.percentile(pos[:, 2], 5)))
        fill, floor_filled = fill_floor(np.asarray(pcd.points), n, args.voxel, floor_z) if floor_z is not None else (np.zeros((0, 3)), 0.0)
    if not args.no_fill_floor:
        patch = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(fill))
        patch.normals = o3d.utility.Vector3dVector(np.tile([0.0, 0.0, 1.0], (len(fill), 1)))
        o3d.io.write_point_cloud(str(fill_path), patch)
        pcd += patch
    elif fill_path.exists():
        fill_path.unlink()

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=args.depth)
    # Poisson замыкает поверхность и там, где точек не было; оставляем только то, что рядом с облаком.
    far = cKDTree(np.asarray(pcd.points)).query(np.asarray(mesh.vertices), workers=-1)[0] > args.keep_dist
    mesh.remove_vertices_by_mask(far)
    bbox = pcd.get_axis_aligned_bounding_box()
    bbox = o3d.geometry.AxisAlignedBoundingBox(bbox.min_bound - 0.05, bbox.max_bound + 0.05)
    mesh = mesh.crop(bbox)
    clusters, sizes, _ = mesh.cluster_connected_triangles()
    sizes = np.asarray(sizes)
    mesh.remove_triangles_by_mask(sizes[np.asarray(clusters)] < 200)  # мелкие обрывки
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(out / "mesh.ply"), mesh)

    report = {
        "session": args.session,
        "poses": args.poses,
        "trajectory": str(tum) if args.poses == "rko" else "gt.npz",
        "raw_points": int(keep.sum()),
        "voxel_m": args.voxel,
        "cloud_points": len(pcd.points),
        "mesh_vertices": len(mesh.vertices),
        "mesh_triangles": len(mesh.triangles),
        "poisson_depth": args.depth,
        "normal_radius_m": args.normal_radius,
        "keep_dist_m": args.keep_dist,
        "floor_z_m": None if floor_z is None else round(floor_z, 4),
        "floor_source": floor_source,
        "floor_filled_m2": None if floor_filled is None else round(floor_filled, 2),
        "bounds_m": [np.round(bbox.min_bound, 2).tolist(), np.round(bbox.max_bound, 2).tolist()],
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
