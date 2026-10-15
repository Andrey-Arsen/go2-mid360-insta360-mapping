#!/usr/bin/env python3
"""Уточняет траекторию RKO-LIO графом поз по ICP между подкартами — убирает медленный дрейф.

1. Запись режется на подкарты по --window секунд; точки ставятся по позам RKO-LIO
   (на время каждой точки) и переводятся в систему подкарты.
2. Рёбра графа: соседние подкарты и все пары, чьи центры ближе --radius (в том числе
   начало и конец петли). Для каждой пары — ICP точка-к-плоскости от позы RKO-LIO,
   грубо и точно; слабые совпадения отбрасываются.
3. Open3D оптимизирует граф (Левенберг–Марквардт), первая подкарта закреплена.
4. Поправка каждой подкарты интерполируется по времени на все позы траектории.
5. Затем каждый скан отдельно стыкуется ICP с общей картой (--scan-iters раз, карта
   пересобирается с новыми позами). Это ловит быстрые ошибки — например, качание
   в такт шагам, которое RKO-LIO по высоте отслеживает плохо, — подкарты их не видят.

Пишет <tum>_refined.txt рядом с исходной траекторией и печатает сводку.

Запуск: .venv/bin/python scripts/refine_poses.py walk01 [--tum путь]
"""

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
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
reg = o3d.pipelines.registration


def to_matrix(pos, rot):
    m = np.eye(4)
    m[:3, :3] = rot.as_matrix()
    m[:3, 3] = pos
    return m


def submap(points_s, t, slerp, tt, pos, t_ref, voxel):
    """Точки окна в мире по позам на время точек -> в систему позы на t_ref."""
    origin = np.c_[[np.interp(t, tt, pos[:, k]) for k in range(3)]].T
    world = slerp(t).apply(points_s) + origin
    ref = to_matrix(np.array([np.interp(t_ref, tt, pos[:, k]) for k in range(3)]), slerp([t_ref])[0])
    local = (world - ref[:3, 3]) @ ref[:3, :3]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(local)).voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return pcd, ref


def icp(source, target, init, voxel):
    result = None
    for dist in (4 * voxel, 1.5 * voxel):  # грубо, потом точно
        result = reg.registration_icp(source, target, dist, init if result is None else result.transformation,
                                      reg.TransformationEstimationPointToPlane(),
                                      reg.ICPConvergenceCriteria(max_iteration=60))
    info = reg.get_information_matrix_from_point_clouds(source, target, 1.5 * voxel, result.transformation)
    return result, info


def scan_to_map(pts, t, scan_id, tt, pos, rot, voxel, it):
    """Каждый скан (с deskew по текущим позам) — ICP к общей карте; поправки на время скана."""
    slerp = Slerp(tt, rot)
    origin = np.c_[[np.interp(t, tt, pos[:, k]) for k in range(3)]].T
    world = slerp(t).apply(pts) + origin
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world)).voxel_down_sample(voxel)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    t_mid, corr = [], []
    for s in np.unique(scan_id):
        sel = scan_id == s
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world[sel])).voxel_down_sample(voxel)
        result = None
        for dist in (3 * voxel, voxel):
            result = reg.registration_icp(src, target, dist, np.eye(4) if result is None else result.transformation,
                                          reg.TransformationEstimationPointToPlane(),
                                          reg.ICPConvergenceCriteria(max_iteration=30))
        if result.fitness > 0.5:
            t_mid.append(t[sel].mean())
            corr.append(result.transformation)
    corr = np.array(corr)
    c_rot = Rotation.from_matrix(corr[:, :3, :3])
    tq = np.clip(tt, t_mid[0], t_mid[-1])
    r = Slerp(t_mid, c_rot)(tq)
    p = np.c_[[np.interp(tq, t_mid, corr[:, k, 3]) for k in range(3)]].T
    shift = np.linalg.norm(corr[:, :3, 3], axis=1)
    print(f"[refine] скан-к-карте, проход {it + 1}: сканов {len(t_mid)}, поправки средн {shift.mean() * 100:.1f} см, "
          f"до {shift.max() * 100:.1f} см", flush=True)
    return r * rot, r.apply(pos) + p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--tum")
    parser.add_argument("--window", type=float, default=2.0, help="длина подкарты, с")
    parser.add_argument("--radius", type=float, default=3.0, help="искать пары подкарт ближе, м")
    parser.add_argument("--voxel", type=float, default=0.05)
    parser.add_argument("--min-fitness", type=float, default=0.3)
    parser.add_argument("--scan-iters", type=int, default=2, help="проходов скан-к-карте; 0 — только граф подкарт")
    parser.add_argument("--map-voxel", type=float, default=0.03)
    args = parser.parse_args()

    sdir = ROOT / "data" / "sessions" / args.session
    tum = Path(args.tum) if args.tum else latest_tum(args.session)
    est = np.loadtxt(tum)
    tt, pos, rot = est[:, 0] - BASE_S, est[:, 1:4], Rotation.from_quat(est[:, 4:8])
    slerp = Slerp(tt, rot)

    lidar = np.load(sdir / "lidar.npz")
    pts, t = lidar["points"].astype(np.float64), lidar["t"]
    scan_id = np.searchsorted(lidar["scan_start"], np.arange(len(t)), side="right") - 1
    keep = (t >= tt[0]) & (t <= tt[-1]) & (np.linalg.norm(pts, axis=1) > 0.3)
    pts, t, scan_id = pts[keep], t[keep], scan_id[keep]

    edges_t = np.arange(tt[0], tt[-1] + 1e-9, args.window)
    maps, refs, t_refs = [], [], []
    for a, b in zip(edges_t[:-1], edges_t[1:]):
        sel = (t >= a) & (t < b)
        if sel.sum() < 5000:
            continue
        t_ref = 0.5 * (a + b)
        pcd, ref = submap(pts[sel], t[sel], slerp, tt, pos, t_ref, args.voxel)
        maps.append(pcd)
        refs.append(ref)
        t_refs.append(t_ref)
    n = len(maps)

    graph = reg.PoseGraph()
    for ref in refs:
        graph.nodes.append(reg.PoseGraphNode(ref))
    centers = np.array([r[:3, 3] for r in refs])
    kept = rejected = 0
    for i in range(n):
        for j in range(i + 1, n):
            neighbour = j == i + 1
            if not neighbour and np.linalg.norm(centers[i] - centers[j]) > args.radius:
                continue
            init = np.linalg.inv(refs[j]) @ refs[i]  # i -> j по позам RKO-LIO
            result, info = icp(maps[i], maps[j], init, args.voxel)
            if result.fitness < args.min_fitness and not neighbour:
                rejected += 1
                continue
            # Ребро i->j; в Open3D source = i, target = j, transformation переводит i в систему j.
            graph.edges.append(reg.PoseGraphEdge(i, j, result.transformation, info, uncertain=not neighbour))
            kept += 1

    option = reg.GlobalOptimizationOption(max_correspondence_distance=1.5 * args.voxel, edge_prune_threshold=0.25,
                                          reference_node=0)
    reg.global_optimization(graph, reg.GlobalOptimizationLevenbergMarquardt(),
                            reg.GlobalOptimizationConvergenceCriteria(), option)
    # После оптимизации node.pose — поза подкарты в мире; поправка = новая * старая^-1.
    corr = [graph.nodes[k].pose @ np.linalg.inv(refs[k]) for k in range(n)]
    corr_rot = Rotation.from_matrix(np.array([c[:3, :3] for c in corr]))
    corr_pos = np.array([c[:3, 3] for c in corr])
    tq = np.clip(tt, t_refs[0], t_refs[-1])
    c_rot = Slerp(t_refs, corr_rot)(tq)
    c_pos = np.c_[[np.interp(tq, t_refs, corr_pos[:, k]) for k in range(3)]].T
    new_rot = c_rot * rot
    new_pos = c_rot.apply(pos) + c_pos
    shift = np.linalg.norm(corr_pos, axis=1)
    print(f"[refine] подкарт {n}, рёбер {kept} (отброшено {rejected}); поправки: сдвиг до {shift.max() * 100:.1f} см, "
          f"поворот до {np.degrees(corr_rot.magnitude()).max():.2f}°")

    for it in range(args.scan_iters):
        new_rot, new_pos = scan_to_map(pts, t, scan_id, tt, new_pos, new_rot, args.map_voxel, it)

    out = tum.with_name(tum.stem + "_refined.txt")
    np.savetxt(out, np.c_[est[:, 0], new_pos, new_rot.as_quat()], fmt="%.6f")
    print(f"[refine] {out}")


if __name__ == "__main__":
    main()
