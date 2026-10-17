#!/usr/bin/env python3
"""Уточняет траекторию после RKO-LIO: каждый скан стыкуется с картой из точек, снятых в ДРУГОЕ
время, — так исчезают «двойные стены» от повторных проходов и дрожь поз между проходами.

Зачем: от точности поз зависит всё после съёмки. На sklad03 с эталонными позами меш точнее в
3.6 раза (4.0 против 14.6 мм), а гауссианы на отложенных кадрах — 32.3 против 24.3 дБ.

Как (несколько проходов):
1. Точки всех сканов ставятся в мир по текущим позам (на время каждой точки — deskew).
2. Карта: воксели 4 см отдельно по отрезкам времени 2 с (проходы не смешиваются), нормали.
3. Каждый скан (~4000 точек) — ICP «точка к плоскости» (Гаусс–Ньютон, вес Хьюбера) к точкам
   карты, снятым не ближе --exclude секунд от него: скан не тянется сам к себе, а сходится с
   тем, что видно с других мест и в другое время.
4. Поправки сканов сглаживаются по времени и переносятся на позы траектории; карта
   пересобирается.
Работает в системе одометрии (эталон не нужен) и пишет <траектория>_ref.txt рядом (не «_refined»:
так её как самую свежую берут и остальные скрипты).

Проверено на sklad03: ошибка траектории после совмещения 2.8 -> 1.1 см (6 проходов, ~8 мин), меш
точнее, гауссианы +1.5 дБ.

Запуск: .venv/bin/python scripts/refine_trajectory.py <сессия> <tum> [--iters 6] [--exclude 3]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
BASE_S = 1_700_000_000


def place(pts, t, tt, pos, rot):
    """Точки в кадре лидара -> мир по позе на время каждой точки."""
    world = np.empty_like(pts)
    slerp = Slerp(tt, rot)
    for a in range(0, len(t), 2_000_000):
        b = min(a + 2_000_000, len(t))
        o = np.c_[[np.interp(t[a:b], tt, pos[:, k]) for k in range(3)]].T
        world[a:b] = slerp(t[a:b]).apply(pts[a:b]) + o
    return world


def build_map(world, t, voxel, tbin):
    """Средние точки вокселей отдельно по отрезкам времени (проходы не смешиваются) + нормали."""
    origin = world.min(0)
    key = np.floor((world - origin) / voxel).astype(np.int64)
    dims = key.max(0) + 1
    code = ((key[:, 0] * dims[1] + key[:, 1]) * dims[2] + key[:, 2]) * 100_000 + np.floor(t / tbin).astype(np.int64)
    _, inv, cnt = np.unique(code, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    m = len(cnt)
    mp = np.stack([np.bincount(inv, world[:, k], minlength=m) for k in range(3)], 1) / cnt[:, None]
    mt = np.bincount(inv, t, minlength=m) / cnt
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mp))
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=3 * voxel, max_nn=30))
    return mp, mt, np.asarray(pcd.normals)


def register_scan(p, c, t_scan, tree, mp, mt, mn, exclude, iters=6, max_dist=0.25, huber=0.03):
    """ICP точка-к-плоскости скана p (мир, текущие позы) к карте без своего времени.
    Возвращает (поворот 3x3, сдвиг 3) вокруг точки c и число соответствий."""
    R_acc, t_acc = np.eye(3), np.zeros(3)
    cur = p.copy()
    n_ok = 0
    for _ in range(iters):
        d, idx = tree.query(cur, k=8, distance_upper_bound=max_dist)
        valid = np.isfinite(d)
        idx = np.where(valid, idx, 0)
        valid &= np.abs(mt[idx] - t_scan) >= exclude
        first = np.argmax(valid, axis=1)
        has = valid[np.arange(len(cur)), first]
        j = idx[np.arange(len(cur)), first][has]
        src, q, n = cur[has], mp[j], mn[j]
        n_ok = len(src)
        if n_ok < 150:
            return np.eye(3), np.zeros(3), 0
        r = np.einsum("ij,ij->i", n, src - q)
        w = np.where(np.abs(r) <= huber, 1.0, huber / np.maximum(np.abs(r), 1e-9))
        J = np.c_[np.cross(src - c, n), n]
        H = (J * w[:, None]).T @ J + 1e-6 * np.eye(6)
        delta = np.linalg.solve(H, -(J * w[:, None]).T @ r)
        dR = Rotation.from_rotvec(delta[:3]).as_matrix()
        cur = (cur - c) @ dR.T + c + delta[3:]
        R_acc, t_acc = dR @ R_acc, dR @ t_acc + delta[3:]
        if np.linalg.norm(delta[:3]) < 1e-5 and np.linalg.norm(delta[3:]) < 1e-4:
            break
    return R_acc, t_acc, n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("tum")
    parser.add_argument("--iters", type=int, default=6, help="проходов (карта пересобирается после каждого)")
    parser.add_argument("--exclude", type=float, default=3.0, help="не стыковать скан с точками ближе этого по времени, с")
    parser.add_argument("--voxel", type=float, default=0.04, help="воксель карты, м")
    parser.add_argument("--scan-points", type=int, default=4000, help="точек скана для стыковки")
    parser.add_argument("--smooth", type=float, default=0.05,
                        help="сглаживание поправок по времени, с (сигма); мелкое — чтобы ловить качание шага 2–3 Гц")
    parser.add_argument("--max-range", type=float, default=40.0)
    args = parser.parse_args()

    sdir = ROOT / "data" / "sessions" / args.session
    tum = Path(args.tum)
    est = np.loadtxt(tum)
    tt, pos, rot = est[:, 0] - BASE_S, est[:, 1:4].copy(), Rotation.from_quat(est[:, 4:8])
    lidar = np.load(sdir / "lidar.npz")
    pts, t, starts = lidar["points"].astype(np.float64), lidar["t"], lidar["scan_start"]
    scan_id = np.zeros(len(t), np.int64)
    scan_id[starts[1:]] = 1
    scan_id = np.cumsum(scan_id)
    rng = np.linalg.norm(pts, axis=1)
    keep = (t >= tt[0]) & (t <= tt[-1]) & (rng > 0.3) & (rng < args.max_range)
    pts, t, scan_id = pts[keep], t[keep], scan_id[keep]
    gen = np.random.default_rng(0)
    sample = {}
    for s in np.unique(scan_id):
        ids = np.flatnonzero(scan_id == s)
        sample[s] = ids if len(ids) <= args.scan_points else np.sort(gen.choice(ids, args.scan_points, replace=False))
    map_sel = gen.random(len(t)) < min(1.0, 8_000_000 / len(t))  # карта — до 8 млн точек

    t0 = time.time()
    history = []
    for it in range(args.iters):
        world = place(pts, t, tt, pos, rot)
        mp, mt, mn = build_map(world[map_sel], t[map_sel], args.voxel, 2.0)
        tree = cKDTree(mp)
        t_mid, rv, tr, ok = [], [], [], []
        slerp = Slerp(tt, rot)
        for s, ids in sample.items():
            tm = float(np.median(t[ids]))
            c = np.array([np.interp(tm, tt, pos[:, k]) for k in range(3)])
            R, tv, n_ok = register_scan(world[ids], c, tm, tree, mp, mt, mn, args.exclude)
            t_mid.append(tm)
            rv.append(Rotation.from_matrix(R).as_rotvec())
            tr.append(tv)
            ok.append(n_ok > 0)
        t_mid, rv, tr, ok = np.array(t_mid), np.array(rv), np.array(tr), np.array(ok)
        # Сглаживание по времени: средний шаг сканов 0.1 с; пропущенные (мало соответствий) — нули с весом 0.
        sig = args.smooth / np.median(np.diff(t_mid))
        wsum = gaussian_filter1d(ok.astype(float), sig)
        rv_s = gaussian_filter1d(rv * ok[:, None], sig, axis=0) / np.maximum(wsum, 1e-6)[:, None]
        tr_s = gaussian_filter1d(tr * ok[:, None], sig, axis=0) / np.maximum(wsum, 1e-6)[:, None]
        rv_t = np.c_[[np.interp(tt, t_mid, rv_s[:, k]) for k in range(3)]].T
        tr_t = np.c_[[np.interp(tt, t_mid, tr_s[:, k]) for k in range(3)]].T
        rot = Rotation.from_rotvec(rv_t) * rot  # поворот вокруг положения датчика -> положение не меняется
        pos = pos + tr_t
        step = {"pass": it + 1, "scans_registered": int(ok.sum()), "scans": len(ok),
                "shift_mean_cm": round(float(np.linalg.norm(tr_s[ok], axis=1).mean() * 100), 2),
                "rot_mean_deg": round(float(np.degrees(np.linalg.norm(rv_s[ok], axis=1)).mean()), 3),
                "map_points": len(mp), "elapsed_s": round(time.time() - t0)}
        history.append(step)
        print(f"[refine] проход {it + 1}: сканов состыковано {step['scans_registered']}/{step['scans']}, поправки средн "
              f"{step['shift_mean_cm']:.2f} см, {step['rot_mean_deg']:.3f}°; карта {len(mp)} точек; {step['elapsed_s']} с", flush=True)

    out = tum.with_name(tum.stem + "_ref.txt")
    res = est.copy()
    res[:, 1:4] = pos
    res[:, 4:8] = rot.as_quat()
    np.savetxt(out, res, fmt="%.9f")
    (out.with_suffix(".json")).write_text(json.dumps({"tum": str(tum), "passes": history}, ensure_ascii=False, indent=1) + "\n")
    print(f"[refine] {out}", flush=True)


if __name__ == "__main__":
    main()
