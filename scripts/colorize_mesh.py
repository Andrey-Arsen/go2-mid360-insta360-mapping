#!/usr/bin/env python3
"""Красит меш кадрами обоих fisheye-объективов (сессия record_session.py).

Поза объектива в момент съёмки = поза лидара по траектории * T_lidar_lens из meta.json.
Объектив — ftheta, равноугольный: угол от оси (камера USD смотрит по -Z, верх +Y)
= k1 * радиус в пикселях. Вершина берёт цвет кадра, только если её видно: луч от
объектива до неё не упирается в другую поверхность. Цвета всех кадров смешиваются
с весами — ближе к центру объектива, ближе к камере, взгляд не по касательной.
Если рядом с кадром есть NNN_<объектив>_mask.png (робот на снимке), его пиксели не берутся.

Меш сначала упрощается (--triangles), красится и пишется в mesh_colored.ply рядом;
mesh_to_usd.py возьмёт его вместо mesh.ply. С --check рисует проекцию модели поверх
снимка — проверка, что модель объектива и позы верны.

Запуск: .venv/bin/python scripts/colorize_mesh.py out/mesh/tilt30_rko tilt30 [--tum путь] [--check 10]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
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


def lidar_poses(sdir, tum, times):
    """Поза livox_frame в мире на заданные моменты: из TUM (odom -> мир по позе старта) или эталона."""
    gt = np.load(sdir / "gt.npz")
    if tum is None:
        tt, pos, rot = gt["t"], gt["pos"], Rotation.from_quat(gt["quat_xyzw"])
    else:
        est = np.loadtxt(tum)
        q0 = Rotation.from_quat(gt["quat_xyzw"][0])
        w = Rotation.from_euler("z", q0.as_euler("ZYX")[0])
        tt, pos, rot = est[:, 0] - BASE_S, w.apply(est[:, 1:4]) + gt["pos"][0], w * Rotation.from_quat(est[:, 4:8])
    times = np.clip(times, tt[0], tt[-1])
    r = Slerp(tt, rot)(times)
    p = np.c_[[np.interp(times, tt, pos[:, k]) for k in range(3)]].T
    out = np.tile(np.eye(4), (len(times), 1, 1))
    out[:, :3, :3] = r.as_matrix()
    out[:, :3, 3] = p
    return out


def project(points_lens, size, fov_deg):
    """Точки в системе объектива -> пиксели (u вправо, v вниз), угол от оси."""
    x, y, z = points_lens.T
    theta = np.arctan2(np.hypot(x, y), -z)
    half = size / 2.0
    r = theta / (math.radians(fov_deg / 2.0) / half)
    phi = np.arctan2(y, x)
    return np.c_[half + r * np.cos(phi), half - r * np.sin(phi)], theta


def bilinear(img, uv):
    h, w, _ = img.shape
    u = np.clip(uv[:, 0] - 0.5, 0, w - 1.001)
    v = np.clip(uv[:, 1] - 0.5, 0, h - 1.001)
    u0, v0 = u.astype(int), v.astype(int)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    return ((img[v0, u0] * (1 - du) + img[v0, u0 + 1] * du) * (1 - dv)
            + (img[v0 + 1, u0] * (1 - du) + img[v0 + 1, u0 + 1] * du) * dv)


def session_cameras(session, tum_arg):
    """meta.json сессии и позы лидара на моменты съёмки. tum_arg: None — последняя RKO-LIO, 'gt' — эталон."""
    sdir = ROOT / "data" / "sessions" / session
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    if tum_arg == "gt":
        tum = None
    elif tum_arg:
        tum = Path(tum_arg)
    else:
        tum = latest_tum(session)
    return sdir, meta, lidar_poses(sdir, tum, np.array(meta["shutter_times_s"])), tum


def views(V, scene, sdir, meta, T_wl, max_angle, shot_ids):
    """Для каждого кадра каждого объектива: какие точки видно и куда они падают на снимке.

    V — точки в мире, scene — RaycastingScene меша, по которому проверяется, не загорожена ли точка.
    """
    size, fov = meta["lens"]["size_px"], meta["lens"]["fov_deg"]
    for k in shot_ids:
        for lens, T_ll in meta["T_lidar_lens"].items():
            T_wc = T_wl[k] @ np.array(T_ll)
            center = T_wc[:3, 3]
            pl = (V - center) @ T_wc[:3, :3]  # в систему объектива
            uv, theta = project(pl, size, fov)
            dist = np.linalg.norm(V - center, axis=1)
            ok = (theta < math.radians(max_angle)) & (uv[:, 0] >= 0) & (uv[:, 0] < size) & (uv[:, 1] >= 0) & (uv[:, 1] < size)
            idx = np.nonzero(ok)[0]
            dirs = (V[idx] - center) / dist[idx, None]
            rays = np.c_[np.tile(center, (len(idx), 1)), dirs].astype(np.float32)
            hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
            # Ничего не загораживает. Допуск растёт с дальностью: на взгляде по касательной
            # луч задевает соседние треугольники той же неровной поверхности.
            vis = idx[hit > dist[idx] - (0.02 + 0.02 * dist[idx])]
            # Маска кадра (render_drive.py: 255 — сцена, 0 — сам робот): тело робота не красит то, что за ним.
            mpath = sdir / "images" / f"{k:03d}_{lens}_mask.png"
            if mpath.exists():
                keep = np.asarray(Image.open(mpath).convert("L")) > 127
                pix = np.clip(uv[vis].astype(int), 0, size - 1)
                vis = vis[keep[pix[:, 1], pix[:, 0]]]
            img = np.asarray(Image.open(sdir / "images" / f"{k:03d}_{lens}.png").convert("RGB"), dtype=np.float64)
            yield k, lens, img, center, vis, uv[vis], theta[vis], dist[vis]


def blend_colors(V, N, scene, sdir, meta, T_wl, max_angle=95.0):
    """Цвет точки — смесь всех кадров, где её видно: ближе к центру объектива, ближе к камере,
    взгляд не по касательной. Возвращает цвета 0..1 (невидимые — серые 0.5) и маску окрашенных."""
    acc = np.zeros((len(V), 3))
    wsum = np.zeros(len(V))
    for _, _, img, center, vis, uv, theta, dist in views(V, scene, sdir, meta, T_wl, max_angle,
                                                         range(len(meta["shutter_times_s"]))):
        cos_view = np.abs(np.einsum("ij,ij->i", N[vis], (center - V[vis]) / dist[:, None]))
        w = (1.0 - theta / math.radians(max_angle)) * cos_view / np.maximum(dist, 0.5) ** 2
        acc[vis] += w[:, None] * bilinear(img, uv)
        wsum[vis] += w
    colored = wsum > 0
    colors = np.full((len(V), 3), 0.5)
    colors[colored] = acc[colored] / wsum[colored, None] / 255.0
    return colors, colored


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir")
    parser.add_argument("session")
    parser.add_argument("--tum", help="траектория; по умолчанию последняя RKO-LIO (та же, что у меша); 'gt' — эталон")
    parser.add_argument("--triangles", type=int, default=500_000)
    parser.add_argument("--max-angle", type=float, default=95.0, help="не брать цвет дальше этого угла от оси, град")
    parser.add_argument("--check", type=int, help="только нарисовать проекцию на кадр с этим номером")
    args = parser.parse_args()

    mdir = Path(args.mesh_dir)
    sdir, meta, T_wl, tum = session_cameras(args.session, args.tum)

    mesh = o3d.io.read_triangle_mesh(str(mdir / "mesh.ply"))
    if len(mesh.triangles) > args.triangles:
        mesh = mesh.simplify_quadric_decimation(args.triangles)
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    V = np.asarray(mesh.vertices)
    N = np.asarray(mesh.vertex_normals)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    if args.check is not None:
        for k, lens, img, _, vis, uv, _, _ in views(V, scene, sdir, meta, T_wl, args.max_angle, [args.check]):
            over = img.copy()
            up = np.abs(N[vis, 2]) > 0.8
            dots = np.where(up[:, None], [255, 60, 60], [60, 160, 255])  # горизонтальное — красным, стены — синим
            u, v = uv.astype(int).T
            over[v, u] = 0.4 * over[v, u] + 0.6 * dots
            Image.fromarray(np.hstack([img, over]).astype(np.uint8)).resize((1920, 960)).save(mdir / f"check_{k:03d}_{lens}.png")
        print(f"[color] проверка: {mdir}/check_{args.check:03d}_*.png")
        return

    colors, colored = blend_colors(V, N, scene, sdir, meta, T_wl, args.max_angle)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_triangle_mesh(str(mdir / "mesh_colored.ply"), mesh)
    print(f"[color] {mdir / 'mesh_colored.ply'}: вершин {len(V)}, окрашено {colored.mean():.1%}, "
          f"кадров {len(meta['shutter_times_s'])} x {len(meta['T_lidar_lens'])}, траектория {tum or 'эталон'}")


if __name__ == "__main__":
    main()
