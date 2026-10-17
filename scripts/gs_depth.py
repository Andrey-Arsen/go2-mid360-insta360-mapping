#!/usr/bin/env python3
"""Глубина для обучения гауссиан: для каждого кадра набора COLMAP (session_to_colmap.py) лучи
той же fisheye-модели из позы камеры пускаются в меш лидара, и рядом с кадром пишется
<кадр>_depth.png — расстояние вдоль луча в миллиметрах (uint16), 0 там, где меша нет или
дальше 65 м.

3DGRUT с нашим патчем (loss.use_depth) тянет гауссианы к этой поверхности: они ложатся на стены
и пол, а не висят в воздухе и не вытягиваются вдоль луча — поэтому с новых точек (выше, сбоку)
сцена не рассыпается на «иголки» и совпадает с коллайдером из того же меша.

Запуск: .venv/bin/python scripts/gs_depth.py <сессия> [--poses rko|gt] [--mesh out/mesh/<с>_rko/mesh.ply]
"""

import argparse
import time
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from eval_views import fisheye_rays, read_colmap

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--poses", choices=("rko", "gt"), default="rko")
    parser.add_argument("--mesh", help="меш лидара; по умолчанию out/mesh/<сессия>_<позы>/mesh.ply")
    parser.add_argument("--data", help="набор COLMAP; по умолчанию data/sessions/<с>/colmap[_gt]")
    args = parser.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / "sessions" / args.session / ("colmap" if args.poses == "rko" else "colmap_gt")
    mesh_path = Path(args.mesh) if args.mesh else ROOT / "out" / "mesh" / f"{args.session}_{args.poses}" / "mesh.ply"
    if not (data / "sparse" / "0" / "images.txt").exists():
        raise SystemExit(f"нет набора {data} — сначала scripts/session_to_colmap.py {args.session}")
    if not mesh_path.exists():
        raise SystemExit(f"нет меша {mesh_path} — сначала scripts/build_mesh.py")

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    cams, imgs = read_colmap(data / "sparse" / "0")
    rays_by_cam = {}
    t0, covered = time.time(), []
    for k, (name, (R_cw, t_cw, cam_id)) in enumerate(sorted(imgs.items())):
        w, h, f, _, cx, cy = cams[cam_id][:6]
        if cam_id not in rays_by_cam:
            rays_by_cam[cam_id] = fisheye_rays(w, h, f, cx, cy)
        d = rays_by_cam[cam_id] @ R_cw  # = R_wc @ d_cam (R_wc = R_cw^T): единичные лучи в мире
        center = -R_cw.T @ t_cw
        rays = np.c_[np.broadcast_to(center, d.shape), d].astype(np.float32)
        t_hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
        depth = np.where(np.isfinite(t_hit) & (t_hit < 65.0), np.round(t_hit * 1000.0), 0).astype(np.uint16)
        Image.fromarray(depth.reshape(h, w)).save(data / "images" / (Path(name).with_suffix("").as_posix() + "_depth.png"))
        covered.append(float((depth > 0).mean()))
        if (k + 1) % 100 == 0:
            print(f"[depth] {k + 1}/{len(imgs)} кадров, {time.time() - t0:.0f} с", flush=True)
    print(f"[depth] {data}: кадров {len(imgs)}, глубина известна в среднем на {np.mean(covered):.0%} кадра, "
          f"меш {mesh_path}, {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
