#!/usr/bin/env python3
"""Качество картинки на отложенных кадрах: гауссианы (3DGRUT) против цветного меша.

Отложенные кадры — те же, что 3DGRUT не видел при обучении: имена кадров набора
(session_to_colmap.py) по алфавиту, каждый 8-й (test_split_interval). Рендеры гауссиан —
<прогон>/ours_<шаг>/renders/NNNNN.png в том же порядке. Меш рисуется здесь же: луч на
пиксель по той же fisheye-модели (r = f·θ) и тем же позам, цвет — интерполяция цветов
вершин. Метрики только по маске (θ < 90° и без сканера) — 3DGRUT считает PSNR по всему
кадру, вместе с чёрным фоном за кругом и самим сканером.

Пишет <прогон>/eval_views.json и картинку сравнения eval_views.png (эталон | гауссианы | меш).
Запуск: .venv/bin/python scripts/eval_views.py tilt30gs [--poses rko|gt]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_metrics import psnr, ssim  # noqa: E402


def read_colmap(sparse):
    cams = {}
    for line in open(sparse / "cameras.txt"):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cams[int(e[0])] = (int(e[2]), int(e[3]), *map(float, e[4:8]))
    imgs = {}
    for line in open(sparse / "images.txt"):
        e = line.split()
        if line.startswith("#") or len(e) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
        imgs[e[9]] = (Rotation.from_quat([qx, qy, qz, qw]).as_matrix(), np.array([tx, ty, tz]), int(e[8]))
    return cams, imgs


def fisheye_rays(w, h, f, cx, cy):
    u, v = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    x, y = (u - cx) / f, (v - cy) / f
    theta = np.hypot(x, y)
    s = np.where(theta > 1e-9, np.sin(theta) / np.maximum(theta, 1e-9), 1.0)
    return np.stack([x * s, y * s, np.cos(theta)], -1).reshape(-1, 3)


def render_mesh(scene, colors, tris, R_cw, t_cw, rays_cam, shape):
    R_wc = R_cw.T
    center = -R_wc @ t_cw
    d = rays_cam @ R_wc.T
    rays = np.c_[np.broadcast_to(center, d.shape), d].astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))
    ids = hit["primitive_ids"].numpy()
    uv = hit["primitive_uvs"].numpy()
    ok = ids != o3d.t.geometry.RaycastingScene.INVALID_ID
    out = np.zeros((len(d), 3))
    t = tris[ids[ok]]
    w1, w2 = uv[ok, 0:1], uv[ok, 1:2]
    out[ok] = (1 - w1 - w2) * colors[t[:, 0]] + w1 * colors[t[:, 1]] + w2 * colors[t[:, 2]]
    return out.reshape(*shape, 3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--poses", choices=("rko", "gt"), default="rko")
    parser.add_argument("--every", type=int, default=8, help="каждый N-й кадр — отложенный (как в 3DGRUT)")
    parser.add_argument("--run", help="папка обучения (с runs/ внутри); по умолчанию out/gs/<сессия>_<позы>")
    args = parser.parse_args()

    data = ROOT / "data" / "sessions" / args.session / ("colmap" if args.poses == "rko" else "colmap_gt")
    run = Path(args.run) if args.run else ROOT / "out" / "gs" / f"{args.session}_{args.poses}"
    # Последний шаг самого свежего обучения (старые прогоны в той же папке не в счёт).
    renders = sorted(run.glob("runs/**/ours_*/renders"), key=lambda p: (p.parent.parent.stat().st_mtime, int(p.parent.name.split("_")[1])))
    if not renders:
        raise SystemExit(f"нет рендеров отложенных кадров в {run}/runs — обучение не закончилось?")
    renders = renders[-1]
    cams, imgs = read_colmap(data / "sparse" / "0")
    names = sorted(imgs)[::args.every]

    mesh = o3d.io.read_triangle_mesh(str(ROOT / "out" / "mesh" / f"{args.session}_rko" / "mesh_colored.ply"))
    colors, tris = np.asarray(mesh.vertex_colors), np.asarray(mesh.triangles)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    rows, strip = [], []
    for k, name in enumerate(names):
        R_cw, t_cw, cid = imgs[name]
        w, h, f, _, cx, cy = cams[cid]
        gt = np.asarray(Image.open(data / "images" / name).convert("RGB"), dtype=np.float64) / 255
        mask = np.asarray(Image.open(data / "images" / name.replace(".png", "_mask.png"))) > 127  # маска этого кадра
        gs = np.asarray(Image.open(renders / f"{k:05d}.png").convert("RGB"), dtype=np.float64) / 255
        me = render_mesh(scene, colors, tris, R_cw, t_cw, fisheye_rays(w, h, f, cx, cy), (h, w))
        rows.append({"frame": name, "psnr_gs": psnr(gs, gt, mask), "psnr_mesh": psnr(me, gt, mask),
                     "ssim_gs": ssim(gs, gt, mask), "ssim_mesh": ssim(me, gt, mask)})
        if k % max(1, len(names) // 4) == 0 and len(strip) < 4:
            crop = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))  # центр кадра, где маска
            strip.append(np.hstack([gt[crop], gs[crop], me[crop]]))
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("psnr_gs", "psnr_mesh", "ssim_gs", "ssim_mesh")}
    report = {"session": args.session, "poses": args.poses, "renders": str(renders), "test_frames": len(rows),
              "mean": mean, "frames": rows}
    (run / "eval_views.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    img = (np.vstack(strip) * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)).save(run / "eval_views.png")
    print(f"[eval] {args.session} ({args.poses}), отложенных кадров {len(rows)}: PSNR гауссианы {mean['psnr_gs']:.2f} дБ, "
          f"меш {mean['psnr_mesh']:.2f} дБ; SSIM {mean['ssim_gs']:.3f} / {mean['ssim_mesh']:.3f}; {run / 'eval_views.png'}")


if __name__ == "__main__":
    main()
