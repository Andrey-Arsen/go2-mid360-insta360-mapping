#!/usr/bin/env python3
"""Сессия -> набор в формате COLMAP для Gaussian splatting (3DGRUT / 3DGUT, gsplat).

COLMAP здесь не запускается: позы камер — из траектории лидара (RKO-LIO или эталон) и
жёсткой связи лидар -> объектив из meta.json, масштаб — метры, z вверх по гравитации.
Поэтому сплат сразу совпадает с мешем-коллайдером, без подгонки.

Камеры — два fisheye-объектива, модель OPENCV_FISHEYE с нулевыми коэффициентами: наш
ftheta равноугольный, r = f * theta, f = (size/2) / (fov/2 в рад). Оси: камера USD
смотрит по -Z, верх +Y; COLMAP/OpenCV — по +Z, y вниз: поворот diag(1, -1, -1).

Маска на объектив (<кадр>_mask.png рядом с кадром; 255 — брать): сканер из rig_mask.py
(стенд) или маска каждого кадра из render_drive.py (робот: ноги двигаются)
и всё дальше --max-angle от оси (край круга и чёрный фон вне него).
Начальные точки — облако лидара с цветом (cloud.laz из export_cloud.py), --points штук.

data/sessions/<s>/colmap/
  images/{front,back}/NNN.png       ссылки на кадры сессии (или уменьшенные копии при --size)
  images/{front,back}/NNN_mask.png  ссылки на маску объектива (или своя маска кадра)
  sparse/0/cameras.txt images.txt points3D.txt

Запуск: .venv/bin/python scripts/session_to_colmap.py tilt30gs [--poses rko|gt] [--size 960]
"""

import argparse
import json
import math
import os
import shutil
import sys
from pathlib import Path

import laspy
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from colorize_mesh import latest_tum, lidar_poses  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
USD_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


def lens_mask(size, fov_deg, max_angle_deg, rig=None):
    yy, xx = np.mgrid[:size, :size] + 0.5
    theta = np.hypot(xx - size / 2, yy - size / 2) / (size / 2) * math.radians(fov_deg / 2)
    keep = theta < math.radians(max_angle_deg)
    if rig is not None:
        keep &= rig
    return keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--poses", choices=("rko", "gt"), default="rko")
    parser.add_argument("--tum", help="траектория RKO-LIO; по умолчанию последняя")
    parser.add_argument("--max-angle", type=float, default=90.0, help="брать пиксели не дальше от оси, град")
    parser.add_argument("--size", type=int, help="уменьшить кадры до этого размера (по умолчанию — как есть, ссылками)")
    parser.add_argument("--points", type=int, default=300_000, help="начальных точек из облака лидара")
    parser.add_argument("--cloud", help="облако с цветом; по умолчанию out/mesh/<s>_rko/cloud.laz")
    parser.add_argument("--out", help="по умолчанию data/sessions/<s>/colmap[_gt]")
    args = parser.parse_args()

    sdir = ROOT / "data" / "sessions" / args.session
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    out = Path(args.out) if args.out else sdir / ("colmap" if args.poses == "rko" else "colmap_gt")
    if out.exists():
        shutil.rmtree(out)
    (out / "sparse" / "0").mkdir(parents=True)

    native, fov = meta["lens"]["size_px"], meta["lens"]["fov_deg"]
    size = args.size or native
    f = (size / 2) / math.radians(fov / 2)
    lenses = list(meta["T_lidar_lens"])
    shots = np.array(meta["shutter_times_s"])
    tum = None if args.poses == "gt" else Path(args.tum) if args.tum else latest_tum(args.session)
    T_wl = lidar_poses(sdir, tum, shots)

    lens_masks = {}
    with open(out / "sparse" / "0" / "cameras.txt", "w") as fc:
        fc.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy k1 k2 k3 k4\n")
        for cid, lens in enumerate(lenses, 1):
            fc.write(f"{cid} OPENCV_FISHEYE {size} {size} {f:.6f} {f:.6f} {size / 2:.6f} {size / 2:.6f} 0 0 0 0\n")
            rig = None
            if (sdir / "masks" / f"{lens}.png").exists():
                m = Image.open(sdir / "masks" / f"{lens}.png").convert("L").resize((size, size), Image.NEAREST)
                rig = np.asarray(m) > 127
            mask = lens_mask(size, fov, args.max_angle, rig)
            lens_masks[lens] = mask
            (out / "images" / lens).mkdir(parents=True)
            Image.fromarray(np.where(mask, 255, 0).astype(np.uint8)).save(out / "images" / f"{lens}_mask.png")
            print(f"[colmap] {lens}: f = {f:.2f} px, берётся {mask.mean():.1%} кадра"
                  f"{' (с маской сканера)' if rig is not None else ' (маски сканера нет — scripts/rig_mask.py)'}")

    n_img = 0
    with open(out / "sparse" / "0" / "images.txt", "w") as fi:
        fi.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME, затем пустая строка (2D-точек нет)\n")
        for k in range(len(shots)):
            for cid, lens in enumerate(lenses, 1):
                src = sdir / "images" / f"{k:03d}_{lens}.png"
                if not src.exists():
                    continue
                T_wc = T_wl[k] @ np.array(meta["T_lidar_lens"][lens]) @ USD_TO_CV
                R_cw = T_wc[:3, :3].T
                t_cw = -R_cw @ T_wc[:3, 3]
                qx, qy, qz, qw = Rotation.from_matrix(R_cw).as_quat()
                name = f"{lens}/{k:03d}.png"
                dst = out / "images" / name
                if size == native:
                    os.symlink(os.path.relpath(src, dst.parent), dst)
                else:
                    Image.open(src).convert("RGB").resize((size, size), Image.LANCZOS).save(dst)
                frame_mask = sdir / "images" / f"{k:03d}_{lens}_mask.png"
                if frame_mask.exists():  # робот в кадре (render_drive.py): маска объектива И маска кадра
                    m = np.asarray(Image.open(frame_mask).convert("L").resize((size, size), Image.NEAREST)) > 127
                    Image.fromarray(np.where(lens_masks[lens] & m, 255, 0).astype(np.uint8)).save(
                        dst.with_name(f"{k:03d}_mask.png"))
                else:
                    os.symlink(f"../{lens}_mask.png", dst.with_name(f"{k:03d}_mask.png"))
                n_img += 1
                fi.write(f"{n_img} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} {t_cw[0]:.6f} {t_cw[1]:.6f} {t_cw[2]:.6f} {cid} {name}\n\n")

    cloud = Path(args.cloud) if args.cloud else ROOT / "out" / "mesh" / f"{args.session}_rko" / "cloud.laz"
    las = laspy.read(cloud)
    xyz = np.c_[las.x, las.y, las.z]
    rgb = (np.c_[las.red, las.green, las.blue] >> 8).astype(np.uint8)
    pick = np.random.default_rng(0).choice(len(xyz), min(args.points, len(xyz)), replace=False)
    with open(out / "sparse" / "0" / "points3D.txt", "w") as fp:
        fp.write("# POINT3D_ID X Y Z R G B ERROR (треков нет: точки из лидара)\n")
        for i, (p, c) in enumerate(zip(xyz[pick], rgb[pick]), 1):
            fp.write(f"{i} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]} 0\n")
    (out / "source.json").write_text(json.dumps({
        "session": args.session, "poses": str(tum) if tum else "gt.npz", "cloud": str(cloud),
        "size_px": size, "focal_px": f, "max_angle_deg": args.max_angle, "images": n_img, "points": len(pick),
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[colmap] {out}: кадров {n_img} ({len(shots)} снимков x {len(lenses)}), начальных точек {len(pick)}, "
          f"позы {'эталон' if tum is None else tum.name}")


if __name__ == "__main__":
    main()
