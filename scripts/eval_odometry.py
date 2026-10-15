#!/usr/bin/env python3
"""Сравнивает траекторию RKO-LIO (TUM из dump_results) с эталоном сессии.

Одометрия ведётся в odom, а odom = поза стойки в начале (так же, как /tf_static в
session_to_bag.py), поэтому сравниваем без подгонки — это честный дрейф. Для
справки — ATE после жёсткого совмещения (Umeyama без масштаба).

Запуск: ~/isaacsim/python.sh scripts/eval_odometry.py walk01 out/rko/walk01_0/walk01_tum_0.txt
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
BASE_S = 1_700_000_000


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((dst - mu_d).T @ (src - mu_s))
    s = np.diag([1, 1, np.sign(np.linalg.det(u @ vt))])
    r = u @ s @ vt
    return r, mu_d - r @ mu_s


def draw(path_png, gt, est, title):
    lo, hi = np.minimum(gt[:, :2].min(0), est[:, :2].min(0)) - 0.5, np.maximum(gt[:, :2].max(0), est[:, :2].max(0)) + 0.5
    scale = 700 / max(hi - lo)
    w, h = int((hi[0] - lo[0]) * scale) + 1, int((hi[1] - lo[1]) * scale) + 40
    img = Image.new("RGB", (w, h), (24, 24, 28))
    d = ImageDraw.Draw(img)
    to_px = lambda p: [((x - lo[0]) * scale, 40 + (hi[1] - y) * scale) for x, y in p[:, :2]]
    d.line(to_px(gt), fill=(90, 200, 90), width=3)
    d.line(to_px(est), fill=(240, 120, 40), width=2)
    d.text((8, 8), title, fill=(230, 230, 230))
    d.text((8, 22), "green = ground truth, orange = RKO-LIO", fill=(180, 180, 180))
    img.save(path_png)


def main():
    session, tum = sys.argv[1], Path(sys.argv[2])
    sdir = ROOT / "data" / "sessions" / session
    gt = np.load(sdir / "gt.npz")
    est = np.loadtxt(tum)
    t = est[:, 0] - BASE_S
    t = t[(t >= gt["t"][0]) & (t <= gt["t"][-1])]
    est = est[: len(t)] if len(t) == len(est) else est[np.isin(est[:, 0] - BASE_S, t)]

    p0, q0 = gt["pos"][0], Rotation.from_quat(gt["quat_xyzw"][0])
    world_odom = Rotation.from_euler("z", q0.as_euler("ZYX")[0])
    est_pos = world_odom.apply(est[:, 1:4]) + p0
    est_rot = world_odom * Rotation.from_quat(est[:, 4:8])
    gt_pos = np.c_[[np.interp(t, gt["t"], gt["pos"][:, i]) for i in range(3)]].T
    gt_rot = Rotation.from_quat(gt["quat_xyzw"][np.clip(np.searchsorted(gt["t"], t), 0, len(gt["t"]) - 1)])

    err = np.linalg.norm(est_pos - gt_pos, axis=1)
    r, tr = umeyama(est_pos, gt_pos)
    err_aligned = np.linalg.norm((est_pos @ r.T + tr) - gt_pos, axis=1)
    rot_err = np.degrees((gt_rot.inv() * est_rot).magnitude())
    walked = float(np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1)))
    report = {
        "session": session,
        "poses": int(len(t)),
        "walked_m": round(walked, 2),
        "ate_rmse_m": round(float(np.sqrt(np.mean(err ** 2))), 4),
        "ate_max_m": round(float(err.max()), 4),
        "final_error_m": round(float(err[-1]), 4),
        "final_error_pct_of_path": round(float(100 * err[-1] / max(walked, 1e-6)), 2),
        "ate_rmse_aligned_m": round(float(np.sqrt(np.mean(err_aligned ** 2))), 4),
        "rot_err_mean_deg": round(float(rot_err.mean()), 3),
        "rot_err_max_deg": round(float(rot_err.max()), 3),
    }
    out = tum.parent
    draw(out / "trajectory_top.png", gt_pos, est_pos,
         f"{session}: ATE {report['ate_rmse_m'] * 100:.1f} cm, end error {report['final_error_m'] * 100:.1f} cm over {walked:.1f} m")
    (out / "eval.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
