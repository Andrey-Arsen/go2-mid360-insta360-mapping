#!/usr/bin/env python3
"""Выравнивает траекторию RKO-LIO по силе тяжести за всю запись -> <траектория>_level.txt.

Низ RKO-LIO определяет по IMU в покое в начале записи. Если покоя не было (робот сразу пошёл),
ошибка в градус наклоняет всю карту: у sklad02 крен 1.4°, и через 30 м пути высота ушла на
54 см, хотя форма траектории верна (после совмещения 2.8 см). Средняя за всю запись сила тяжести
в кадре одометрии — среднее R(t)·a(t) по IMU (ускорения робота за минуту в среднем почти
нулевые) — должна смотреть ровно вверх; отклонение снимаем поворотом вокруг начала одометрии
(только крен и тангаж, курс не трогаем). Проверено: sklad02 — найдено 1.29°/0.39°, настоящее
1.36°/0.32°; записи с покоем в начале — 0.1–0.3°.

Запуск: .venv/bin/python scripts/level_trajectory.py <сессия> <tum RKO-LIO>
"""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
BASE_S = 1_700_000_000


def main():
    session, tum = sys.argv[1], Path(sys.argv[2])
    est = np.loadtxt(tum)
    t, rot = est[:, 0] - BASE_S, Rotation.from_quat(est[:, 4:8])
    imu = np.load(ROOT / "data" / "sessions" / session / "imu.npz")
    m = (imu["t"] >= t[0]) & (imu["t"] <= t[-1])
    g = Slerp(t, rot)(imu["t"][m]).apply(imu["accel"][m]).mean(0)
    fix = Rotation.align_vectors([[0.0, 0.0, 1.0]], [g / np.linalg.norm(g)])[0]
    roll, pitch, _ = fix.as_euler("xyz", degrees=True)
    fix = Rotation.from_euler("xy", [roll, pitch], degrees=True)  # без курса
    out = tum.with_name(tum.stem + "_level.txt")
    level = est.copy()
    level[:, 1:4] = fix.apply(est[:, 1:4])
    level[:, 4:8] = (fix * rot).as_quat()
    np.savetxt(out, level, fmt="%.9f")
    tilt = np.degrees(np.arctan2(np.hypot(g[0], g[1]), g[2]))
    print(f"[level] наклон карты {tilt:.2f}° (крен {roll:+.2f}°, тангаж {pitch:+.2f}°) снят -> {out}")


if __name__ == "__main__":
    main()
