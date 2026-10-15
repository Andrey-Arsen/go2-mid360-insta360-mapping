#!/usr/bin/env python3
"""Облако с цветом в E57 (ASTM E2807) — для CloudCompare, Revit/ReCap, Autodesk, Faro Scene.

Берёт cloud.laz из export_cloud.py (точки по позам одометрии, цвет с камер, пол — класс 2) и
пишет <mesh_dir>/cloud.e57: один скан в мировых координатах модели (те же, что у меша и USD),
x/y/z в метрах (double), цвет 0..255. Класс точки (пол) не пишется: в стандарте E57 такого поля нет.

Запуск: .venv/bin/python scripts/export_e57.py out/mesh/sklad01_rko
"""

import sys
from pathlib import Path

import laspy
import numpy as np
import pye57


def main():
    mdir = Path(sys.argv[1])
    src = mdir / "cloud.laz"
    if not src.exists():
        raise SystemExit(f"нет {src} — сначала scripts/export_cloud.py")
    las = laspy.read(src)
    rgb = np.c_[las.red, las.green, las.blue].astype(np.float64)
    if rgb.max() > 255:  # LAS хранит цвет 16-битным
        rgb = rgb / 257.0
    data = {
        "cartesianX": np.asarray(las.x, np.float64),
        "cartesianY": np.asarray(las.y, np.float64),
        "cartesianZ": np.asarray(las.z, np.float64),
        "colorRed": np.round(rgb[:, 0]).astype(np.uint8),
        "colorGreen": np.round(rgb[:, 1]).astype(np.uint8),
        "colorBlue": np.round(rgb[:, 2]).astype(np.uint8),
    }
    out = mdir / "cloud.e57"
    out.unlink(missing_ok=True)
    e57 = pye57.E57(str(out), mode="w")
    e57.write_scan_raw(data, name=mdir.name)
    e57.close()
    check = pye57.E57(str(out))
    n = check.get_header(0).point_count
    check.close()
    print(f"[e57] {out}: точек {n}, {out.stat().st_size / 1e6:.0f} МБ")


if __name__ == "__main__":
    main()
