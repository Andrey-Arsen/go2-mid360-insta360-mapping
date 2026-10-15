#!/usr/bin/env python3
"""Конфиг RKO-LIO под сессию: config/rko_lio_sim.yaml + размер вокселя карты по дальностям лидара.

Воксель карты RKO-LIO должен расти с размером помещения: 0.2–0.3 м хороши в комнате, а на
складе (дальности до 30 м) с ними одометрия хуже вдвое — там нужен 1 м (как в конфигах самого
RKO-LIO для улицы и для четвероногого робота). Правило — медиана дальности точек / 10, в
пределах 0.2–1.0 м (по образцу KISS-ICP: воксель ~ дальность / 100 от максимальной). Проверено:
комната tilt30 — медиана 3.0 м -> 0.3 м, ATE 1.9 см; склад test_wh01 — 11.7 м -> 1.0 м, ATE 5.7 см
против 7.6 см с 0.2 м.

RKO_VOXEL=0.5 — задать воксель вручную.

Запуск: .venv/bin/python scripts/rko_config.py <сессия> <куда записать .yaml>
"""

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    session, out = sys.argv[1], Path(sys.argv[2])
    points = np.load(ROOT / "data" / "sessions" / session / "lidar.npz")["points"][::20]
    rng = np.linalg.norm(points, axis=1)
    median = float(np.median(rng[rng > 0.3]))
    if os.environ.get("RKO_VOXEL"):
        voxel, why = float(os.environ["RKO_VOXEL"]), "задан RKO_VOXEL"
    else:
        voxel, why = float(np.clip(round(median / 10, 2), 0.2, 1.0)), f"медиана дальности {median:.1f} м / 10"
    lines = [ln for ln in (ROOT / "config" / "rko_lio_sim.yaml").read_text(encoding="utf-8").splitlines()
             if not ln.startswith("voxel_size:")]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines + [f"voxel_size: {voxel}"]) + "\n", encoding="utf-8")
    print(f"[rko] воксель карты {voxel} м ({why}) -> {out}")


if __name__ == "__main__":
    main()
