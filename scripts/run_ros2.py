#!/usr/bin/env python3
"""Открывает стенд в Isaac Sim и публикует датчики в ROS 2 (см. ros2_sensors.py).

Запускать из терминала, где настроен ROS 2: Isaac Sim возьмёт оттуда ROS_DOMAIN_ID,
и RViz из соседнего терминала (scripts/rviz.sh) увидит топики.

Запуск: ~/isaacsim/python.sh scripts/run_ros2.py [--headless] [--rays] [--seconds 0]
"""

import argparse
import os
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="без окна Isaac Sim")
parser.add_argument("--rays", action="store_true", help="рисовать в окне лучи и точки лидара")
parser.add_argument("--seconds", type=float, default=0.0, help="сколько секунд работать; 0 — пока не закроют окно")
parser.add_argument("--image-size", type=int, default=960, help="размер кадров камер в ROS, px")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": args.headless})

import omni.timeline  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid360_driver  # noqa: E402
import ros2_sensors  # noqa: E402

SCENE = Path(__file__).resolve().parents[1] / "scenes" / "simple_room_mid360_fisheye.usda"


def main():
    if not omni.usd.get_context().open_stage(str(SCENE)):
        raise RuntimeError(f"не открылась сцена {SCENE}")
    for _ in range(10):
        app.update()
    stage = omni.usd.get_context().get_stage()
    sim = SimulationContext(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    mid360_driver.start(stage)
    ros2_sensors.start(stage, image_size=args.image_size)
    if args.rays:
        mid360_driver.show_points(stage)
        mid360_driver.show_rays(stage)
    print(f"[run_ros2] ROS_DOMAIN_ID={os.environ.get('ROS_DOMAIN_ID', '0 (не задан)')}", flush=True)

    sim.initialize_physics()
    sim.play()
    timeline = omni.timeline.get_timeline_interface()
    while app.is_running():
        app.update()
        if args.seconds and timeline.get_current_time() >= args.seconds:
            break
    sim.stop()
    ros2_sensors.stop()
    mid360_driver.stop()


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
