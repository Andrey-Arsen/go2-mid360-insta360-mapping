#!/usr/bin/env python3
"""Isaac Sim: робот IsaacLegs с его политикой ходьбы — в любом окружении, в том числе в нашей
восстановленной модели (scenes/reconstructed/<имя>_rko.usda — меш, или <имя>_gs.usda — гауссианы;
коллайдер в обеих).

Роботы (third_party/IsaacLegs, двойники с графами ROS 2): go2 — Unitree Go2 (четвероногий),
g1 — Unitree G1 (гуманоид, 23 сустава), g1_29dof — G1 с 29 суставами. Ходит контроллер
`ros2 launch fullbody_controller <робот>.launch.py`, руль — /cmd_vel (scripts/teleop_wasd.py).
Обычно всё вместе запускает scripts/robot_test.sh.

Модель восстановлена в тех же координатах, что исходная сцена, поэтому место появления робота
по умолчанию — там же, где стоял робот, когда снимал (--spawn X Y КУРС — другое).
Настройки рендера из корневого слоя окружения (у сцены гауссиан — обратная тональная кривая
NuRec) применяются и здесь, иначе гауссианы по ссылке выходят белёсыми.

Запуск: ~/isaacsim/python.sh scripts/run_robot_isaac.py --robot g1 --env scenes/reconstructed/sklad03_rko.usda
        [--spawn X Y YAW] [--headless] [--no-play] [--seconds 0] [--isaaclegs-sensors]
"""

import argparse
import signal
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
ISAACLEGS = ROOT / "third_party" / "IsaacLegs" / "isaacsim_envs" / "gui"
ROBOTS = {  # сцена IsaacLegs, путь робота в ней
    "go2": ("go2.usd", "/World/go2"),
    "g1": ("g1.usd", "/World/g1"),
    "g1_29dof": ("g1_29dof.usd", "/World/g1_29dof_rev_1_0"),
}
parser = argparse.ArgumentParser()
parser.add_argument("--robot", choices=sorted(ROBOTS), default="go2")
parser.add_argument("--env", default="full_warehouse", help="окружение Isaac (как у go2_warehouse.sh) или путь к .usd/.usda")
parser.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"))
parser.add_argument("--lift", type=float, default=0.03,
                    help="поднять робота при появлении, м: пол модели — неровный меш, ступни не должны начинать в нём")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-play", action="store_true", help="не запускать симуляцию сразу (Play в окне)")
parser.add_argument("--seconds", type=float, default=0.0, help="выйти через столько секунд симуляции (для проверки)")
parser.add_argument("--isaaclegs-sensors", action="store_true", help="оставить головной лидар и камеру Go2 (медленнее)")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})

STOP = []
signal.signal(signal.SIGINT, lambda *_: STOP.append(1))
signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))

import carb  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from pxr import Sdf  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
enable_extension("isaacsim.ros2.bridge")
app.update()
import go2_scene  # noqa: E402


def env_render_settings(url):
    """renderSettings корневого слоя окружения -> настройки RTX (rtx:post:tonemap:op -> /rtx/post/tonemap/op)."""
    if url is None or "://" in url:
        return {}
    settings = Sdf.Layer.FindOrOpen(url).customLayerData.get("renderSettings", {})
    for key, value in settings.items():
        carb.settings.get_settings().set("/" + key.replace(":", "/"), value)
    return settings


def add_clock(stage):
    """/clock из времени симуляции — контроллер IsaacLegs работает по нему; у Go2 своего графа нет."""
    graphs = stage.GetPrimAtPath("/Graphs")
    if graphs.IsValid() and any("Clock" in c.GetName() for c in graphs.GetChildren()):
        return
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/Graphs/RobotTest_Clock", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
                ("context.outputs:context", "clock.inputs:context"),
            ],
        },
    )


def main():
    scene, robot_path = ROBOTS[args.robot]
    scene = ISAACLEGS / scene
    if not scene.exists():
        raise SystemExit(f"нет {scene} — нужен third_party/IsaacLegs (см. README, установка)")
    stage, url = go2_scene.open_scene(app, args.env, scene)
    if not stage.GetPrimAtPath(robot_path).IsValid():
        raise SystemExit(f"в {scene.name} нет робота {robot_path}")
    go2_scene.set_spawn(stage, args.spawn, robot_path)
    if args.lift:
        from pxr import Gf, Usd, UsdGeom
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.Xformable(stage.GetPrimAtPath(robot_path)).AddTranslateOp(
                UsdGeom.XformOp.PrecisionDouble, "lift").Set(Gf.Vec3d(0.0, 0.0, args.lift))
    applied = env_render_settings(url)
    if not args.isaaclegs_sensors:
        go2_scene.disable_graphs(["/Graphs/ROS_LidarRTX", "/Graphs/ROS_Camera"])
    add_clock(stage)
    app.update()
    timeline = omni.timeline.get_timeline_interface()
    if not args.no_play:
        timeline.play()
    print(f"[robot] {args.robot} в {url}; настроек рендера окружения {len(applied)}", flush=True)
    print("[robot] готово: симуляция идёт" if not args.no_play else "[robot] готово: нажмите Play", flush=True)
    t_end = args.seconds if args.seconds > 0 else None
    while app.is_running() and not STOP:
        app.update()
        if t_end is not None and timeline.get_current_time() >= t_end:
            break
    timeline.stop()


try:
    main()
except SystemExit as e:
    print(f"[robot] {e}", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
