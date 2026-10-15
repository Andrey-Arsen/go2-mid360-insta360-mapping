#!/usr/bin/env python3
"""Проверка: как Go2 стоит с надстройкой — на полу, приводы держат стойку (без политики ходьбы).

Сцена открывается, под робота кладётся пол (z = 0), физика идёт --seconds. Цели приводов
в go2.usd IsaacLegs — стойка политики (бедро ±0.1, бедренный 0.8/1.0, голень −1.5 рад), поэтому
робот сам садится в стойку. Печатает высоту корпуса, крен/тангаж, высоту лидара и объективов
над полом и рисует три вида (без физики во время рендера). С --compare то же для Go2 без
надстройки — видно, что меняет её масса.

Политика при ходьбе держит корпус сама (обучена с добавкой массы к корпусу −5…+5 кг), так
что это проверка посадки и баланса, а не походки.

Запуск: ~/isaacsim/python.sh scripts/check_go2_stand.py [robot/go2_mapping/go2_mapping.usda] [--compare] [--seconds 3]
        -> robot/go2_mapping/stand_check.png
"""

import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SCENE = Path(ARGS[0]).resolve() if ARGS else ROOT / "robot" / "go2_mapping" / "go2_mapping.usda"
PLAIN = ROOT / "third_party" / "IsaacLegs" / "isaacsim_envs" / "gui" / "go2.usd"
SECONDS = float(sys.argv[sys.argv.index("--seconds") + 1]) if "--seconds" in sys.argv else 3.0
FRAMES = {"лидар": "/World/go2/base/sensor_mount/frames/livox_frame",
          "объектив перед.": "/World/go2/base/sensor_mount/frames/insta360_front",
          "объектив задн.": "/World/go2/base/sensor_mount/frames/insta360_back"}


def world(stage, path):
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def stand(scene, render_to=None):
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene)):
        raise RuntimeError(f"не открылась {scene}")
    for _ in range(20):
        app.update()
    stage = ctx.get_stage()
    ground = UsdGeom.Plane.Define(stage, "/CheckGround")
    ground.CreateAxisAttr("Z")
    ground.CreateWidthAttr(20.0)
    ground.CreateLengthAttr(20.0)
    UsdPhysics.CollisionAPI.Apply(ground.GetPrim())
    ground.CreateDisplayColorAttr([Gf.Vec3f(0.55, 0.57, 0.6)])
    UsdLux.DomeLight.Define(stage, "/CheckLight").CreateIntensityAttr(1000.0)
    sim = SimulationContext(physics_dt=1 / 200, rendering_dt=1 / 50, stage_units_in_meters=1.0)
    sim.initialize_physics()
    sim.play()
    for _ in range(int(SECONDS * 200)):
        sim.step(render=False)
    sim.pause()
    B = world(stage, "/World/go2/base")
    roll, pitch, _ = Rotation.from_matrix(B[:3, :3]).as_euler("xyz", degrees=True)
    res = {"корпус над полом, мм": B[2, 3] * 1000, "крен, °": roll, "тангаж, ° (+ нос вниз)": pitch}
    for name, path in FRAMES.items():
        if stage.GetPrimAtPath(path).IsValid():
            res[f"{name} над полом, мм"] = world(stage, path)[2, 3] * 1000
    if render_to:
        carb.settings.get_settings().set("/app/player/playSimulations", False)  # рендер без шагов физики
        c = B[:3, 3] + np.array([0, 0, 0.12])
        annots = []
        for k, (az, el, dist) in enumerate(((35, 18, 1.5), (90, 2, 1.4), (-150, 30, 1.5))):
            a, e = np.radians(az), np.radians(el)
            eye = c + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
            f = (c - eye) / np.linalg.norm(c - eye)
            right = np.cross(f, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, f)
            cam = UsdGeom.Camera.Define(stage, f"/CheckCams/c{k}")
            cam.CreateFocalLengthAttr(24.0)
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
            UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*right, 0, *up, 0, *(-f), 0, *eye, 1))
            an = rep.AnnotatorRegistry.get_annotator("rgb")
            an.attach([rep.create.render_product(str(cam.GetPath()), (560, 560))])
            annots.append(an)
        for _ in range(4):
            rep.orchestrator.step(rt_subframes=8)
        Image.fromarray(np.hstack([np.array(a.get_data())[..., :3] for a in annots])).save(render_to)
        carb.settings.get_settings().set("/app/player/playSimulations", True)
    sim.stop()
    return res


def main():
    out = SCENE.parent / "stand_check.png"
    rows = {"с надстройкой": stand(SCENE, out)}
    if "--compare" in sys.argv:
        rows["Go2 без надстройки"] = stand(PLAIN)
    for label, res in rows.items():
        print(f"[stand] {label}: " + "; ".join(f"{k} {v:.1f}" for k, v in res.items()), flush=True)
    print(f"[stand] {out}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
