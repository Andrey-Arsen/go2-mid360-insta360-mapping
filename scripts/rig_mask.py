#!/usr/bin/env python3
"""Маска сканера для кадров fisheye: что в кадре закрывает сам Mid-360 (и его тень).

Лидар жёстко стоит под камерой, поэтому маска на объектив одна на все кадры. Кадр
рендерится дважды с одной позы — с лидаром и без — и разница > порога даёт маску; так
в неё попадает и тень лидара на полу. Объединение по нескольким позам маршрута (тень
от разных ламп), затем расширение на --grow пикселей.

Пишет data/sessions/<session>/masks/{front,back}.png: 255 — брать пиксель, 0 — сканер.
Запуск: ~/isaacsim/python.sh scripts/rig_mask.py tilt30gs
"""

import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handheld_trajectory import HandheldTrajectory  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RIG = "/World/Rig"
BODY = "/World/Rig/MID360"
LIDAR = "/World/Rig/MID360/torso_link/mid360_link/mid360_native_approx"
LENSES = {"front": "/World/Rig/Insta360/lens_front", "back": "/World/Rig/Insta360/lens_back"}


def world_matrix(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def main():
    session = sys.argv[1]
    sdir = ROOT / "data" / "sessions" / session
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    size = meta["lens"]["size_px"]
    ctx = omni.usd.get_context()
    ctx.open_stage(meta["scene"])
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(RIG))
        xf.ClearXformOpOrder()
        op = xf.AddTransformOp()
        op.Set(Gf.Matrix4d(1.0))
    app.update()
    T_rig_lidar = world_matrix(stage.GetPrimAtPath(LIDAR))
    T_carrier_rig = np.eye(4)
    T_carrier_rig[:3, 3] = -T_rig_lidar[:3, 3]
    traj = HandheldTrajectory(height=meta["trajectory"]["height_m"], speed=meta["trajectory"]["speed_mps"])

    annots = {}
    for name, path in LENSES.items():
        a = rep.AnnotatorRegistry.get_annotator("rgb")
        a.attach([rep.create.render_product(path, (size, size))])
        annots[name] = a
    body = UsdGeom.Imageable(stage.GetPrimAtPath(BODY))

    def shoot():
        rep.orchestrator.step(rt_subframes=16)
        rep.orchestrator.step(rt_subframes=16)  # первый шаг после изменения может быть старым
        return {k: np.asarray(a.get_data())[..., :3].astype(np.float32) for k, a in annots.items()}

    masks = {k: np.zeros((size, size), bool) for k in LENSES}
    for t in (5.0, 20.0, 35.0):
        pos, rot = traj.pose(t)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()[0]
        T[:3, 3] = pos[0]
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            op.Set(Gf.Matrix4d((T @ T_carrier_rig).T.tolist()))
            body.MakeVisible()
        with_body = shoot()
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            body.MakeInvisible()
        without = shoot()
        for k in LENSES:
            diff = np.abs(with_body[k] - without[k]).max(-1)
            masks[k] |= diff > 12
            print(f"[mask] t={t:.0f} с {k}: отличается {(diff > 12).mean():.2%} кадра", flush=True)
    out = sdir / "masks"
    out.mkdir(exist_ok=True)
    grow = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    for k, m in masks.items():
        m = ndimage.binary_opening(m, iterations=1)  # шум рендера
        m = ndimage.binary_fill_holes(ndimage.binary_closing(m, iterations=4))
        m = ndimage.binary_dilation(m, iterations=grow)
        Image.fromarray(np.where(m, 0, 255).astype(np.uint8)).save(out / f"{k}.png")
        print(f"[mask] {out / (k + '.png')}: закрыто {m.mean():.2%} кадра", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
