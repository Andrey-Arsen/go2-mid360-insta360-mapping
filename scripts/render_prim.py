#!/usr/bin/env python3
"""Картинка проверки: объект сцены с трёх сторон (спереди-сбоку, сбоку, сзади-сверху).

Запуск: ~/isaacsim/python.sh scripts/render_prim.py robot/go2_mapping/go2_mapping.usda /World/go2 out.png [--size 0.8]
  --size — размер кадра вокруг объекта, м (по умолчанию по габариту объекта)
"""

import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def main():
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    scene, prim_path, out = Path(sys.argv[1]).resolve(), sys.argv[2], Path(sys.argv[3])
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene)):
        raise SystemExit(f"не открылась {scene}")
    for _ in range(20):
        app.update()
    stage = ctx.get_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise SystemExit(f"нет {prim_path}")
    UsdLux.DomeLight.Define(stage, "/CheckLight").CreateIntensityAttr(1000.0)  # ровный свет со всех сторон
    # Габарит — только по сеткам (у лидаров и камер бывают огромные условные границы).
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    box = Gf.Range3d()
    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if p.GetTypeName() == "Mesh":
            box = Gf.Range3d.GetUnion(box, cache.ComputeWorldBound(p).ComputeAlignedRange())
    lo, hi = np.array(box.GetMin()), np.array(box.GetMax())
    c = (lo + hi) / 2
    size = float(sys.argv[sys.argv.index("--size") + 1]) if "--size" in sys.argv else float(np.max(hi - lo))
    print(f"[render] габарит {np.round(lo, 2)} .. {np.round(hi, 2)}, кадр {size:.2f} м", flush=True)
    annots = []
    for k, (az, el) in enumerate(((35, 20), (90, 5), (-150, 35))):
        a, e = np.radians(az), np.radians(el)
        eye = c + 1.9 * size * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
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
    Image.fromarray(np.hstack([np.array(a.get_data())[..., :3] for a in annots])).save(out)
    print(f"[render] {out}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
