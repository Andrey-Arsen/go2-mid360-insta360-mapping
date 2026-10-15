#!/usr/bin/env python3
"""URDF -> USD для Isaac Sim без окна (импортёр isaacsim.asset.importer.urdf) и картинка проверки.

Пишет <out>.usd (с папкой configuration/ рядом) и <out>.png — вид спереди-сбоку со светом.
Печатает сетки (есть ли все) и кадры датчиков (где лежат, куда смотрят оси).

Запуск: ~/isaacsim/python.sh scripts/urdf_to_usd.py robot/go2_sensor_mount/urdf/go2_sensor_mount.urdf \
            robot/go2_sensor_mount/usd/go2_sensor_mount.usd [--fix-base]
"""

import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})


def main():
    import numpy as np
    import omni.kit.commands
    import omni.replicator.core as rep
    import omni.usd
    from isaacsim.core.utils.extensions import enable_extension
    from PIL import Image
    from pxr import Gf, Usd, UsdGeom, UsdLux

    enable_extension("isaacsim.asset.importer.urdf")
    for _ in range(5):
        app.update()
    urdf, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    _, cfg = omni.kit.commands.execute("URDFCreateImportConfig")
    cfg.merge_fixed_joints = False  # кадры датчиков должны остаться отдельными
    cfg.fix_base = "--fix-base" in sys.argv
    cfg.make_default_prim = True
    cfg.distance_scale = 1.0
    ok, root = omni.kit.commands.execute("URDFParseAndImportFile", urdf_path=str(urdf), import_config=cfg,
                                         dest_path=str(out))
    if not ok:
        raise SystemExit(f"импорт не удался — смотрите предупреждения importer.urdf выше (имена с дефисами/пробелами?)")
    st = Usd.Stage.Open(str(out))
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    meshes = [p for p in Usd.PrimRange(st.GetPseudoRoot(), Usd.TraverseInstanceProxies()) if p.GetTypeName() == "Mesh"]
    print(f"[usd] {out}: корень {root}, сеток {len(meshes)}", flush=True)
    for p in meshes:
        r = cache.ComputeWorldBound(p).ComputeAlignedRange()
        size = np.array(r.GetMax()) - np.array(r.GetMin())
        print(f"[usd]   {p.GetPath()}  размер {np.round(size * 1000).astype(int)} мм", flush=True)
    for p in st.Traverse():
        if (p.IsA(UsdGeom.Xformable) and "optical" not in p.GetName()
                and any(k in p.GetName() for k in ("livox", "insta360_front", "insta360_back"))):
            M = np.array(UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
            print(f"[usd]   кадр {p.GetName():16s} {np.round(M[:3, 3] * 1000, 1)} мм, X {np.round(M[:3, 0], 2)}, Z {np.round(M[:3, 2], 2)}",
                  flush=True)

    # Картинка: робот на пустой сцене, свет, камера спереди-сбоку-сверху.
    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.DefinePrim("/World/Robot", "Xform").GetReferences().AddReference(str(out))
    UsdLux.DomeLight.Define(stage, "/World/Dome").CreateIntensityAttr(800.0)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(50, 0, 30))
    for _ in range(10):
        app.update()
    bound = cache.ComputeWorldBound(stage.GetPrimAtPath("/World/Robot")).ComputeAlignedRange()
    c = (np.array(bound.GetMin()) + np.array(bound.GetMax())) / 2
    size = float(np.max(np.array(bound.GetMax()) - np.array(bound.GetMin())))
    views = []
    for k, (az, el) in enumerate(((40, 25), (-140, 20))):
        a, e = np.radians(az), np.radians(el)
        eye = c + 2.2 * size * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        f = (c - eye) / np.linalg.norm(c - eye)
        right = np.cross(f, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, f)
        cam = UsdGeom.Camera.Define(stage, f"/World/Cam{k}")
        cam.CreateFocalLengthAttr(24.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
        m = Gf.Matrix4d(*right, 0, *up, 0, *(-f), 0, *eye, 1)
        UsdGeom.Xformable(cam).AddTransformOp().Set(m)
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rep.create.render_product(str(cam.GetPath()), (640, 640))])
        views.append(annot)
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=8)
    img = np.hstack([np.array(a.get_data())[..., :3] for a in views])
    Image.fromarray(img).save(out.with_suffix(".png"))
    print(f"[usd] картинка {out.with_suffix('.png')}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
