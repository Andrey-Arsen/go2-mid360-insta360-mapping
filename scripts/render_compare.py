#!/usr/bin/env python3
"""Рендерит исходную комнату и восстановленную модель с одних и тех же точек — для глаза.

Запуск: ~/isaacsim/python.sh scripts/render_compare.py scenes/reconstructed/walk01_rko.usda [--views walk] [--session ИМЯ]
        --session с окружением (сессия робота): оригинал — окружение, ракурсы — по пути робота
        -> out/compare_<имя>[_walk].png (сверху исходная сцена, снизу модель); walk — виды с высоты
        камеры сканера у маршрута (для гауссиан), по умолчанию — сверху из углов
"""

import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

enable_extension("omni.usd.schema.omni_nurec_types")  # гауссианы NuRec (gs_scene.py), если они есть в сцене

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from pxr import Gf, UsdGeom  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "scenes" / "simple_room_mid360_fisheye.usda"
VIEW_SETS = {  # глаз, точка взгляда (м, пол на z = 0)
    # сверху из углов — видно всю комнату (меш, коллайдер)
    "high": {"corner": ((3.9, 4.3, 2.9), (-0.5, -0.5, 0.6)), "back": ((-3.9, -2.9, 2.6), (1.0, 1.5, 0.8))},
    # с высоты камеры сканера (1.45 м) у маршрута — гауссианы хороши только около снятых видов
    "walk": {"table": ((2.8, 2.6, 1.45), (0.0, 0.0, 0.5)), "window": ((-2.6, -1.6, 1.45), (0.5, 4.8, 1.8))},
}
VIEW_SET = sys.argv[sys.argv.index("--views") + 1] if "--views" in sys.argv else "high"
VIEWS = VIEW_SETS.get(VIEW_SET)
SESSION = sys.argv[sys.argv.index("--session") + 1] if "--session" in sys.argv else None


def route_views(session):
    """Ракурсы по пути сессии робота: с высоты человека (1.6 м) на 30% и 70% пути, по ходу
    движения и чуть вниз; плюс вид с 4 м над началом пути на его середину."""
    gt = np.load(ROOT / "data" / "sessions" / session / "gt.npz")
    p = gt["pos"]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    views = {}
    for name, share in (("route_30", 0.3), ("route_70", 0.7)):
        i = int(np.searchsorted(s, share * s[-1]))
        j = int(np.searchsorted(s, min(s[i] + 0.5, s[-1])))
        d = p[min(j, len(p) - 1), :2] - p[max(i - 1, 0), :2]
        d = d / max(np.linalg.norm(d), 1e-6)
        eye = (p[i, 0], p[i, 1], 1.6)
        views[name] = (eye, (p[i, 0] + 4 * d[0], p[i, 1] + 4 * d[1], 0.8))
    c = p.mean(0)
    views["overview"] = ((p[0, 0], p[0, 1], 4.0), (c[0], c[1], 0.0))
    return views


def render(scene_path, size=(800, 500)):
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene_path)):
        raise RuntimeError(f"не открылась {scene_path}")
    for _ in range(10):
        app.update()
    while is_stage_loading():  # большое окружение с сервера грузится дольше
        app.update()
    stage = ctx.get_stage()
    annots = {}
    for name, (eye, target) in VIEWS.items():
        cam = UsdGeom.Camera.Define(stage, f"/CompareCams/{name}")
        cam.CreateFocalLengthAttr(12.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 100.0))
        f = (Gf.Vec3d(*target) - Gf.Vec3d(*eye)).GetNormalized()
        right = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized()
        up = Gf.Cross(right, f)
        m = Gf.Matrix4d(*right, 0, *up, 0, *(-f), 0, *eye, 1)
        UsdGeom.Xformable(cam).AddTransformOp().Set(m)
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rep.create.render_product(str(cam.GetPath()), size)])
        annots[name] = annot
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=8)
    return {n: Image.fromarray(np.array(a.get_data())[..., :3]) for n, a in annots.items()}


def main():
    global VIEWS
    recon = Path(sys.argv[1]).resolve()
    original = ORIGINAL
    if SESSION:
        meta = json.loads((ROOT / "data" / "sessions" / SESSION / "meta.json").read_text(encoding="utf-8"))
        if meta.get("environment"):  # сессия робота: оригинал — само окружение, ракурсы — по пути
            original = meta["environment"]
            VIEWS = route_views(SESSION)
    if VIEWS is None:
        raise SystemExit(f"нет набора ракурсов {VIEW_SET}")
    rows = [("original scene", render(original)), (f"reconstruction: {recon.stem}", render(recon))]
    w, h = 800, 500
    canvas = Image.new("RGB", (w * len(VIEWS), (h + 24) * len(rows)), (24, 24, 28))
    for r, (label, imgs) in enumerate(rows):
        ImageDraw.Draw(canvas).text((8, r * (h + 24) + 6), label, fill=(230, 230, 230))
        for c, name in enumerate(VIEWS):
            canvas.paste(imgs[name], (c * w, r * (h + 24) + 24))
    out = ROOT / "out" / f"compare_{recon.stem}{'' if VIEW_SET == 'high' else '_' + VIEW_SET}.png"
    canvas.save(out)
    print(f"[compare] {out}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
