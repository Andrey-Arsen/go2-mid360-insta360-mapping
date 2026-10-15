#!/usr/bin/env python3
"""Прогон сцены simple_room_mid360_fisheye.usda: облако Mid-360 и кадры обоих объективов.

Пишет в out/room/: cloud_world.ply (полные сканы в мировых координатах),
lens_front.png, lens_back.png, preview.png и report.json. В отчёте — точки на
скан, доля точек внутри комнаты и доля кадра, занятая кругом fisheye.

Запуск: ~/isaacsim/python.sh scripts/run_room.py [--seconds 2]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--seconds", type=float, default=2.0, help="сколько секунд писать лидар")
parser.add_argument("--warmup", type=int, default=30, help="кадров на прогрев рендера")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid360_driver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / "simple_room_mid360_fisheye.usda"
OUT = ROOT / "out" / "room"
LENSES = ("/World/Rig/Insta360/lens_front", "/World/Rig/Insta360/lens_back")
ROOM_BOUNDS = ((-4.6, 4.6), (-3.5, 5.0), (-0.05, 4.3))  # с запасом на стены и оконные рамы
DT = 1.0 / 60.0


def write_ply(path, pts):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(pts, dtype="<f4").tobytes())


def top_view(world, scale=50):
    (x0, x1), (y0, y1), _ = ROOM_BOUNDS
    w, h = int((x1 - x0) * scale), int((y1 - y0) * scale)
    img = np.zeros((h, w), np.uint8)
    u = ((world[:, 0] - x0) * scale).astype(int)
    v = ((y1 - world[:, 1]) * scale).astype(int)
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    np.add.at(img, (v[ok], u[ok]), 1)
    return np.clip(img.astype(np.int32) * 40, 0, 255).astype(np.uint8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(SCENE)):
        raise RuntimeError(f"не открылась сцена {SCENE}")
    stage = ctx.get_stage()
    for _ in range(10):
        app.update()

    lidar = next(p for p in stage.Traverse() if p.GetTypeName() == "OmniLidar")
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    mid360_driver.start(stage)

    lidar_rp = rep.create.render_product(
        str(lidar.GetPath()), resolution=(32, 32), render_vars=["GenericModelOutput", "RtxSensorMetadata"]
    )
    scan_annot = rep.AnnotatorRegistry.get_annotator(
        "IsaacCreateRTXLidarScanBuffer", init_params={"outputDistance": True}
    )
    scan_annot.attach([lidar_rp])
    rgb = {}
    for path in LENSES:
        res = int(stage.GetPrimAtPath(path).GetAttribute("omni:lensdistortion:ftheta:nominalWidth").Get())
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rep.create.render_product(path, resolution=(res, res))])
        rgb[path] = annot

    sim.initialize_physics()
    sim.play()
    for _ in range(args.warmup):
        sim.step(render=True)

    scans = []
    for _ in range(int(round(args.seconds / DT))):
        sim.step(render=True)
        data = scan_annot.get_data()
        pts = data.get("data") if isinstance(data, dict) else None
        if pts is None or not len(pts):
            continue
        if scans and len(pts) == len(scans[-1][0]) and np.array_equal(pts, scans[-1][0]):
            continue  # скан ещё не сменился
        scans.append((np.array(pts, dtype=np.float64), np.asarray(data["distance"]) > 0))
    images = {path: np.array(a.get_data())[..., :3] for path, a in rgb.items()}
    sim.stop()
    if len(scans) < 2:
        raise RuntimeError(f"лидар выдал {len(scans)} сканов")

    scans = scans[1:]  # первый скан начат до записи
    local = np.concatenate([p[v] for p, v in scans])
    m = np.array(UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world = (np.c_[local, np.ones(len(local))] @ m)[:, :3]
    inside = np.ones(len(world), bool)
    for axis, (lo, hi) in enumerate(ROOM_BOUNDS):
        inside &= (world[:, axis] >= lo) & (world[:, axis] <= hi)

    report = {
        "scene": str(SCENE),
        "lidar_world_position": [round(float(v), 3) for v in m[3, :3]],
        "scans": len(scans),
        "points_per_scan": int(np.mean([v.sum() for _, v in scans])),
        "invalid_per_scan": float(np.mean([(~v).sum() for _, v in scans])),
        "share_inside_room": round(float(inside.mean()), 4),
        "z_range_m": [round(float(world[:, 2].min()), 3), round(float(world[:, 2].max()), 3)],
    }
    write_ply(OUT / "cloud_world.ply", world.astype(np.float32))

    tiles = []
    for path, img in images.items():
        name = path.rsplit("/", 1)[-1]
        Image.fromarray(img).save(OUT / f"{name}.png")
        lit = img.max(axis=2) > 8
        yy, xx = np.nonzero(lit)
        c = img.shape[0] / 2.0
        r = np.hypot(xx + 0.5 - c, yy + 0.5 - c)
        report[name] = {
            "size_px": list(img.shape[:2]),
            "lit_share": round(float(lit.mean()), 3),  # круг на всю ширину квадрата = pi/4 = 0.785
            "circle_radius_px_p99": round(float(np.percentile(r, 99)), 1) if len(r) else 0.0,
            "mean_brightness": round(float(img.mean()), 1),
        }
        tiles.append(Image.fromarray(img).resize((640, 640)))
    preview = Image.new("RGB", (1280, 640 + 10 + 450))
    preview.paste(tiles[0], (0, 0))
    preview.paste(tiles[1], (640, 0))
    top = Image.fromarray(top_view(world)).convert("RGB")
    top.thumbnail((1280, 450))
    preview.paste(top, (0, 650))
    preview.save(OUT / "preview.png")

    text = json.dumps(report, ensure_ascii=False, indent=2)
    (OUT / "report.json").write_text(text + "\n", encoding="utf-8")
    print("ROOM_REPORT " + text, flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
