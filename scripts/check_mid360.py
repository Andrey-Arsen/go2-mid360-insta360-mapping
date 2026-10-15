#!/usr/bin/env python3
"""Проверка ассета Mid-360 из 123tthh/Livox_MID360_IsaacSim на установленном Isaac Sim.

Ставит лидар в коробку 8 x 6 x 3 м, запускает драйвер узора и собирает полные
сканы за --seconds секунд симуляции. Печатает точки на скан, долю точек на
стенах коробки, углы места и рост покрытия по направлениям: у лепесткового
узора оно растёт все 4 с, а если встало после 0.1 с — узор не подменяется.

Сканы берутся аннотатором IsaacCreateRTXLidarScanBuffer. Покадровый
IsaacExtractRTXSensorPointCloudNoAccumulator в 5.0 иногда отдаёт прошлый кадр
повторно и теряет следующий — для записи данных он не годится.

Запуск: ~/isaacsim/python.sh scripts/check_mid360.py [--profile petal|rotary]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--profile", choices=("petal", "rotary"), default="petal")
parser.add_argument("--seconds", type=float, default=4.5)
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid360_driver  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party" / "Livox_MID360_IsaacSim"
ASSETS = {
    "petal": REPO / "assets/petal_scan/standalone/Livox_MID360_Petal_Scan.usd",
    "rotary": REPO / "assets/rotary_scan/standalone/Livox_MID360_Rotary_Scan.usd",
}
OUT = ROOT / "out"

ROOM = (8.0, 6.0, 3.0)  # внутренние размеры коробки, м
WALL = 0.1
SENSOR_POS = (0.4, -0.3, 1.0)  # не в центре, чтобы стены были на разной дальности
DT = 1.0 / 60.0


def build_box_room(stage):
    sx, sy, sz = ROOM
    parts = {
        "floor": ((0, 0, -WALL / 2), (sx + 2 * WALL, sy + 2 * WALL, WALL)),
        "ceiling": ((0, 0, sz + WALL / 2), (sx + 2 * WALL, sy + 2 * WALL, WALL)),
        "wall_xp": ((sx / 2 + WALL / 2, 0, sz / 2), (WALL, sy, sz)),
        "wall_xn": ((-sx / 2 - WALL / 2, 0, sz / 2), (WALL, sy, sz)),
        "wall_yp": ((0, sy / 2 + WALL / 2, sz / 2), (sx, WALL, sz)),
        "wall_yn": ((0, -sy / 2 - WALL / 2, sz / 2), (sx, WALL, sz)),
    }
    for name, (pos, size) in parts.items():
        cube = UsdGeom.Cube.Define(stage, f"/World/Room/{name}")
        cube.CreateSizeAttr(1.0)
        xf = UsdGeom.XformCommonAPI(cube)
        xf.SetTranslate(Gf.Vec3d(*pos))
        xf.SetScale(Gf.Vec3f(*size))
    UsdLux.DistantLight.Define(stage, "/World/Light").CreateIntensityAttr(1000.0)


def surface_distance(pts):
    """Расстояние от точки до ближайшей внутренней грани коробки."""
    sx, sy, sz = ROOM
    return np.min(
        np.stack(
            [
                np.abs(sx / 2 - np.abs(pts[:, 0])),
                np.abs(sy / 2 - np.abs(pts[:, 1])),
                np.abs(pts[:, 2]),
                np.abs(sz - pts[:, 2]),
            ]
        ),
        axis=0,
    )


def write_ply(path, pts):
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with open(path, "wb") as f:
        f.write(header.encode())
        f.write(np.ascontiguousarray(pts, dtype="<f4").tobytes())


def save_preview(path, az, el, world):
    """Сверху — узор лучей (азимут/угол места), снизу — вид сверху на облако."""
    w, h = 1440, 300
    left = np.zeros((h, w), np.uint8)
    u = ((az + 180.0) / 360.0 * (w - 1)).astype(int)
    v = ((60.0 - el) / 75.0 * (h - 1)).astype(int)  # от +60 до -15 градусов
    ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    left[v[ok], u[ok]] = 255
    s = 60  # пикселей на метр
    tw, th = int(ROOM[0] * s) + 20, int(ROOM[1] * s) + 20
    top = np.zeros((th, tw), np.uint8)
    tu = ((world[:, 0] + ROOM[0] / 2) * s + 10).astype(int)
    tv = ((ROOM[1] / 2 - world[:, 1]) * s + 10).astype(int)
    ok = (tu >= 0) & (tu < tw) & (tv >= 0) & (tv < th)
    top[tv[ok], tu[ok]] = 255
    canvas = np.zeros((h + th + 10, max(w, tw)), np.uint8)
    canvas[:h, :w] = left
    canvas[h + 10 :, :tw] = top
    Image.fromarray(canvas).save(path)


def main():
    OUT.mkdir(exist_ok=True)
    report = {"profile": args.profile, "kit": omni.kit.app.get_app().get_build_version()}

    ctx = omni.usd.get_context()
    ctx.new_stage()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/World").GetPrim())
    build_box_room(stage)

    holder = UsdGeom.Xform.Define(stage, "/World/MID360")
    holder.GetPrim().GetReferences().AddReference(str(ASSETS[args.profile]))
    holder.ClearXformOpOrder()
    holder.AddTranslateOp().Set(Gf.Vec3d(*SENSOR_POS))
    for _ in range(5):
        app.update()

    lidars = [p for p in stage.Traverse() if p.GetTypeName() == "OmniLidar"]
    if len(lidars) != 1:
        raise RuntimeError(f"ожидался один OmniLidar, найдено {len(lidars)}")
    lidar = lidars[0]
    core = "omni:sensor:Core:"
    for key in ("scanType", "numberOfEmitters", "scanRateBaseHz", "farRangeM"):
        value = lidar.GetAttribute(core + key).Get()
        report[key] = value.item() if isinstance(value, np.generic) else value
    report["scanType"] = str(report["scanType"])

    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    drivers = mid360_driver.start(stage) if args.profile == "petal" else []

    rp = rep.create.render_product(
        str(lidar.GetPath()), resolution=(32, 32), render_vars=["GenericModelOutput", "RtxSensorMetadata"]
    )
    annot = rep.AnnotatorRegistry.get_annotator(
        "IsaacCreateRTXLidarScanBuffer", init_params={"outputDistance": True}
    )
    annot.attach([rp])

    sim.initialize_physics()
    sim.play()
    scans = []
    for _ in range(int(round(args.seconds / DT))):
        sim.step(render=True)
        data = annot.get_data()
        pts = data.get("data") if isinstance(data, dict) else None
        if pts is None or not len(pts):
            continue
        if scans and len(pts) == len(scans[-1][1]) and np.array_equal(pts, scans[-1][1]):
            continue  # скан ещё не сменился
        dist = np.asarray(data.get("distance", np.ones(len(pts))))
        scans.append((sim.current_time, np.array(pts, dtype=np.float64), dist > 0))
    sim.stop()
    if not scans:
        raise RuntimeError("лидар не выдал ни одного скана")

    scans = scans[1:]  # первый скан неполный: таймлайн стартует посреди него
    times = np.concatenate([np.full(len(p), t) for t, p, _ in scans])
    valid = np.concatenate([v for _, _, v in scans])
    local = np.concatenate([p for _, p, _ in scans])
    report["scans"] = len(scans)
    report["points_per_scan"] = int(np.mean([len(p) for _, p, _ in scans]))
    report["invalid_points"] = int((~valid).sum())
    times, local = times[valid], local[valid]

    m = np.array(UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    world = (np.c_[local, np.ones(len(local))] @ m)[:, :3]
    report["points_on_walls_5cm"] = round(float(np.mean(surface_distance(world) < 0.05)), 4)

    az = np.degrees(np.arctan2(local[:, 1], local[:, 0]))
    el = np.degrees(np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1])))
    report["elevation_deg_p0.1_p99.9"] = [round(float(np.percentile(el, q)), 2) for q in (0.1, 99.9)]
    report["share_above_30deg"] = round(float(np.mean(el > 30.0)), 3)
    if drivers:
        n = min(len(scans), drivers[0].states)
        report["share_above_30deg_official"] = round(float(np.mean(drivers[0].el[:n] > 30.0)), 3)

    # Рост покрытия: сколько разных ячеек 0.25 x 0.25 градуса задето к моменту t.
    cell = np.floor((az + 180.0) / 0.25).astype(np.int64) * 10000 + np.floor((el + 90.0) / 0.25).astype(np.int64)
    t_rel = times - times.min()
    report["unique_directions_by_time"] = {
        f"{t:g}s": int(len(np.unique(cell[t_rel < t - 1e-6])))
        for t in (0.2, 0.5, 1.0, 2.0, 4.0)
        if t <= t_rel.max() + DT
    }

    tag = f"mid360_{args.profile}_boxroom"
    write_ply(OUT / f"{tag}.ply", world.astype(np.float32))
    save_preview(OUT / f"{tag}.png", az, el, world)
    text = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    (OUT / f"{tag}.json").write_text(text + "\n", encoding="utf-8")
    print("MID360_REPORT " + text, flush=True)


try:
    main()
except BaseException:
    # app.close() гасит процесс раньше, чем Python успевает напечатать ошибку.
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
