#!/usr/bin/env python3
"""Записывает сессию «ручного сканера»: стойка идёт по маршруту, пишутся лидар, IMU, кадры и эталон.

Папка data/sessions/<name>/:
  lidar.npz   points (сенсор, float32), t (с, время каждой точки), intensity, scan_start (индексы)
  imu.npz     t, gyro, accel (с шумом), gyro_true, accel_true — 200 Гц
  gt.npz      t, pos, quat_xyzw — поза livox_frame в мире, 200 Гц
  images/     NNN_front.png, NNN_back.png — кадры объективов в моменты съёмки
  meta.json   моменты съёмки, переходы лидар -> объективы, параметры

Время — секунды симуляции. Задержки RTX измерены опытом на Isaac Sim 5.0 при шаге 1/60:
скан, выданный после шага с временем T, начался в T - 9 кадров, время точки внутри —
смещение от начала её кадра рендера; кадр камеры, прочитанный после шага при T,
показывает позу, заданную за 5 кадров до T.

Запуск: ~/isaacsim/python.sh scripts/record_session.py --name walk01 [--seconds 10]
"""

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
parser.add_argument("--scene", default="simple_room_mid360_fisheye.usda", help="сцена в scenes/ (наклон лидара задаётся в ней)")
parser.add_argument("--seconds", type=float, default=0.0, help="обрезать сессию; 0 — весь маршрут")
parser.add_argument("--height", type=float, default=1.3, help="высота лидара, м")
parser.add_argument("--speed", type=float, default=0.5, help="скорость ходьбы, м/с")
parser.add_argument("--shutter-every", type=float, default=1.0, help="снимать кадры раз в N секунд")
parser.add_argument("--image-size", type=int, default=1920)
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from PIL import Image  # noqa: E402
from pxr import Gf, Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mid360_driver  # noqa: E402
from handheld_trajectory import HandheldTrajectory  # noqa: E402
from rtx_timing import CAMERA_LATENCY_FRAMES, DT, LIDAR_LATENCY_FRAMES, point_times  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / args.scene
RIG = "/World/Rig"
LIDAR = "/World/Rig/MID360/torso_link/mid360_link/mid360_native_approx"
LENSES = {"front": "/World/Rig/Insta360/lens_front", "back": "/World/Rig/Insta360/lens_back"}


def main():
    out = ROOT / "data" / "sessions" / args.name
    (out / "images").mkdir(parents=True, exist_ok=True)
    traj = HandheldTrajectory(height=args.height, speed=args.speed)
    duration = min(traj.duration, args.seconds) if args.seconds else traj.duration

    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(SCENE)):
        raise RuntimeError(f"не открылась сцена {SCENE}")
    for _ in range(10):
        app.update()
    stage = ctx.get_stage()
    rig = stage.GetPrimAtPath(RIG)
    # Сессионный слой: сцена на диске не меняется, стойку ведём одной матрицей.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        rig_xf = UsdGeom.Xformable(rig)
        rig_xf.ClearXformOpOrder()
        rig_op = rig_xf.AddTransformOp()
        rig_op.Set(Gf.Matrix4d(1.0))
    app.update()
    lidar_prim = stage.GetPrimAtPath(LIDAR)
    T_rig_lidar = world_matrix(lidar_prim)  # стойка в нуле
    T_lidar_rig = np.linalg.inv(T_rig_lidar)
    T_lidar_lens = {k: T_lidar_rig @ world_matrix(stage.GetPrimAtPath(p)) for k, p in LENSES.items()}
    # Носитель маршрута — оптический центр лидара с осями стойки; лидар повёрнут относительно него.
    T_carrier_rig = np.eye(4)
    T_carrier_rig[:3, 3] = -T_rig_lidar[:3, 3]
    mount = Rotation.from_matrix(T_rig_lidar[:3, :3])

    def place_rig(t):
        pos, rot = traj.pose(t)
        T = np.eye(4)
        T[:3, :3] = rot.as_matrix()[0]
        T[:3, 3] = pos[0]
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            rig_op.Set(Gf.Matrix4d((T @ T_carrier_rig).T.tolist()))

    place_rig(0.0)
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    mid360_driver.start(stage)
    lidar_rp = rep.create.render_product(LIDAR, [32, 32], render_vars=["GenericModelOutput", "RtxSensorMetadata"])
    scan_annot = rep.AnnotatorRegistry.get_annotator(
        "IsaacCreateRTXLidarScanBuffer",
        init_params={"outputTimestamp": True, "outputIntensity": True, "outputDistance": True},
    )
    scan_annot.attach([lidar_rp])
    rgb = {}
    for name, path in LENSES.items():
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rep.create.render_product(path, (args.image_size, args.image_size))])
        rgb[name] = annot

    sim.initialize_physics()
    sim.play()
    pts, times, inten, starts = [], [], [], []
    prev_scan = None
    pending = {}  # шаг, после которого читать кадры -> время позы
    shots = []
    next_shot = 0.5
    steps = int(np.ceil((duration + (LIDAR_LATENCY_FRAMES + 2) * DT) / DT))
    wall0 = time.time()
    for k in range(steps):
        t = sim.current_time
        place_rig(min(t, traj.duration))
        if t >= next_shot and t <= duration:
            pending[k + CAMERA_LATENCY_FRAMES] = t
            next_shot += args.shutter_every
        sim.step(render=True)

        if k in pending:
            idx = len(shots)
            for name, annot in rgb.items():
                Image.fromarray(np.array(annot.get_data())[..., :3]).save(out / "images" / f"{idx:03d}_{name}.png")
            shots.append(pending.pop(k))

        data = scan_annot.get_data()
        p = data.get("data") if isinstance(data, dict) else None
        if p is None or not len(p) or (prev_scan is not None and len(p) == len(prev_scan) and np.array_equal(p, prev_scan)):
            continue
        prev_scan = np.array(p)
        n = len(prev_scan)
        ts = np.array(data["timestamp"]).view(np.uint32).astype(np.int64)  # в 5.0 это uint32 парами в uint64
        n = min(n, len(ts))  # при нечётном числе точек у последней нет времени
        tp = point_times(sim.current_time, ts[:n])
        keep = (np.asarray(data["distance"])[:n] > 0) & (tp >= 0.0) & (tp <= duration)
        if keep.sum() < 1000 or np.any(np.diff(tp) < -1e-9):
            continue  # неполный первый скан или скан за концом записи
        starts.append(sum(len(x) for x in pts))
        pts.append(prev_scan[:n][keep].astype(np.float32))
        times.append(tp[keep])
        inten.append(np.asarray(data["intensity"])[:n][keep].astype(np.float32))
        if len(starts) % 50 == 0:
            print(f"[record] t={sim.current_time:.1f}/{duration:.1f} с, сканов {len(starts)}, кадров {len(shots)}, "
                  f"{(k + 1) / (time.time() - wall0):.1f} шагов/с", flush=True)
    sim.stop()
    mid360_driver.stop()

    np.savez(out / "lidar.npz", points=np.concatenate(pts), t=np.concatenate(times),
             intensity=np.concatenate(inten), scan_start=np.array(starts))
    imu = traj.imu(mount=mount)
    sel = imu["t"] <= duration
    np.savez(out / "imu.npz", **{k: (v[sel] if v.ndim and len(v) == len(sel) else v) for k, v in imu.items()})
    gt_pos, gt_rot = traj.pose(imu["t"][sel])
    gt_rot = gt_rot * mount  # эталон — поза livox_frame
    np.savez(out / "gt.npz", t=imu["t"][sel], pos=gt_pos, quat_xyzw=gt_rot.as_quat())
    meta = {
        "scene": str(SCENE),
        "duration_s": duration,
        "dt_render_s": DT,
        "shutter_times_s": shots,
        "lidar_frame": "livox_frame",
        "imu_frame": "livox_frame",
        "T_lidar_imu": np.eye(4).tolist(),
        "T_lidar_lens": {k: v.tolist() for k, v in T_lidar_lens.items()},
        "lens": {
            "model": "ftheta equidistant",
            "size_px": args.image_size,
            "fov_deg": float(stage.GetPrimAtPath(LENSES["front"]).GetAttribute("omni:lensdistortion:ftheta:maxFov").Get()),
            "axes": "USD camera: смотрит по -Z, верх +Y",
        },
        "trajectory": {"height_m": args.height, "speed_mps": args.speed, "loop_length_m": traj.loop_length},
        "lidar_mount_rotvec_deg": np.degrees(mount.as_rotvec()).round(3).tolist(),
        "imu_noise": {"gyro_bias": imu["gyro_bias"].tolist(), "accel_bias": imu["accel_bias"].tolist()},
        "intensity": "сырое значение RTX (~0.003..0.013), не отражательность Livox 0..255",
    }
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[record] готово: {out} — сканов {len(starts)}, точек {sum(len(x) for x in pts)}, кадров {len(shots)}, "
          f"{time.time() - wall0:.0f} с", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
