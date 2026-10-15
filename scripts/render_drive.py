#!/usr/bin/env python3
"""Перерисовка записанной поездки Go2 с полными датчиками -> сессия для обработки (pipeline.sh).

Вход: data/drives/<поездка>/ (scripts/go2_warehouse.sh ... --record <поездка>): позы всех
звеньев робота в каждом кадре и сцена (окружение). Физика здесь выключена: робот кадр за
кадром ставится в записанные позы, а датчики работают как на стенде (record_session.py) —
лидар полными сканами с временем каждой точки, оба fisheye в полном разрешении, без спешки.

Что пишется в data/sessions/<сессия>/ (формат record_session.py, обработка та же):
  lidar.npz   точки в кадре лидара, время каждой, яркость, начала сканов; точки самого робота
              (тело, ноги, надстройка) вырезаны рамкой в кадре лидара — как фильтр на роботе
  imu.npz     IMU в livox_frame, 200 Гц: из траектории лидара (сплайн по кадрам) + шум ICM-40609
  gt.npz      эталонная поза livox_frame, 200 Гц
  legs.npz    высота лидара над подошвой опорной ноги по кадрам (на настоящем Go2 — из углов
              суставов): по ней build_mesh.py кладёт пол под путём, которого лидар не видит
  images/     NNN_front.png, NNN_back.png и маски NNN_<объектив>_mask.png (0 — робот в кадре:
              сегментация по объектам Isaac, всё под /World/go2)
  meta.json   моменты съёмки, переходы лидар -> объективы, окружение и путь к его эталонному мешу

Начало сессии — первый покой робота после появления (по нему RKO-LIO выставляет IMU по
гравитации): корпус почти не движется --rest секунд подряд. Конец — конец записи.

Запуск: ~/isaacsim/python.sh scripts/render_drive.py <поездка> [--session ИМЯ] [--shutter-every 0.25]
        [--image-size 1920] [--seconds 0] [--force]
"""

import argparse
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("drive", help="имя поездки в data/drives/")
parser.add_argument("--session", help="имя сессии (по умолчанию — как поездка)")
parser.add_argument("--shutter-every", type=float, default=0.25, help="снимок обоими объективами раз в N секунд")
parser.add_argument("--image-size", type=int, default=1920, help="сторона кадра fisheye, пикселей")
parser.add_argument("--seconds", type=float, default=0.0, help="обрезать сессию (0 — до конца поездки)")
parser.add_argument("--rest", type=float, default=2.0, help="сколько секунд покоя нужно в начале")
parser.add_argument("--crop-margin", type=float, default=0.06, help="запас рамки вокруг робота для вырезки его точек, м")
parser.add_argument("--force", action="store_true", help="перезаписать сессию, если есть")
args, _ = parser.parse_known_args()

DRIVE = ROOT / "data" / "drives" / args.drive
if not (DRIVE / "motion.npz").exists():
    print(f"[render] нет записи {DRIVE}/motion.npz — сначала scripts/go2_warehouse.sh <окружение> --record {args.drive}")
    sys.exit(1)
OUT = ROOT / "data" / "sessions" / (args.session or args.drive)
if (OUT / "lidar.npz").exists() and not args.force:
    print(f"[render] сессия {OUT} уже есть (--force — перезаписать)")
    sys.exit(1)

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from PIL import Image  # noqa: E402
import carb  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.spatial.transform import Rotation, RotationSpline  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
enable_extension("isaacsim.ros2.bridge")  # графы IsaacLegs в сцене из узлов моста; сами графы выключим
app.update()
import go2_scene  # noqa: E402
import mid360_driver  # noqa: E402
from handheld_trajectory import imu_from_poses  # noqa: E402
from rtx_timing import (DT, REPLAY_CAMERA_LATENCY_FRAMES, REPLAY_SCAN_END_LAG_FRAMES, SCAN_FRAMES,  # noqa: E402
                        point_times)


def to_matrix(pos, quat_xyzw):
    T = np.tile(np.eye(4), (len(pos), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(quat_xyzw).as_matrix()
    T[:, :3, 3] = pos
    return T


class Motion:
    """Записанные позы звеньев; поза на любой момент — линейно/по сфере между кадрами записи."""

    def __init__(self, path):
        d = np.load(path)
        self.t, self.paths = d["t"], [str(p) for p in d["paths"]]
        self.pos, self.quat = d["pos"], d["quat_xyzw"].astype(np.float64)
        # Один знак кватерниона подряд — чтобы интерполяция шла коротким путём.
        for n in range(1, len(self.t)):
            self.quat[n, np.einsum("ki,ki->k", self.quat[n], self.quat[n - 1]) < 0] *= -1

    def at(self, t):
        """Мировые 4x4 всех звеньев на момент t записи."""
        i = int(np.clip(np.searchsorted(self.t, t) - 1, 0, len(self.t) - 2))
        a = float(np.clip((t - self.t[i]) / (self.t[i + 1] - self.t[i]), 0.0, 1.0))
        pos = (1 - a) * self.pos[i] + a * self.pos[i + 1]
        q = (1 - a) * self.quat[i] + a * self.quat[i + 1]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        return to_matrix(pos, q)

    def window(self, base, rest_s):
        """Начало: первый отрезок покоя rest_s после появления; конец — конец записи."""
        k = self.paths.index(base)
        p, t = self.pos[:, k], self.t
        v = np.linalg.norm(np.gradient(p, t, axis=0), axis=1)
        q = Rotation.from_quat(self.quat[:, k])
        w = np.r_[0.0, (q[:-1].inv() * q[1:]).magnitude() / np.diff(t)]
        calm = (v < 0.03) & (w < 0.1)
        dt = float(np.median(np.diff(t)))
        need = max(1, int(round(rest_s / dt)))
        run = 0
        for i in range(len(t)):
            run = run + 1 if calm[i] else 0
            if run >= need and t[i - need + 1] > t[0] + 0.5:  # первые полсекунды — падение с высоты появления
                return float(t[i - need + 1]), float(t[-1]), True
        # Покоя нет: берём самое спокойное окно в первые 20 с.
        span = min(len(t) - need, int(20 / dt))
        score = [np.mean(v[i:i + need]) + np.mean(w[i:i + need]) for i in range(max(1, span))]
        i = int(np.argmin(score))
        return float(t[i]), float(t[-1]), False


def robot_points(stage, root):
    """Вершины всех видимых сеток робота в мире (для рамки вырезки его точек из скана)."""
    pts = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(root), Usd.TraverseInstanceProxies()):
        if p.GetTypeName() != "Mesh":
            continue
        img = UsdGeom.Imageable(p)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() == UsdGeom.Tokens.guide:
            continue
        V = np.array(p.GetAttribute("points").Get() or [], float)
        if len(V):
            M = go2_scene.world_matrix(stage, str(p.GetPath()))
            pts.append(V @ M[:3, :3].T + M[:3, 3])
    return np.vstack(pts)


def main():
    drive = json.loads((DRIVE / "drive.json").read_text(encoding="utf-8"))
    motion = Motion(DRIVE / "motion.npz")
    t0, t1, rested = motion.window(go2_scene.BASE, args.rest)
    if args.seconds > 0:
        t1 = min(t1, t0 + args.seconds)
    duration = t1 - t0
    print(f"[render] поездка {args.drive}: {motion.t[-1] - motion.t[0]:.1f} с записи, сессия с {t0:.2f} с, "
          f"{duration:.1f} с{'' if rested else ' (ВНИМАНИЕ: покоя в начале нет — IMU выставится хуже)'}", flush=True)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    stage, url = go2_scene.open_scene(app, drive.get("environment"), drive["scene"])
    go2_scene.disable_graphs()
    # Физика роботу не нужна: он ставится в записанные позы. Снимаем её с робота (твёрдые тела,
    # артикуляция, суставы, IMU) на сессионном слое — он становится обычной иерархией, как стойка
    # стенда. Иначе физика шагает вместе с нами и тянет звенья по-своему (SimulationContext сам
    # включает playSimulations обратно), и позы на снимках и в сканах расходятся с записанными.
    robot_prims = list(Usd.PrimRange(stage.GetPrimAtPath(go2_scene.ROBOT)))
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for prim in robot_prims:
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        for prim in robot_prims:
            if prim.IsValid() and (prim.IsA(UsdPhysics.Joint) or "Imu" in prim.GetTypeName()):
                prim.SetActive(False)

    # Звенья — одной матрицей на сессионном слое (файл сцены не меняется).
    links = [p for p in motion.paths if stage.GetPrimAtPath(p).IsValid()]
    missing = set(motion.paths) - set(links)
    if missing:
        print(f"[render] в сцене нет {len(missing)} звеньев из записи: {sorted(missing)[:3]}...", flush=True)
    idx = {p: motion.paths.index(p) for p in links}
    ops, parent_static = {}, {}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for p in links:
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(p))
            xf.ClearXformOpOrder()
            ops[p] = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    app.update()
    for p in links:
        parent = str(stage.GetPrimAtPath(p).GetParent().GetPath())
        if parent not in idx:
            parent_static[p] = go2_scene.world_matrix(stage, parent)
    order = sorted(links, key=lambda p: p.count("/"))  # родители раньше детей

    def place(t_drive):
        W = motion.at(t_drive)
        with Sdf.ChangeBlock():
            for p in order:
                parent = str(Path(p).parent)
                P = W[idx[parent]] if parent in idx else parent_static[p]
                ops[p].Set(Gf.Matrix4d((np.linalg.inv(P) @ W[idx[p]]).T.tolist()))
        return W

    lidar = go2_scene.lidar_path(stage)
    W0 = place(t0)
    app.update()
    base_k = idx[go2_scene.BASE]
    T_base_lidar = np.linalg.inv(W0[base_k]) @ go2_scene.world_matrix(stage, lidar)
    # Подошва: центр звена ступни минус её радиус; радиус — по покою, когда низ сетки робота на полу.
    feet_k = [idx[p] for p in links if p.endswith("_foot")]
    foot_r = float(W0[feet_k, 2, 3].min() - robot_points(stage, go2_scene.ROBOT)[:, 2].min()) if feet_k else None
    T_lidar_lens = {k: np.linalg.inv(go2_scene.world_matrix(stage, lidar)) @ go2_scene.world_matrix(stage, p)
                    for k, p in go2_scene.LENSES.items()}

    # Рамка робота в кадре лидара: по позам с разных моментов поездки (ноги двигаются).
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for t in np.linspace(t0, t1, 12):
        W = place(t)
        app.update()
        L = W[base_k] @ T_base_lidar
        P = robot_points(stage, go2_scene.ROBOT)
        P = (P - L[:3, 3]) @ L[:3, :3]
        lo, hi = np.minimum(lo, P.min(0)), np.maximum(hi, P.max(0))
    lo, hi = lo - args.crop_margin, hi + args.crop_margin
    print(f"[render] рамка робота в кадре лидара, м: {np.round(lo, 2).tolist()} .. {np.round(hi, 2).tolist()}", flush=True)

    # Ждать конца рендера каждого кадра: иначе на тяжёлой сцене (склад, две камеры 1920, запись
    # снимков) данные лидара доходят до цикла с плавающей задержкой и время точек сбивается.
    # Задержки датчиков в этом режиме — REPLAY_* в rtx_timing.
    carb.settings.get_settings().set_bool("/app/hydraEngine/waitIdle", True)
    sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    mid360_driver.start(stage)
    lidar_rp = rep.create.render_product(lidar, [32, 32], render_vars=["GenericModelOutput", "RtxSensorMetadata"])
    scan_annot = rep.AnnotatorRegistry.get_annotator(
        "IsaacCreateRTXLidarScanBuffer",
        init_params={"outputTimestamp": True, "outputIntensity": True, "outputDistance": True},
    )
    scan_annot.attach([lidar_rp])
    rgb, seg = {}, {}
    for name, path in go2_scene.LENSES.items():
        rp = rep.create.render_product(path, (args.image_size, args.image_size))
        rgb[name] = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb[name].attach([rp])
        seg[name] = rep.AnnotatorRegistry.get_annotator("instance_id_segmentation", init_params={"colorize": False})
        seg[name].attach([rp])

    sim.initialize_physics()
    sim.play()
    pts, times, inten, starts = [], [], [], []
    dropped = anchored = anchor_shift = 0
    scan_kj = []  # шаг получения скана и кадр его позы (info.transform) — для проверки задержек
    prev_scan = None
    pending, shots = {}, []
    next_shot = 0.5
    steps = int(np.ceil(duration / DT)) + 2 * SCAN_FRAMES  # и ещё кадры, чтобы дошёл последний скан
    pose_t, pose_lidar, sole = [], [], []
    wall0 = time.time()
    for k in range(steps):
        t = sim.current_time
        W = place(t0 + min(t, duration))
        pose_t.append(t)
        pose_lidar.append(W[base_k] @ T_base_lidar)
        if feet_k:
            sole.append(W[feet_k, 2, 3].min() - foot_r)
        if next_shot <= t <= duration:
            pending[k + REPLAY_CAMERA_LATENCY_FRAMES] = t
            next_shot += args.shutter_every
        sim.step(render=True)

        if k in pending:
            n_shot = len(shots)
            for name in go2_scene.LENSES:
                Image.fromarray(np.array(rgb[name].get_data())[..., :3]).save(OUT / "images" / f"{n_shot:03d}_{name}.png")
                s = seg[name].get_data()
                ids = [int(i) for i, path in s["info"]["idToLabels"].items() if str(path).startswith(go2_scene.ROBOT)]
                robot = ndimage.binary_dilation(np.isin(np.asarray(s["data"]), ids), iterations=3)
                Image.fromarray(np.where(robot, 0, 255).astype(np.uint8)).save(OUT / "images" / f"{n_shot:03d}_{name}_mask.png")
            shots.append(pending.pop(k))

        data = scan_annot.get_data()
        p = data.get("data") if isinstance(data, dict) else None
        if p is None or not len(p) or (prev_scan is not None and len(p) == len(prev_scan) and np.array_equal(p, prev_scan)):
            continue
        prev_scan = np.array(p)
        n = len(prev_scan)
        ts = np.array(data["timestamp"]).view(np.uint32).astype(np.int64)  # в 5.0 это uint32 парами в uint64
        n = min(n, len(ts))
        # Скан несёт позу лидара на свой конец (info.transform): это поза шага k − 1 (rtx_timing).
        # Сверяем с позами, которые ставили, и время точек считаем от найденного кадра — если
        # задержка вдруг сдвинется, время останется верным, а счётчик расхождений попадёт в meta.json.
        j = k - REPLAY_SCAN_END_LAG_FRAMES
        T_info = np.array(data["info"]["transform"], dtype=float).reshape(4, 4).T
        d = [np.linalg.norm(pose_lidar[i][:3, 3] - T_info[:3, 3]) + np.linalg.norm(pose_lidar[i][:3, :3] - T_info[:3, :3])
             for i in range(max(0, k - 12), k + 1)]
        order_d = np.argsort(d)
        if d[order_d[0]] < 1e-4 and (len(d) < 2 or d[order_d[1]] > 1e-4):  # однозначно (стоя на месте — неоднозначно)
            j = max(0, k - 12) + int(order_d[0])
            anchored += 1
            if j != k - REPLAY_SCAN_END_LAG_FRAMES:
                anchor_shift += 1
        tp = point_times(pose_t[j], ts[:n], latency_frames=SCAN_FRAMES)  # скан кончился на позе шага j
        P = prev_scan[:n]
        keep = (np.asarray(data["distance"])[:n] > 0) & (tp >= 0.0) & (tp <= duration)
        if keep.sum() < 1000 or np.any(np.diff(tp) < -1e-9):
            continue
        self_hit = np.all((P > lo) & (P < hi), axis=1)
        dropped += int((keep & self_hit).sum())
        keep &= ~self_hit
        starts.append(sum(len(x) for x in pts))
        scan_kj.append((k, j))
        pts.append(P[keep].astype(np.float32))
        times.append(tp[keep])
        inten.append(np.asarray(data["intensity"])[:n][keep].astype(np.float32))
        if len(starts) % 50 == 0:
            done = min(t / duration, 1.0)
            el = time.time() - wall0
            print(f"[render] {t:.1f}/{duration:.1f} с, сканов {len(starts)}, снимков {len(shots)}, "
                  f"{(k + 1) / el:.1f} кадр/с, осталось ~{el / max(done, 1e-3) * (1 - done) / 60:.0f} мин", flush=True)
    sim.stop()
    mid360_driver.stop()

    np.savez(OUT / "lidar.npz", points=np.concatenate(pts), t=np.concatenate(times),
             intensity=np.concatenate(inten), scan_start=np.array(starts), scan_k_j=np.array(scan_kj))
    # IMU и эталон — из траектории лидара по кадрам (поза, поставленная перед шагом, — на время шага).
    pose_t, pose_lidar = np.array(pose_t), np.array(pose_lidar)
    sel = np.r_[True, np.diff(pose_t) > 1e-9]
    pose_t, pose_lidar = pose_t[sel], pose_lidar[sel]
    sensor_height = None
    if feet_k:
        h = pose_lidar[:, 2, 3] - np.array(sole)[sel]
        np.savez(OUT / "legs.npz", t=pose_t, sensor_height_m=h)
        sensor_height = round(float(np.median(h)), 4)
    pos_spline = CubicSpline(pose_t, pose_lidar[:, :3, 3])
    rot_spline = RotationSpline(pose_t, Rotation.from_matrix(pose_lidar[:, :3, :3]))
    fine = np.arange(0.0, duration + 1e-9, 0.001)
    imu = imu_from_poses(fine, pos_spline(fine), rot_spline(fine), rate_hz=200.0)
    np.savez(OUT / "imu.npz", **imu)
    np.savez(OUT / "gt.npz", t=imu["t"], pos=pos_spline(imu["t"]), quat_xyzw=rot_spline(imu["t"]).as_quat())

    env_stem = Path(url).stem if url else "none"
    lens_prim = stage.GetPrimAtPath(go2_scene.LENSES["front"])
    meta = {
        "scene": drive["scene"],
        "rig": "go2",
        "environment": url,
        "gt_mesh": f"out/gt/{env_stem}.ply" if url else None,
        "drive": args.drive,
        "drive_t0_s": t0,
        "duration_s": duration,
        "dt_render_s": DT,
        "rtx_latency_frames": {"wait_idle": True, "scan_end_lag": REPLAY_SCAN_END_LAG_FRAMES,
                               "scan_frames": SCAN_FRAMES, "camera": REPLAY_CAMERA_LATENCY_FRAMES},
        "lidar_scans_anchored": {"by_pose": anchored, "shifted_from_expected": anchor_shift},
        "shutter_times_s": shots,
        "lidar_frame": "livox_frame",
        "imu_frame": "livox_frame",
        "T_lidar_imu": np.eye(4).tolist(),
        "T_lidar_lens": {k: v.tolist() for k, v in T_lidar_lens.items()},
        "lens": {
            "model": "ftheta equidistant",
            "size_px": args.image_size,
            "fov_deg": float(lens_prim.GetAttribute("omni:lensdistortion:ftheta:maxFov").Get()),
            "axes": "USD camera: смотрит по -Z, верх +Y",
        },
        "image_masks": True,
        "trajectory": {"source": "drive", "drive": args.drive, "rested_at_start": rested},
        "lidar_crop_box_m": [lo.round(3).tolist(), hi.round(3).tolist()],
        "sensor_height_m": sensor_height,  # медиана legs.npz: лидар над полом на ходу
        "foot_radius_m": None if foot_r is None else round(foot_r, 4),
        "lidar_self_points_dropped": dropped,
        "imu_noise": {"gyro_bias": imu["gyro_bias"].tolist(), "accel_bias": imu["accel_bias"].tolist()},
        "intensity": "сырое значение RTX (~0.003..0.013), не отражательность Livox 0..255",
    }
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    total = sum(len(x) for x in pts)
    print(f"[render] готово: {OUT} — сканов {len(starts)}, точек {total} (робота вырезано {dropped}, "
          f"{dropped / max(total + dropped, 1):.1%}), снимков {len(shots)}, лидар над полом {sensor_height} м, "
          f"{(time.time() - wall0) / 60:.1f} мин; "
          f"сканов сверено по позе {anchored}, со сдвигом {anchor_shift}", flush=True)


try:
    main()
except SystemExit as e:
    print(f"[render] {e}", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
