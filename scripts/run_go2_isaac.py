#!/usr/bin/env python3
"""Isaac Sim: Go2 (IsaacLegs) с надстройкой, наши Mid-360 и Insta360 — в окружении Isaac.

Открывает robot/go2_mapping/go2_mapping.usda (scripts/build_go2_mapping.py), добавляет
окружение с сервера ассетов, ставит робота в --spawn, включает мост ROS 2 и запускает:
  * драйвер лепестковой развёртки Mid-360 (scripts/mid360_driver.py);
  * публикацию (scripts/ros2_sensors.py): /livox/lidar, /insta360/front|back/image_raw,
    /tf (world -> base_link, livox_frame, insta360_front, insta360_back), /clock.
Графы IsaacLegs работают как есть: /joint_states, /joint_command, /imu, /odom — по ним
ходит контроллер политики (ros2 launch fullbody_controller go2.launch.py). Головной лидар и
камера IsaacLegs (/point_cloud, /scan, /rgb) по умолчанию выключены — у нас свои датчики,
а рендер лишних тянет кадры вниз (--isaaclegs-sensors — оставить).

--record ИМЯ — записать поездку в data/drives/ИМЯ/: позы всех звеньев робота в каждом кадре
(motion.npz) и что за сцена (drive.json). По записи scripts/render_drive.py потом перерисует
ту же поездку с полными датчиками — уже без физики и не в реальном времени.

Обычно запускается из scripts/go2_warehouse.sh вместе с контроллером, телеуправлением и RViz.
Запуск отдельно: ~/isaacsim/python.sh scripts/run_go2_isaac.py [--env full_warehouse]
        [--spawn X Y YAW_DEG] [--image-size 480] [--record ИМЯ] [--headless] [--no-play]
Камеры для езды вживую — 480 px: на 960 кадров в секунду в полтора раза меньше (склад: 39 -> 24).
"""

import argparse
import json
import signal
import sys
import time
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--scene", default=str(ROOT / "robot" / "go2_mapping" / "go2_mapping.usda"))
parser.add_argument("--env", default="full_warehouse", help="окружение (см. go2_scene.ENVIRONMENTS) или путь к .usd")
parser.add_argument("--spawn", type=float, nargs=3, default=None, metavar=("X", "Y", "YAW"),
                    help="где поставить робота, м и градусы (по умолчанию — где стоит в сцене, в начале координат)")
parser.add_argument("--image-size", type=int, default=480, help="сторона кадра fisheye для ROS, пикселей; 0 — без камер (быстрее)")
parser.add_argument("--image-every", type=int, default=6, help="публиковать каждый N-й кадр рендера")
parser.add_argument("--isaaclegs-sensors", action="store_true", help="оставить головной лидар и камеру IsaacLegs")
parser.add_argument("--record", metavar="ИМЯ", help="записать поездку в data/drives/ИМЯ")
parser.add_argument("--headless", action="store_true")
parser.add_argument("--no-play", action="store_true", help="не запускать симуляцию сразу (Play в окне)")
parser.add_argument("--seconds", type=float, default=0.0, help="выйти через столько секунд симуляции (для проверки)")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720, "sync_loads": True,
                     "renderer": "RaytracedLighting"})

# Остановка по Ctrl+C / SIGTERM — флагом, который цикл проверяет между кадрами: исключение
# из обработчика могло бы прилететь внутрь чужого колбэка Kit и потеряться, а запись надо
# сохранить. Фоновым процессам оболочка выставляет SIGINT «игнорировать» — возвращаем свой.
STOP = []
signal.signal(signal.SIGINT, lambda *_: STOP.append(1))
signal.signal(signal.SIGTERM, lambda *_: STOP.append(1))

import numpy as np  # noqa: E402
import omni.physx  # noqa: E402
import omni.timeline  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
enable_extension("isaacsim.ros2.bridge")
app.update()
import go2_scene  # noqa: E402
import mid360_driver  # noqa: E402
import ros2_sensors  # noqa: E402


class MotionRecorder:
    """Позы звеньев робота (из PhysX, мир) после каждого кадра: t, pos [N,K,3], quat_xyzw [N,K,4]."""

    def __init__(self, stage, name, info, save_every_s=10.0):
        self.out = ROOT / "data" / "drives" / name
        if (self.out / "motion.npz").exists():
            raise SystemExit(f"запись {self.out} уже есть — выберите другое имя")
        self.out.mkdir(parents=True, exist_ok=True)
        self.paths = go2_scene.robot_links(stage)
        self.physx = omni.physx.get_physx_interface()
        self.t, self.pos, self.quat = [], [], []
        self.info = {**info, "links": self.paths, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.save_every, self.last_save = save_every_s, time.time()
        print(f"[go2] запись поездки: {self.out} ({len(self.paths)} звеньев)", flush=True)

    def sample(self, t):
        if self.t and t <= self.t[-1]:
            return  # время не шло (пауза) — новый кадр не пишем
        pos, quat = np.empty((len(self.paths), 3)), np.empty((len(self.paths), 4))
        for k, p in enumerate(self.paths):
            r = self.physx.get_rigidbody_transformation(p)
            pos[k], quat[k] = r["position"], r["rotation"]  # кватернион x, y, z, w
        self.t.append(t)
        self.pos.append(pos)
        self.quat.append(quat)
        if time.time() - self.last_save > self.save_every:
            self.save()

    def save(self):
        self.last_save = time.time()
        if not self.t:
            return
        tmp = self.out / "motion.tmp.npz"
        np.savez_compressed(tmp, t=np.array(self.t), paths=np.array(self.paths), pos=np.array(self.pos),
                            quat_xyzw=np.array(self.quat).astype(np.float32))
        tmp.replace(self.out / "motion.npz")
        info = {**self.info, "frames": len(self.t), "duration_s": round(self.t[-1] - self.t[0], 3)}
        (self.out / "drive.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    stage, url = go2_scene.open_scene(app, args.env, args.scene)
    go2_scene.set_spawn(stage, args.spawn)
    app.update()
    if not args.isaaclegs_sensors:
        go2_scene.disable_graphs(["/Graphs/ROS_LidarRTX", "/Graphs/ROS_Camera"])

    lidar = go2_scene.lidar_path(stage)
    lenses = {go2_scene.LENSES["front"]: ("insta360/front/image_raw", "insta360_front"),
              go2_scene.LENSES["back"]: ("insta360/back/image_raw", "insta360_back")}
    if args.image_size <= 0:
        lenses = {}  # камеры в сцене есть, но не рендерятся и не публикуются
    frames = {go2_scene.BASE: "base_link", lidar: "livox_frame", **{p: f for p, (_, f) in lenses.items()}}
    mid360_driver.start(stage)
    ros2_sensors.start(stage, image_size=args.image_size, image_every_n_frames=args.image_every,
                       lidar=lidar, lenses=lenses, frames=frames, parent="/World")

    recorder = None
    if args.record:
        recorder = MotionRecorder(stage, args.record, {
            "scene": str(Path(args.scene).resolve()), "environment": url, "env_name": args.env,
            "spawn": args.spawn, "dt_render_s": 1.0 / 60.0})
    timeline = omni.timeline.get_timeline_interface()
    if not args.no_play:
        timeline.play()
    print("[go2] готово: симуляция идёт" if not args.no_play else "[go2] готово: нажмите Play", flush=True)
    print("[go2] ходить: ros2 launch fullbody_controller go2.launch.py; руль: scripts/teleop_wasd.py", flush=True)
    t_end = args.seconds if args.seconds > 0 else None
    try:
        while app.is_running() and not STOP:
            app.update()
            if recorder is not None and timeline.is_playing():
                recorder.sample(timeline.get_current_time())
            if t_end is not None and timeline.get_current_time() >= t_end:
                break
    finally:
        if recorder is not None:
            recorder.save()
            print(f"[go2] поездка записана: {recorder.out} — кадров {len(recorder.t)}, "
                  f"{(recorder.t[-1] - recorder.t[0]) if recorder.t else 0:.1f} с симуляции. "
                  f"Дальше: scripts/process_drive.sh {args.record}", flush=True)
    timeline.stop()


try:
    main()
except SystemExit as e:
    print(f"[go2] {e}", flush=True)
except KeyboardInterrupt:
    print("[go2] остановлено", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    mid360_driver.stop()
    ros2_sensors.stop()
    app.close()
