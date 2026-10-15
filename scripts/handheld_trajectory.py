"""Маршрут «ручного сканера» по Simple Room и IMU, посчитанный из него же.

Поза задаётся для «носителя» — точки в оптическом центре Mid-360 с осями стойки
(x вперёд, z вверх). Если лидар на стойке наклонён, его поза = носитель * поворот
установки (mount); IMU стоит в livox_frame, как в ассете. Стойка обходит стол по замкнутому сплайну на
высоте ~1.3 м, смотрит по ходу движения и покачивается в такт шагам. В начале и
в конце — покой: RKO-LIO по нему оценивает гравитацию.

IMU считается численным дифференцированием позы на сетке 1 кГц: угловая скорость
в осях датчика и удельная сила R^T (a - g) — в покое +g по z (REP-145). Шум и
смещения нулей — порядка ICM-40609 внутри Mid-360.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

G = 9.81
# Точки обхода (м, пол комнаты на z = 0): стол в центре ±1.6 x ±0.8, стены x ±4.45,
# задняя стена y = -3.35, окно y = 4.85.
WAYPOINTS = [(-3.0, -2.2), (0.0, -2.4), (3.0, -2.2), (3.4, 0.6), (3.0, 3.4), (0.0, 3.6), (-3.0, 3.4), (-3.4, 0.6)]
TABLE_HALF = (1.6, 0.8)
WALLS = (-4.45, 4.45, -3.35, 4.85)


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


class HandheldTrajectory:
    def __init__(self, height=1.3, speed=0.5, rest_start=3.0, rest_end=2.0, ramp=2.0, laps=1.0,
                 step_hz=1.8, bob_m=0.02, roll_deg=1.5, pitch_deg=1.0, rate_hz=1000.0):
        pts = np.array(WAYPOINTS + WAYPOINTS[:1], dtype=float)
        knots = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        self.spline = CubicSpline(knots, pts, bc_type="periodic")
        self.period = knots[-1]
        uu = np.linspace(0.0, self.period, 20001)
        du = np.linalg.norm(self.spline(uu, 1), axis=1)
        arc = np.r_[0.0, np.cumsum(0.5 * (du[1:] + du[:-1]) * np.diff(uu))]
        self.loop_length = arc[-1]
        self._u_of_s = lambda s: np.interp(np.mod(s, self.loop_length), arc, uu)
        self.height, self.speed = height, speed
        self.step_hz, self.bob_m = step_hz, bob_m
        self.roll, self.pitch = np.radians(roll_deg), np.radians(pitch_deg)

        # Профиль скорости: покой, плавный разгон, ход, плавное торможение, покой.
        length = laps * self.loop_length
        cruise = length / speed  # разгон и торможение вместе проходят столько же, сколько ramp секунд хода
        self.t_move0 = rest_start
        self.t_move1 = rest_start + ramp + cruise
        self.duration = self.t_move1 + rest_end
        self.dt = 1.0 / rate_hz
        self.t = np.arange(0.0, self.duration + 3 * self.dt, self.dt)
        self.v = speed * _smoothstep((self.t - self.t_move0) / ramp) * _smoothstep((self.t_move1 - self.t) / ramp)
        self.s = np.r_[0.0, np.cumsum(0.5 * (self.v[1:] + self.v[:-1]) * self.dt)]

    # --- поза ---------------------------------------------------------------
    def _path(self, t):
        s = np.interp(t, self.t, self.s)
        v = np.interp(t, self.t, self.v)
        u = self._u_of_s(s)
        xy = self.spline(u)
        d = self.spline(u, 1)
        return xy, np.arctan2(d[..., 1], d[..., 0]), v / self.speed

    def pose(self, t):
        """Поза носителя в мире: (позиция [N,3], Rotation)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        xy, yaw, walk = self._path(t)
        yaw = np.unwrap(yaw) if len(yaw) > 1 else yaw
        phase = 2.0 * np.pi * self.step_hz * t
        z = self.height + self.bob_m * walk * np.sin(phase)
        roll = self.roll * walk * np.sin(0.5 * phase)  # раскачка с шагом каждой ноги
        pitch = self.pitch * walk * np.sin(phase + 0.7)
        pos = np.c_[xy, z]
        rot = Rotation.from_euler("ZYX", np.c_[yaw, pitch, roll])
        return pos, rot

    # --- IMU ------------------------------------------------------------------
    def imu(self, rate_hz=200.0, noise=True, seed=0, mount=None):
        """Отсчёты IMU в livox_frame: t, gyro [рад/с], accel [м/с²] — с шумом и смещениями нулей; плюс чистые.

        mount — поворот лидара относительно носителя (Rotation); центр у них общий.
        """
        pos, rot = self.pose(self.t)
        if mount is not None:
            rot = rot * mount
        return imu_from_poses(self.t, pos, rot, rate_hz, noise, seed)

    def clearance(self):
        """Наименьшие зазоры до стола и стен вдоль всего маршрута, м."""
        pos, _ = self.pose(self.t[:: 50])
        x, y = pos[:, 0], pos[:, 1]
        dx = np.maximum(np.abs(x) - TABLE_HALF[0], 0.0)
        dy = np.maximum(np.abs(y) - TABLE_HALF[1], 0.0)
        table = np.min(np.hypot(dx, dy))
        walls = np.min(np.r_[x - WALLS[0], WALLS[1] - x, y - WALLS[2], WALLS[3] - y])
        return float(table), float(walls)


def imu_from_poses(t, pos, rot, rate_hz=200.0, noise=True, seed=0):
    """IMU датчика по его позам на частой равномерной сетке t (шаг << 1/rate_hz):
    t, gyro [рад/с], accel [м/с²] в осях датчика — с шумом ICM-40609 и смещениями нулей; плюс чистые.
    Акселерометр меряет удельную силу: a − g (в покое +g по оси вверх, REP-145)."""
    dt = float(t[1] - t[0])
    acc_w = np.gradient(np.gradient(pos, dt, axis=0), dt, axis=0)
    rel = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt  # угловая скорость в осях датчика, между узлами
    gyro = np.vstack([rel[:1], 0.5 * (rel[1:] + rel[:-1]), rel[-1:]])
    f_w = acc_w - np.array([0.0, 0.0, -G])
    accel = rot.inv().apply(f_w)
    step = max(1, int(round(1.0 / (rate_hz * dt))))
    edge = 3  # края численной производной отбрасываем
    sel = np.arange(edge, len(t) - edge, step)
    out = {"t": t[sel], "gyro_true": gyro[sel], "accel_true": accel[sel]}
    rng = np.random.default_rng(seed)
    if noise:
        # ICM-40609: шум гироскопа ~0.0038 град/с/√Гц, акселерометра ~100 мкg/√Гц.
        g_sigma = np.radians(0.0038) * np.sqrt(rate_hz)
        a_sigma = 100e-6 * G * np.sqrt(rate_hz)
        g_bias = rng.normal(0.0, 0.002, 3)
        a_bias = rng.normal(0.0, 0.03, 3)
        out["gyro"] = out["gyro_true"] + g_bias + rng.normal(0.0, g_sigma, out["gyro_true"].shape)
        out["accel"] = out["accel_true"] + a_bias + rng.normal(0.0, a_sigma, out["accel_true"].shape)
        out["gyro_bias"], out["accel_bias"] = g_bias, a_bias
    else:
        out["gyro"], out["accel"] = out["gyro_true"].copy(), out["accel_true"].copy()
    return out
