"""Драйвер лепесткового узора Mid-360 для ассета 123tthh/Livox_MID360_IsaacSim (MIT).

Приём тот же, что в их drive_mid360_petal_scan_isaacsim51.py: каждые 0.1 с
подменять 20 000 направлений лучей OmniLidar следующим куском официального
узора Livox (40 кусков = 4 с, дальше повтор). Здесь углы распакованы в numpy
один раз, драйвер можно запускать из своего скрипта и останавливать.

Script Editor: выполнить файл целиком, затем Play — заодно во вьюпорте рисуются
точки и лучи; остановить — stop(). Из своего скрипта: mid360_driver.start(stage).
"""

import zlib

import numpy as np
import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, Vt

AZ_ATTR = "omni:sensor:Core:emitterState:s001:azimuthDeg"
EL_ATTR = "omni:sensor:Core:emitterState:s001:elevationDeg"
PATTERN_SCOPE = "mid360_nonrepetitive_pattern"
STATE_PERIOD_S = 0.1

_active = []
_drawn = []
_rays = []


class PatternDriver:
    def __init__(self, stage, lidar):
        pattern = lidar.GetParent().GetChild(PATTERN_SCOPE)
        raw = zlib.decompress(bytes(pattern.GetAttribute("lidarHiking:compressedDirections").Get()))
        values = np.frombuffer(raw, dtype="<u2").astype(np.float32).reshape(-1, 2)
        self.states = int(pattern.GetAttribute("lidarHiking:trajectoryStates").Get())
        per_state = int(pattern.GetAttribute("lidarHiking:pointsPerState").Get())
        az = values[:, 0] * 0.01
        self.az = np.where(az < 180.0, az, az - 360.0).reshape(self.states, per_state)
        self.el = (values[:, 1] * 0.01 - 10.0).reshape(self.states, per_state)
        self.stage = stage
        self.path = lidar.GetPath()
        self.current = None
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name=f"mid360 pattern {self.path}"
        )

    def _on_update(self, _event):
        timeline = omni.timeline.get_timeline_interface()
        if timeline.is_playing():
            state = int(timeline.get_current_time() / STATE_PERIOD_S + 1e-6) % self.states
            if state != self.current:
                self.apply(state)

    def apply(self, state):
        lidar = self.stage.GetPrimAtPath(self.path)
        # Сессионный слой: при сохранении сцены рабочие углы не попадут в файл.
        with Usd.EditContext(self.stage, self.stage.GetSessionLayer()):
            lidar.GetAttribute(AZ_ATTR).Set(Vt.FloatArray.FromNumpy(self.az[state]))
            lidar.GetAttribute(EL_ATTR).Set(Vt.FloatArray.FromNumpy(self.el[state]))
        self.current = state

    def stop(self):
        self.sub = None
        layer = self.stage.GetSessionLayer()
        prim_spec = layer.GetPrimAtPath(self.path)
        if prim_spec is not None:
            for name in (AZ_ATTR, EL_ATTR):
                if name in prim_spec.properties:
                    prim_spec.RemoveProperty(prim_spec.properties[name])


def _mid360_lidars(stage):
    lidars = [
        p
        for p in stage.Traverse()
        if p.GetTypeName() == "OmniLidar" and p.GetParent().GetChild(PATTERN_SCOPE).IsValid()
    ]
    if not lidars:
        raise RuntimeError("на сцене нет Mid-360 с лепестковым узором")
    return lidars


def start(stage=None):
    """Запускает драйвер для каждого Mid-360 с лепестковым узором на сцене."""
    stop()
    stage = stage or omni.usd.get_context().get_stage()
    lidars = _mid360_lidars(stage)
    _active.extend(PatternDriver(stage, lidar) for lidar in lidars)
    print(f"[mid360_driver] запущен для {len(lidars)} лидар(ов)", flush=True)
    return list(_active)


def show_points(stage=None):
    """Рисует облако каждого Mid-360 во вьюпорте (для GUI)."""
    import omni.replicator.core as rep

    stage = stage or omni.usd.get_context().get_stage()
    for lidar in _mid360_lidars(stage):
        render_product = rep.create.render_product(str(lidar.GetPath()), [1, 1])
        writer = rep.writers.get("RtxLidarDebugDrawPointCloudBuffer")
        writer.initialize()
        writer.attach([render_product])
        _drawn.append((writer, render_product))


class RayDrawer:
    """Каждый кадр — отрезки от лидара до части его попаданий в этом кадре."""

    def __init__(self, stage, lidar, max_rays, color):
        import omni.replicator.core as rep
        from isaacsim.util.debug_draw import _debug_draw

        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.stage = stage
        self.path = lidar.GetPath()
        self.max_rays = max_rays
        self.color = color
        self.rng = np.random.default_rng(0)
        self.render_product = rep.create.render_product(
            str(self.path), [1, 1], render_vars=["GenericModelOutput", "RtxSensorMetadata"]
        )
        # Покадровый аннотатор иногда повторяет кадр — для картинки это не важно.
        self.annot = rep.AnnotatorRegistry.get_annotator("IsaacExtractRTXSensorPointCloudNoAccumulator")
        self.annot.attach([self.render_product])
        self.sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
            self._on_update, name=f"mid360 rays {self.path}"
        )

    def _on_update(self, _event):
        if not omni.timeline.get_timeline_interface().is_playing():
            return  # на паузе лучи остаются на экране
        data = self.annot.get_data()
        pts = data.get("data") if isinstance(data, dict) else None
        if pts is None or not len(pts):
            return
        pts = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]  # лучи без возврата приходят нулями
        if len(pts) > self.max_rays:
            pts = pts[self.rng.choice(len(pts), self.max_rays, replace=False)]
        lidar = self.stage.GetPrimAtPath(self.path)
        m = np.array(UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        ends = (np.c_[pts, np.ones(len(pts))] @ m)[:, :3]
        origin = tuple(float(v) for v in m[3, :3])
        n = len(ends)
        self.draw.clear_lines()
        self.draw.draw_lines([origin] * n, [tuple(e) for e in ends.tolist()], [self.color] * n, [1] * n)

    def stop(self):
        self.sub = None
        self.draw.clear_lines()
        self.annot.detach()
        return self.render_product


def show_rays(stage=None, max_rays=600, color=(0.1, 1.0, 0.3, 0.35)):
    """Рисует лучи каждого Mid-360: max_rays случайных лучей за кадр (из ~3300)."""
    stage = stage or omni.usd.get_context().get_stage()
    for lidar in _mid360_lidars(stage):
        _rays.append(RayDrawer(stage, lidar, max_rays, color))


def stop():
    while _active:
        _active.pop().stop()
    # Сначала всё отсоединить, потом уничтожать render product: если уничтожить один
    # раньше, detach у второго на том же лидаре падает в omni.syntheticdata.
    render_products = []
    while _drawn:
        writer, render_product = _drawn.pop()
        writer.detach()
        render_products.append(render_product)
    while _rays:
        render_products.append(_rays.pop().stop())
    for render_product in render_products:
        render_product.destroy()


if __name__ == "__main__":
    start()
    show_points()
    show_rays()
