"""Публикация датчиков стенда в ROS 2: облако Mid-360, кадры обоих fisheye, TF и /clock.

Топики (имена как у настоящих драйверов, чтобы потом подменить симуляцию железом):
  /livox/lidar                 sensor_msgs/PointCloud2, полные сканы 10 Гц, кадр livox_frame
  /insta360/front/image_raw    sensor_msgs/Image rgb8, кадр insta360_front
  /insta360/back/image_raw     sensor_msgs/Image rgb8, кадр insta360_back
  /tf                          world -> livox_frame, insta360_front, insta360_back
  /clock                       время симуляции (RViz запускать с use_sim_time)

Домен ROS берётся из ROS_DOMAIN_ID окружения, в котором запущен Isaac Sim.
Оси кадров камер — как у камеры USD (смотрит по -Z, верх +Y), не оптические ROS.

Из своего скрипта или Script Editor: ros2_sensors.start(); остановить — stop().
Для другой сцены (робот) пути лидара, объективов и кадров TF передаются в start().
"""

import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
import omni.usd
from isaacsim.core.utils.extensions import enable_extension
from pxr import Sdf, Usd

WORLD = "/World"
LIDAR = "/World/Rig/MID360/torso_link/mid360_link/mid360_native_approx"
LENSES = {
    "/World/Rig/Insta360/lens_front": ("insta360/front/image_raw", "insta360_front"),
    "/World/Rig/Insta360/lens_back": ("insta360/back/image_raw", "insta360_back"),
}
GRAPH = "/ROS2SensorsGraph"

_writers = []
_render_products = []


def _set_frame_names(stage, frame_names):
    # Сессионный слой: имена кадров нужны только ROS, в файл сцены не попадут.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for path, name in frame_names.items():
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"[ros2_sensors] нет прима {path} для кадра {name}")
            prim.CreateAttribute("isaac:nameOverride", Sdf.ValueTypeNames.String).Set(name)


def _build_tf_clock_graph(parent, targets):
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": GRAPH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("sim_time", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("context", "isaacsim.ros2.bridge.ROS2Context"),
                ("clock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ("tf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "clock.inputs:execIn"),
                ("tick.outputs:tick", "tf.inputs:execIn"),
                ("sim_time.outputs:simulationTime", "clock.inputs:timeStamp"),
                ("sim_time.outputs:simulationTime", "tf.inputs:timeStamp"),
                ("context.outputs:context", "clock.inputs:context"),
                ("context.outputs:context", "tf.inputs:context"),
            ],
            keys.SET_VALUES: [
                ("tf.inputs:parentPrim", [Sdf.Path(parent)]),
                ("tf.inputs:targetPrims", [Sdf.Path(p) for p in targets]),
            ],
        },
    )


def start(stage=None, image_size=960, image_every_n_frames=6, lidar=LIDAR, lenses=None, frames=None, parent=WORLD):
    """Включает публикацию. Кадры камер: image_size px, каждый N-й кадр рендера (6 -> 10 Гц при 60).

    lidar — путь OmniLidar Mid-360; lenses — {путь камеры: (топик, кадр)}; frames — {путь: имя
    кадра} для TF от parent (по умолчанию лидар и объективы). По умолчанию — стенд."""
    stop()
    enable_extension("isaacsim.ros2.bridge")
    stage = stage or omni.usd.get_context().get_stage()
    lenses = LENSES if lenses is None else lenses
    if frames is None:
        frames = {lidar: "livox_frame", **{p: f for p, (_, f) in lenses.items()}}
    _set_frame_names(stage, {parent: "world", **frames})
    _build_tf_clock_graph(parent, list(frames))

    lidar_rp = rep.create.render_product(lidar, [1, 1], name="Mid360ROS2")
    writer = rep.writers.get("RtxLidarROS2PublishPointCloudBuffer")  # полные сканы, без потерь кадров
    writer.initialize(topicName="livox/lidar", frameId="livox_frame")
    writer.attach([lidar_rp])
    _writers.append(writer)
    _render_products.append(lidar_rp)

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(sd.SensorType.Rgb.name)
    for path, (topic, frame) in lenses.items():
        rp = rep.create.render_product(path, (image_size, image_size), name=f"{frame}_ROS2")
        writer = rep.writers.get(f"{rv}ROS2PublishImage")
        writer.initialize(topicName=topic, frameId=frame, queueSize=1)
        writer.attach([rp])
        gate = omni.syntheticdata.SyntheticData._get_node_path(rv + "IsaacSimulationGate", rp.path)
        og.Controller.attribute(gate + ".inputs:step").set(image_every_n_frames)
        _writers.append(writer)
        _render_products.append(rp)
    print(f"[ros2_sensors] публикую /livox/lidar, {', '.join('/' + t for t, _ in lenses.values())}, /tf, /clock", flush=True)


def stop():
    # Сначала отсоединить всё, потом уничтожать render product (см. mid360_driver.stop).
    while _writers:
        _writers.pop().detach()
    while _render_products:
        _render_products.pop().destroy()
    stage = omni.usd.get_context().get_stage()
    if stage and stage.GetPrimAtPath(GRAPH).IsValid():
        stage.RemovePrim(GRAPH)
