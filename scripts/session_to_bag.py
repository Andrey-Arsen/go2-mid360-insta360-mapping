#!/usr/bin/env python3
"""Собирает из сессии (record_session.py) rosbag2 в том виде, в каком его пишет ATLAS.

Топики:
  /livox/lidar            PointCloud2 как у livox_ros_driver2 (x,y,z,intensity,tag,line,timestamp[нс])
  /livox/imu              Imu 200 Гц, livox_frame, м/с² и рад/с (REP-145; живой драйвер Livox даёт g!)
  /camera/shutter_time    Float64 — момент съёмки, секунды (как shutter_event_publisher ATLAS)
  /tf_static              world -> odom (поза в начале), base_link -> livox_frame
  /ground_truth/odom      Odometry эталона в world, 50 Гц
  /ground_truth/path      Path эталона (10 поз/с), растёт раз в 0.5 с
  /insta360/{front,back}/image_raw   кадры в моменты съёмки (с --images)

Время симуляции сдвинуто на BASE_S: нулевая метка в TF означает «последнее».

Запуск (системный python с ROS 2): python3 scripts/session_to_bag.py walk01 [--images]
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import rosbag2_py
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.serialization import serialize_message
from sensor_msgs.msg import Image, Imu, PointCloud2, PointField
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage

ROOT = Path(__file__).resolve().parents[1]
BASE_S = 1_700_000_000
INTENSITY_SCALE = 1e4  # сырое RTX ~0.003..0.013 -> ~30..130, похоже на отражательность Livox

LIVOX_FIELDS = [
    ("x", 0, PointField.FLOAT32), ("y", 4, PointField.FLOAT32), ("z", 8, PointField.FLOAT32),
    ("intensity", 12, PointField.FLOAT32), ("tag", 16, PointField.UINT8), ("line", 17, PointField.UINT8),
    ("timestamp", 18, PointField.FLOAT64),
]
LIVOX_DTYPE = np.dtype({"names": ["x", "y", "z", "intensity", "tag", "line", "timestamp"],
                        "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f8"],
                        "offsets": [0, 4, 8, 12, 16, 17, 18], "itemsize": 26})


def stamp(t_s):
    ns = int(round((BASE_S + t_s) * 1e9))
    return Time(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000), ns


def transform(parent, child, t_s, pos, quat_xyzw):
    msg = TransformStamped()
    msg.header.stamp, _ = stamp(t_s)
    msg.header.frame_id, msg.child_frame_id = parent, child
    msg.transform.translation.x, msg.transform.translation.y, msg.transform.translation.z = map(float, pos)
    q = msg.transform.rotation
    q.x, q.y, q.z, q.w = map(float, quat_xyzw)
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--images", action="store_true", help="положить в bag кадры объективов (960 px)")
    args = parser.parse_args()
    sdir = ROOT / "data" / "sessions" / args.session
    bag_dir = sdir / "bag"
    if bag_dir.exists():
        shutil.rmtree(bag_dir)
    meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
    lidar, imu, gt = np.load(sdir / "lidar.npz"), np.load(sdir / "imu.npz"), np.load(sdir / "gt.npz")

    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    topics = {
        "/livox/lidar": "sensor_msgs/msg/PointCloud2", "/livox/imu": "sensor_msgs/msg/Imu",
        "/camera/shutter_time": "std_msgs/msg/Float64", "/tf_static": "tf2_msgs/msg/TFMessage",
        "/ground_truth/odom": "nav_msgs/msg/Odometry", "/ground_truth/path": "nav_msgs/msg/Path",
    }
    if args.images:
        topics.update({f"/insta360/{k}/image_raw": "sensor_msgs/msg/Image" for k in ("front", "back")})
    static_qos = ("- history: 1\n  depth: 1\n  reliability: 1\n  durability: 1\n  deadline: {sec: 0, nsec: 0}\n"
                  "  lifespan: {sec: 0, nsec: 0}\n  liveliness: 1\n  liveliness_lease_duration: {sec: 0, nsec: 0}\n"
                  "  avoid_ros_namespace_conventions: false\n")
    for name, type_name in topics.items():
        qos = static_qos if name == "/tf_static" else ""
        writer.create_topic(rosbag2_py.TopicMetadata(name=name, type=type_name, serialization_format="cdr",
                                                     offered_qos_profiles=qos))
    out = []  # (время нс, топик, сообщение)

    # /tf_static: odom совпадает с позой в начале (RKO-LIO ведёт одометрию от неё, z по гравитации).
    t0 = float(gt["t"][0])
    p0, q0 = gt["pos"][0], gt["quat_xyzw"][0]
    yaw0 = np.arctan2(2 * (q0[3] * q0[2] + q0[0] * q0[1]), 1 - 2 * (q0[1] ** 2 + q0[2] ** 2))
    tf_static = TFMessage(transforms=[
        transform("world", "odom", t0, p0, (0.0, 0.0, np.sin(yaw0 / 2), np.cos(yaw0 / 2))),
        transform("base_link", "livox_frame", t0, (0, 0, 0), (0, 0, 0, 1)),
    ])
    out.append((stamp(t0)[1], "/tf_static", tf_static))

    # /livox/lidar
    pts, tt, inten, starts = lidar["points"], lidar["t"], lidar["intensity"], lidar["scan_start"]
    ends = np.r_[starts[1:], len(pts)]
    for a, b in zip(starts.tolist(), ends.tolist()):
        order = np.argsort(tt[a:b], kind="stable")
        rec = np.zeros(b - a, dtype=LIVOX_DTYPE)
        p = pts[a:b][order]
        rec["x"], rec["y"], rec["z"] = p[:, 0], p[:, 1], p[:, 2]
        rec["intensity"] = np.clip(inten[a:b][order] * INTENSITY_SCALE, 0, 255)
        rec["timestamp"] = (BASE_S + tt[a:b][order]) * 1e9
        msg = PointCloud2(height=1, width=b - a, is_bigendian=False, point_step=26, row_step=26 * (b - a), is_dense=True,
                          fields=[PointField(name=n, offset=o, datatype=d, count=1) for n, o, d in LIVOX_FIELDS])
        msg.header.stamp, ns = stamp(float(tt[a:b][order][0]))
        msg.header.frame_id = meta["lidar_frame"]
        msg.data = rec.tobytes()
        out.append((ns, "/livox/lidar", msg))

    # /livox/imu
    for t, g, a in zip(imu["t"], imu["gyro"], imu["accel"]):
        msg = Imu()
        msg.header.stamp, ns = stamp(float(t))
        msg.header.frame_id = meta["imu_frame"]
        msg.orientation_covariance[0] = -1.0  # ориентации нет
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = map(float, g)
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = map(float, a)
        out.append((ns, "/livox/imu", msg))

    # эталон
    path = PathMsg()
    path.header.frame_id = "world"
    next_path = t0
    for k in range(0, len(gt["t"]), 4):  # 200 -> 50 Гц
        t, p, q = float(gt["t"][k]), gt["pos"][k], gt["quat_xyzw"][k]
        msg = Odometry()
        msg.header.stamp, ns = stamp(t)
        msg.header.frame_id, msg.child_frame_id = "world", "gt_livox"
        msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = map(float, p)
        o = msg.pose.pose.orientation
        o.x, o.y, o.z, o.w = map(float, q)
        out.append((ns, "/ground_truth/odom", msg))
        if k % 20 == 0:  # в Path 10 поз/с: длинная линия тяжела для RViz
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = msg.pose.pose
            path.poses.append(ps)
        if t >= next_path:
            path.header.stamp = msg.header.stamp
            out.append((ns, "/ground_truth/path", PathMsg(header=path.header, poses=list(path.poses))))
            next_path += 0.5

    # съёмка
    for idx, t in enumerate(meta["shutter_times_s"]):
        _, ns = stamp(t)
        out.append((ns, "/camera/shutter_time", Float64(data=BASE_S + t)))
        if args.images:
            from PIL import Image as PILImage

            for lens in ("front", "back"):
                img = PILImage.open(sdir / "images" / f"{idx:03d}_{lens}.png").convert("RGB").resize((960, 960))
                msg = Image(height=960, width=960, encoding="rgb8", is_bigendian=0, step=960 * 3)
                msg.header.stamp, _ = stamp(t)
                msg.header.frame_id = f"insta360_{lens}"
                msg.data = np.asarray(img).tobytes()
                out.append((ns, f"/insta360/{lens}/image_raw", msg))

    out.sort(key=lambda x: x[0])
    for ns, topic, msg in out:
        writer.write(topic, serialize_message(msg), ns)
    del writer
    counts = {}
    for _, topic, _ in out:
        counts[topic] = counts.get(topic, 0) + 1
    print(f"[bag] {bag_dir}: " + ", ".join(f"{k} {v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
