#!/usr/bin/env python3
"""Проверка со стороны ROS 2: идут ли топики стенда, с какой частотой и что внутри.

Запуск (системный python с ROS 2): python3 scripts/check_ros2_topics.py [--seconds 8]
"""

import argparse
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, PointCloud2
from tf2_msgs.msg import TFMessage

IMAGES = ("/insta360/front/image_raw", "/insta360/back/image_raw")


class Checker(Node):
    def __init__(self):
        super().__init__("stend_topic_check")
        self.stats = {}
        self.tf_edges = set()
        self.create_subscription(PointCloud2, "/livox/lidar", self.on_cloud, qos_profile_sensor_data)
        for topic in IMAGES:
            # Кадр 2.7 МБ режется на много UDP-пакетов: при best effort часть кадров теряется.
            self.create_subscription(Image, topic, lambda m, t=topic: self.on_image(t, m), 5)
        self.create_subscription(TFMessage, "/tf", self.on_tf, 10)
        self.create_subscription(Clock, "/clock", lambda m: self.count("/clock", m.clock), 10)

    def count(self, topic, stamp, **info):
        s = self.stats.setdefault(topic, {"n": 0, "first": None, "last": None})
        s["n"] += 1
        t = stamp.sec + stamp.nanosec * 1e-9
        s["first"] = t if s["first"] is None else s["first"]
        s["last"] = t
        s.update(info)

    def on_cloud(self, m):
        n = m.width * m.height
        xyz = None
        names = [f.name for f in m.fields]
        if {"x", "y", "z"} <= set(names):
            off = {f.name: f.offset for f in m.fields}
            raw = np.frombuffer(bytes(m.data), np.uint8).reshape(n, m.point_step)
            xyz = np.stack([raw[:, off[a]:off[a] + 4].copy().view("<f4")[:, 0] for a in "xyz"], 1)
        self.count("/livox/lidar", m.header.stamp, frame=m.header.frame_id, points=n, fields=names,
                   zero_points=int((np.abs(xyz).sum(1) == 0).sum()) if xyz is not None else None)

    def on_image(self, topic, m):
        img = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, -1)
        lit = img.max(axis=2) > 8
        yy, xx = np.nonzero(lit)
        r = np.hypot(xx + 0.5 - m.width / 2, yy + 0.5 - m.height / 2)
        self.count(topic, m.header.stamp, frame=m.header.frame_id, size=f"{m.width}x{m.height}",
                   encoding=m.encoding, circle_radius_px=round(float(np.percentile(r, 99)), 1) if len(r) else 0)

    def on_tf(self, m):
        for t in m.transforms:
            self.tf_edges.add(f"{t.header.frame_id} -> {t.child_frame_id}")
        if m.transforms:
            self.count("/tf", m.transforms[0].header.stamp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=8.0)
    args = parser.parse_args()
    rclpy.init()
    node = Checker()
    end = time.time() + args.seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    for topic in ("/livox/lidar", *IMAGES, "/tf", "/clock"):
        s = node.stats.get(topic)
        if not s:
            print(f"{topic}: НЕТ СООБЩЕНИЙ")
            continue
        span = (s["last"] - s["first"]) or float("nan")
        extra = {k: v for k, v in s.items() if k not in ("n", "first", "last")}
        print(f"{topic}: {s['n']} сообщ., {(s['n'] - 1) / span:.1f} Гц по времени симуляции {extra}")
    print("TF:", sorted(node.tf_edges))
    rclpy.shutdown()


if __name__ == "__main__":
    main()
