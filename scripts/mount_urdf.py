#!/usr/bin/env python3
"""Экспорт надстройки из Fusion 360 (fusion2urdf) -> чистый URDF с кадрами датчиков.

Что исправляет в экспорте Fusion:
  * имена: дефисы и пробелы (`mid-360-asm_v1_1`, `Rigid 6`) импортёр Isaac Sim превращает
    в недопустимый путь USD — сетка лидара пропадает; здесь всё латиницей и подчёркиваниями;
  * массы: Fusion считает по плотности материала по умолчанию (корпус выходил 6 кг) —
    берутся из аргументов, инерция — как у коллизионных фигур той же массы;
  * коллизии: вместо сеток на 130 тыс. треугольников — коробки и цилиндр;
  * кадры датчиков, которых в экспорте нет:
      mid360_link = livox_frame — система координат облака Mid-360 по руководству Livox:
        начало O в центре купола на 47.0 мм над дном, +X — от разъёма, Z вверх;
      insta360_link — середина между объективами, X вперёд (к переднему объективу), Z вверх;
      insta360_front / insta360_back — объективы, X — из объектива наружу, Z вверх;
      *_optical — то же в оптической конвенции ROS (Z из объектива, X вправо, Y вниз).
    Положение находится по самим сеткам: разъём лидара — дальняя от центра купола сторона,
    объективы Insta360 — две грани по тонкой оси у верха камеры; передний — тот, что
    смотрит вперёд робота.

Корень — sensor_mount_link: точка крепления — центр подошвы надстройки (низ коробки Orin),
X — вперёд робота (--front: какая ось Fusion смотрит вперёд), Z вверх. Его ставят прямо на
крышку спины робота без поворота. livox_frame остаётся по Livox (+X от разъёма), куда бы
ни смотрел разъём.

Запуск: .venv/bin/python scripts/mount_urdf.py "<папка экспорта Fusion>" [--front -y] [--out robot/go2_sensor_mount]
        [--mass-mount 0.8] [--mass-lidar 0.265] [--mass-camera 0.2]
"""

import argparse
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
LIVOX_ORIGIN_ABOVE_BOTTOM = 0.047  # руководство Mid-360, «Установочные размеры»: O на 47.0 мм над дном


def origin_matrix(el):
    """<origin xyz rpy> -> 4x4 (URDF: rpy — фиксированные оси X, Y, Z)."""
    T = np.eye(4)
    if el is None:
        return T
    T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def fmt_origin(T):
    rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz")
    rpy = [0.0 if abs(a) < 1e-12 else a for a in rpy]
    xyz = [0.0 if abs(v) < 1e-12 else v for v in T[:3, 3]]
    return f'xyz="{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}" rpy="{rpy[0]:.6f} {rpy[1]:.6f} {rpy[2]:.6f}"'


def frame(t, x_axis, z_axis=(0, 0, 1)):
    """Кадр с началом t и осями X, Z (Y = Z x X)."""
    x = np.asarray(x_axis, float) / np.linalg.norm(x_axis)
    z = np.asarray(z_axis, float)
    z = z - x * (x @ z)
    z /= np.linalg.norm(z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, np.cross(z, x), z, t
    return T


def box_inertia(m, sx, sy, sz):
    return m * (sy * sy + sz * sz) / 12, m * (sx * sx + sz * sz) / 12, m * (sx * sx + sy * sy) / 12


def cylinder_inertia(m, r, h):
    return m * (3 * r * r + h * h) / 12, m * (3 * r * r + h * h) / 12, m * r * r / 2


def load_fusion(pkg):
    """Звенья экспорта: имя -> (сетка в координатах корня, путь к STL)."""
    urdf = next((pkg / "urdf").glob("*.urdf"))
    robot = ET.parse(urdf).getroot()
    parent_of = {j.find("child").get("link"): (j.find("parent").get("link"), origin_matrix(j.find("origin")))
                 for j in robot.findall("joint")}

    def to_root(name):
        T = np.eye(4)
        while name in parent_of:
            name, J = parent_of[name]
            T = J @ T
        return T

    links, masses = {}, {}
    for link in robot.findall("link"):
        m = link.find("inertial/mass")
        if m is not None:
            masses[link.get("name")] = float(m.get("value"))
        vis = link.find("visual")
        if vis is None or vis.find("geometry/mesh") is None:
            continue
        mesh_el = vis.find("geometry/mesh")
        rel = re.sub(r"^package://[^/]+/", "", mesh_el.get("filename"))
        scale = float(mesh_el.get("scale", "1 1 1").split()[0])
        mesh = o3d.io.read_triangle_mesh(str(pkg / rel))
        mesh.scale(scale, center=(0, 0, 0))
        mesh.transform(to_root(link.get("name")) @ origin_matrix(vis.find("origin")))
        links[link.get("name")] = (mesh, pkg / rel)
    roots = [n for n in links if n not in parent_of]
    return links, roots[0], masses


def lidar_frame(mesh):
    """Кадр облака Mid-360 по сетке корпуса: O над центром купола, +X от разъёма."""
    P = np.asarray(mesh.sample_points_uniformly(300_000).points)
    bottom, top = P[:, 2].min(), P[:, 2].max()
    dome = P[P[:, 2] > top - 0.010, :2].mean(0)
    lo, hi = P[:, :2].min(0), P[:, :2].max(0)
    # Разъём выступает с одной стороны: та сторона дальше от центра купола.
    reach = np.array([[dome[0] - lo[0], hi[0] - dome[0]], [dome[1] - lo[1], hi[1] - dome[1]]])
    axis = int(np.argmax(np.abs(reach[:, 0] - reach[:, 1])))
    x = np.zeros(3)
    x[axis] = 1.0 if reach[axis, 0] > reach[axis, 1] else -1.0  # разъём на минусе -> +X в плюс
    return frame(np.r_[dome, bottom + LIVOX_ORIGIN_ABOVE_BOTTOM], x), (bottom, top)


def camera_frames(mesh, forward):
    """Объективы Insta360: две грани по тонкой горизонтальной оси, центр — по выпуклостям у верха."""
    P = np.asarray(mesh.sample_points_uniformly(300_000).points)
    ext = np.ptp(P[:, :2], 0)
    axis = int(np.argmin(ext))
    lenses = []
    for sign in (-1, 1):
        edge = P[:, axis].min() if sign < 0 else P[:, axis].max()
        face = P[np.abs(P[:, axis] - edge) < 0.004]
        lenses.append((sign, face.mean(0)))
    d = np.zeros(3)
    d[axis] = 1.0
    front_sign = 1 if forward @ d > 0 else -1
    (s_a, c_a), (s_b, c_b) = lenses
    front, back = (c_b, c_a) if front_sign > 0 else (c_a, c_b)
    out = front_sign * d
    mid = (front + back) / 2
    return frame(mid, out), frame(front, out), frame(back, -out), float(np.linalg.norm(front - back))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fusion_pkg", help="папка экспорта Fusion (с urdf/ и meshes/)")
    parser.add_argument("--out", default=str(ROOT / "robot" / "go2_sensor_mount"))
    parser.add_argument("--name", default="go2_sensor_mount")
    parser.add_argument("--mass-mount", type=float, default=0.8, help="кронштейн + Jetson Orin Nano, кг (взвесить!)")
    parser.add_argument("--mass-lidar", type=float, default=0.265, help="Mid-360 по паспорту, кг")
    parser.add_argument("--mass-camera", type=float, default=0.2, help="Insta360 X5, кг")
    parser.add_argument("--front", default="-y", choices=["+x", "-x", "+y", "-y"],
                        help="ось Fusion, которая смотрит вперёд робота (сейчас -y: разъём лидара и антенны Orin вперёд)")
    args = parser.parse_args()

    src = Path(args.fusion_pkg)
    if not (src / "urdf").exists():
        cand = [p for p in src.iterdir() if (p / "urdf").exists()]
        if not cand:
            raise SystemExit(f"в {src} нет папки urdf/ — укажите папку экспорта Fusion")
        src = cand[0]
    links, root, fusion_mass = load_fusion(src)
    lidar = next((n for n in links if "mid" in n.lower() and "360" in n), None)
    camera = next((n for n in links if "insta" in n.lower()), None)
    if lidar is None or camera is None or root in (lidar, camera):
        raise SystemExit(f"не нашёл звенья: корень {root}, лидар {lidar}, камера {camera} (ищу 'mid…360' и 'insta')")

    T_lidar, (l_bottom, l_top) = lidar_frame(links[lidar][0])
    front = np.zeros(3)
    front["xyz".index(args.front[1])] = 1.0 if args.front[0] == "+" else -1.0
    T_cam, T_front, T_back, baseline = camera_frames(links[camera][0], front)
    # Точка крепления: центр подошвы (нижний сантиметр корпуса надстройки), X вперёд робота.
    B = np.asarray(links[root][0].sample_points_uniformly(300_000).points)
    sole = B[B[:, 2] < B[:, 2].min() + 0.01]
    T_root = frame(np.r_[(sole[:, :2].min(0) + sole[:, :2].max(0)) / 2, B[:, 2].min()], front)
    to_root = np.linalg.inv(T_root)

    out = Path(args.out)
    (out / "meshes").mkdir(parents=True, exist_ok=True)
    (out / "urdf").mkdir(parents=True, exist_ok=True)
    meshes = {root: "mount_base.stl", lidar: "mid360.stl", camera: "insta360_x5.stl"}
    for name, fname in meshes.items():
        shutil.copyfile(links[name][1], out / "meshes" / fname)
    pkg = f"package://{args.name}/meshes"

    # Сетки остаются в координатах Fusion — в кадре звена их сдвигает обратное преобразование.
    base_mesh = links[root][0]
    bb = base_mesh.get_axis_aligned_bounding_box()
    b_lo, b_hi = np.asarray(bb.min_bound), np.asarray(bb.max_bound)
    b_c = (to_root @ np.r_[(b_hi + b_lo) / 2, 1])[:3]
    b_size = np.abs(to_root[:3, :3]) @ (b_hi - b_lo)
    cam_P = np.asarray(links[camera][0].vertices)
    cam_c_world = (cam_P.min(0) + cam_P.max(0)) / 2
    cam_size = np.ptp(cam_P, 0)
    lid_c_local = np.linalg.inv(T_lidar) @ np.r_[T_lidar[:2, 3], (l_bottom + l_top) / 2, 1]
    cam_c_local = np.linalg.inv(T_cam) @ np.r_[cam_c_world, 1]
    cam_size_local = np.abs(T_cam[:3, :3].T) @ cam_size

    ib = box_inertia(args.mass_mount, *b_size)
    il = cylinder_inertia(args.mass_lidar, 0.0325, 0.060)
    ic = box_inertia(args.mass_camera, *cam_size_local)
    I = np.eye(4)

    def inertial(m, c, i):
        return (f'    <inertial>\n      <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
                f'      <mass value="{m:.4f}"/>\n'
                f'      <inertia ixx="{i[0]:.3e}" iyy="{i[1]:.3e}" izz="{i[2]:.3e}" ixy="0" ixz="0" iyz="0"/>\n'
                f'    </inertial>\n')

    def visual(fname, T_mesh, material):
        return (f'    <visual>\n      <origin {fmt_origin(T_mesh)}/>\n'
                f'      <geometry><mesh filename="{pkg}/{fname}" scale="0.001 0.001 0.001"/></geometry>\n'
                f'      <material name="{material}"/>\n    </visual>\n')

    def fixed(name, parent, child, T):
        return (f'  <joint name="{name}" type="fixed">\n    <origin {fmt_origin(T)}/>\n'
                f'    <parent link="{parent}"/>\n    <child link="{child}"/>\n  </joint>\n')

    # Подсказка ROS: R оптического кадра относительно кадра объектива (X вперёд, Z вверх).
    R_opt = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)  # столбцы — оси оптического в кадре объектива
    T_opt = np.eye(4)
    T_opt[:3, :3] = R_opt
    urdf = f'''<?xml version="1.0"?>
<!-- Сделано scripts/mount_urdf.py из экспорта Fusion 360 ({src.name}). Не править руками —
     переэкспортировать из Fusion и запустить скрипт заново. Массы — из аргументов скрипта. -->
<robot name="{args.name}">
  <material name="mount_gray"><color rgba="0.82 0.82 0.84 1"/></material>
  <material name="lidar_dark"><color rgba="0.18 0.18 0.2 1"/></material>
  <material name="camera_black"><color rgba="0.08 0.08 0.09 1"/></material>

  <link name="sensor_mount_link">
{inertial(args.mass_mount, b_c, ib)}{visual(meshes[root], to_root, "mount_gray")}    <collision>
      <origin xyz="{b_c[0]:.6f} {b_c[1]:.6f} {b_c[2]:.6f}" rpy="0 0 0"/>
      <geometry><box size="{b_size[0]:.4f} {b_size[1]:.4f} {b_size[2]:.4f}"/></geometry>
    </collision>
  </link>

  <!-- Mid-360: кадр звена = система координат облака (руководство Livox: O на 47 мм над дном,
       +X от разъёма, Z вверх). livox_frame — то же имя, что у livox_ros_driver2. -->
  <link name="mid360_link">
{inertial(args.mass_lidar, lid_c_local, il)}{visual(meshes[lidar], np.linalg.inv(T_lidar), "lidar_dark")}    <collision>
      <origin xyz="{lid_c_local[0]:.6f} {lid_c_local[1]:.6f} {lid_c_local[2]:.6f}" rpy="0 0 0"/>
      <geometry><cylinder radius="0.0325" length="0.060"/></geometry>
    </collision>
  </link>
{fixed("mid360_joint", "sensor_mount_link", "mid360_link", to_root @ T_lidar)}  <link name="livox_frame"/>
{fixed("livox_frame_joint", "mid360_link", "livox_frame", I)}
  <!-- Insta360 X5: кадр звена — середина между объективами, X к переднему объективу, Z вверх. -->
  <link name="insta360_link">
{inertial(args.mass_camera, cam_c_local, ic)}{visual(meshes[camera], np.linalg.inv(T_cam), "camera_black")}    <collision>
      <origin xyz="{cam_c_local[0]:.6f} {cam_c_local[1]:.6f} {cam_c_local[2]:.6f}" rpy="0 0 0"/>
      <geometry><box size="{cam_size_local[0]:.4f} {cam_size_local[1]:.4f} {cam_size_local[2]:.4f}"/></geometry>
    </collision>
  </link>
{fixed("insta360_joint", "sensor_mount_link", "insta360_link", to_root @ T_cam)}  <link name="insta360_front"/>
{fixed("insta360_front_joint", "insta360_link", "insta360_front", np.linalg.inv(T_cam) @ T_front)}  <link name="insta360_back"/>
{fixed("insta360_back_joint", "insta360_link", "insta360_back", np.linalg.inv(T_cam) @ T_back)}  <link name="insta360_front_optical"/>
{fixed("insta360_front_optical_joint", "insta360_front", "insta360_front_optical", T_opt)}  <link name="insta360_back_optical"/>
{fixed("insta360_back_optical_joint", "insta360_back", "insta360_back_optical", T_opt)}</robot>
'''
    (out / "urdf" / f"{args.name}.urdf").write_text(urdf, encoding="utf-8")
    (out / "package.xml").write_text(f'''<?xml version="1.0"?>
<package format="3">
  <name>{args.name}</name>
  <version>0.1.0</version>
  <description>Надстройка Go2: Jetson Orin Nano, Livox Mid-360, Insta360 X5 (из Fusion 360, scripts/mount_urdf.py)</description>
  <maintainer email="user@example.com">user</maintainer>
  <license>MIT</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <export><build_type>ament_cmake</build_type></export>
</package>
''', encoding="utf-8")
    (out / "CMakeLists.txt").write_text(f'''cmake_minimum_required(VERSION 3.8)
project({args.name})
find_package(ament_cmake REQUIRED)
install(DIRECTORY urdf meshes DESTINATION share/${{PROJECT_NAME}})
ament_package()
''', encoding="utf-8")

    sole_len = np.ptp(sole[:, :2] @ front[:2])
    L, F, K = (to_root @ T for T in (T_lidar, T_front, T_back))
    print(f"[mount] {out / 'urdf' / (args.name + '.urdf')}")
    print(f"[mount] вперёд робота — {args.front} Fusion; X лидара (от разъёма) смотрит {'вперёд' if L[0, 0] > 0 else 'назад'} робота; "
          f"подошва {sole_len * 1000:.0f} мм вдоль робота")
    print(f"[mount] в кадре крепления (мм, X вперёд): лидар O {np.round(L[:3, 3] * 1000, 1)}, "
          f"объективы {np.round(F[:3, 3] * 1000, 1)} / {np.round(K[:3, 3] * 1000, 1)}, между ними {baseline * 1000:.0f} мм")
    print(f"[mount] массы: надстройка {args.mass_mount} кг, Mid-360 {args.mass_lidar} кг, Insta360 {args.mass_camera} кг "
          f"(в экспорте Fusion: {fusion_mass.get(root, 0):.2f} / {fusion_mass.get(lidar, 0):.2f} / {fusion_mass.get(camera, 0):.2f} кг)")


if __name__ == "__main__":
    main()
