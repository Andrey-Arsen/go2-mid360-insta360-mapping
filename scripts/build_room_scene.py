#!/usr/bin/env python3
"""Собирает scenes/simple_room_mid360_fisheye.usda: Simple Room, Mid-360 и 360-камера.

Комната — локальная копия Simple Room (scripts/fetch_simple_room.py), поднятая
так, чтобы пол был на z = 0. Датчики стоят на одной стойке /World/Rig: двигать
и поворачивать её целиком. Камера собрана как Insta360: два fisheye-объектива
спина к спине (модель ftheta, равноугольная, 200 градусов), над лидаром.

Запуск: ~/isaacsim/python.sh scripts/build_room_scene.py [--lidar-height 0.55 ...]
"""

import argparse
import math
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

parser = argparse.ArgumentParser()
parser.add_argument("--rig-xy", type=float, nargs=2, default=(-2.0, -1.8), help="место стойки на полу, м")
parser.add_argument("--yaw", type=float, default=0.0, help="поворот стойки, градусы")
parser.add_argument("--lidar-height", type=float, default=0.55, help="оптический центр Mid-360 над полом, м")
parser.add_argument("--inverted", action="store_true", help="Mid-360 вверх ногами: обзор -52..+7 вместо -7..+52")
parser.add_argument("--lidar-pitch", type=float, default=0.0, help="наклон Mid-360 носом вниз, градусы (камера остаётся вертикальной)")
parser.add_argument("--output", default="simple_room_mid360_fisheye.usda", help="имя файла сцены в scenes/")
parser.add_argument("--camera-above", type=float, default=0.15, help="центр камеры над центром лидара, м")
parser.add_argument("--baseline", type=float, default=0.03, help="расстояние между объективами, м")
parser.add_argument("--resolution", type=int, default=1920, help="кадр каждого объектива, px (квадрат)")
parser.add_argument("--fov", type=float, default=200.0, help="поле зрения объектива, градусы")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "scenes" / args.output
ROOM = "../assets/isaac/Isaac/Environments/Simple_Room/simple_room.usd"
MID360 = "../third_party/Livox_MID360_IsaacSim/assets/petal_scan/standalone/Livox_MID360_Petal_Scan.usd"
FLOOR_MESH = "/World/Environment/Towel_Room01_floor_bottom_218"


def look_rotation(forward):
    """Камера USD смотрит по -Z с верхом по +Y; разворачиваем её вдоль forward с верхом по +Z."""
    f = Gf.Vec3d(*forward).GetNormalized()
    up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(f, up).GetNormalized()
    # Строки — образы локальных осей X, Y, Z (USD умножает вектор-строку на матрицу).
    m = Gf.Matrix3d(right[0], right[1], right[2], up[0], up[1], up[2], -f[0], -f[1], -f[2])
    return m.ExtractRotation().GetQuat()


def add_fisheye(stage, path, position, forward):
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))  # по умолчанию 1 м — отрезало бы всё рядом
    xf = UsdGeom.Xformable(cam)
    xf.AddTranslateOp().Set(Gf.Vec3d(*position))
    xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble).Set(look_rotation(forward))
    prim = cam.GetPrim()
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    half = args.resolution / 2.0
    attrs = {
        "omni:lensdistortion:model": (Sdf.ValueTypeNames.Token, "ftheta"),
        "omni:lensdistortion:ftheta:nominalWidth": (Sdf.ValueTypeNames.Float, float(args.resolution)),
        "omni:lensdistortion:ftheta:nominalHeight": (Sdf.ValueTypeNames.Float, float(args.resolution)),
        "omni:lensdistortion:ftheta:opticalCenter": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(half, half)),
        "omni:lensdistortion:ftheta:maxFov": (Sdf.ValueTypeNames.Float, args.fov),
        # Равноугольная проекция: угол от оси = k1 * радиус в пикселях, край круга — fov/2.
        "omni:lensdistortion:ftheta:k0": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k1": (Sdf.ValueTypeNames.Float, math.radians(args.fov / 2.0) / half),
        "omni:lensdistortion:ftheta:k2": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k3": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k4": (Sdf.ValueTypeNames.Float, 0.0),
    }
    for name, (type_name, value) in attrs.items():
        prim.CreateAttribute(name, type_name).Set(value)
    return cam


def main():
    SCENE.parent.mkdir(exist_ok=True)
    stage = Usd.Stage.CreateNew(str(SCENE))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Как в сценах, сохранённых самим Isaac Sim: без диапазона таймлайн стоит на нуле,
    # и драйвер узора Mid-360 не переключает куски.
    stage.SetTimeCodesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(1000000)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())

    env = UsdGeom.Xform.Define(stage, "/World/Environment")
    env.GetPrim().GetReferences().AddReference(ROOM)
    floor_top = (
        UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        .ComputeWorldBound(stage.GetPrimAtPath(FLOOR_MESH))
        .ComputeAlignedRange()
        .GetMax()[2]
    )
    env.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -floor_top))  # пол комнаты на z = 0

    UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")

    rig = UsdGeom.Xform.Define(stage, "/World/Rig")
    rig.AddTranslateOp().Set(Gf.Vec3d(args.rig_xy[0], args.rig_xy[1], 0.0))
    rig.AddRotateZOp().Set(args.yaw)

    mid = UsdGeom.Xform.Define(stage, "/World/Rig/MID360")
    mid.GetPrim().GetReferences().AddReference(MID360)
    # В ассете корпус сконвертирован из STEP без поворота Y-up -> Z-up: основание
    # лежит в плоскости X-Z, верх смотрит в +Y, а лучи заданы для стоящего датчика
    # (Z вверх). Ставим корпус на основание: +90 градусов вокруг X переводит +Y в +Z.
    cad = UsdGeom.Xformable(stage.OverridePrim(mid.GetPath().AppendPath("torso_link/mid360_link/CAD")))
    cad.AddRotateXOp().Set(90.0)
    lidar = next(p for p in Usd.PrimRange(mid.GetPrim()) if p.GetTypeName() == "OmniLidar")
    # Где оптический центр относительно начала ассета (при нулевой стойке).
    rig_inv = UsdGeom.Xformable(rig).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).GetInverse()
    center = (UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) * rig_inv).ExtractTranslation()
    if args.inverted:
        mid.AddTranslateOp().Set(Gf.Vec3d(-center[0], center[1], args.lidar_height + center[2]))
    else:
        mid.AddTranslateOp().Set(Gf.Vec3d(-center[0], -center[1], args.lidar_height - center[2]))
    if args.lidar_pitch:
        mid.AddRotateYOp().Set(args.lidar_pitch)  # +Y поворачивает ось X (вперёд) вниз; центр лидара — начало ассета
    if args.inverted:
        mid.AddRotateXOp().Set(180.0)

    cam_z = args.lidar_height + args.camera_above
    cam = UsdGeom.Xform.Define(stage, "/World/Rig/Insta360")
    cam.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, cam_z))
    b = args.baseline / 2.0
    add_fisheye(stage, "/World/Rig/Insta360/lens_front", (b, 0.0, 0.0), (1.0, 0.0, 0.0))
    add_fisheye(stage, "/World/Rig/Insta360/lens_back", (-b, 0.0, 0.0), (-1.0, 0.0, 0.0))

    stage.GetRootLayer().customLayerData = {
        "rig": {
            "lidar_height_m": args.lidar_height,
            "lidar_inverted": args.inverted,
            "lidar_pitch_deg": args.lidar_pitch,
            "camera_above_lidar_m": args.camera_above,
            "lens_baseline_m": args.baseline,
            "lens_resolution_px": args.resolution,
            "lens_fov_deg": args.fov,
            "environment_z_offset_m": -floor_top,
        }
    }
    stage.GetRootLayer().Save()

    lidar_world = UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()
    print(f"[build] {SCENE}", flush=True)
    x_axis = UsdGeom.Xformable(lidar).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).TransformDir(Gf.Vec3d(1, 0, 0))
    print(f"[build] пол поднят на {-floor_top:.4f} м; лидар {lidar.GetPath()} в {tuple(round(v, 3) for v in lidar_world)}, "
          f"ось X лидара {tuple(round(v, 3) for v in x_axis)}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
