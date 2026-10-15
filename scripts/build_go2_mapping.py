#!/usr/bin/env python3
"""Go2 из IsaacLegs + надстройка (Jetson Orin, Mid-360, Insta360) с нашими датчиками.

Пишет robot/go2_mapping/go2_mapping.usda — слой поверх сцены IsaacLegs (go2.usd подключён
подслоем, поэтому его графы ROS 2, пути и политика работают как есть) — и рядом
sensor_mount_geom.usdc с сетками надстройки. Что добавляется к /World/go2/base:

  sensor_mount            кадр крепления из URDF (robot/go2_sensor_mount, scripts/mount_urdf.py):
                          центр подошвы коробки Orin, X вперёд робота; подошва лежит на ровной
                          площадке спины, задний край — на задней кромке площадки (--rear-x, --shift)
    visuals               сетки из STL (видимые)
    colliders             коробки/цилиндр из URDF (purpose guide — не рисуются, но сталкиваются)
    frames/<звено>        кадры датчиков из URDF: livox_frame, insta360_front/back (+ _optical)
      livox_frame/Mid360  ассет Mid-360 (Petal Scan, наш драйвер) — точка лидара в livox_frame,
                          его собственный корпус скрыт (виден корпус из Fusion)
      insta360_*/fisheye  ftheta-камера 200°, равноугольная, как на стенде (--resolution)
  масса надстройки добавляется к массе корпуса (центр масс и инерция пересчитаны).

Спина меряется по поверхности визуальных сеток Go2 (base, base_white, base_black).
Нужен интернет: модель Go2 в сцене IsaacLegs подгружается с сервера NVIDIA.

Запуск: ~/isaacsim/python.sh scripts/build_go2_mapping.py [--shift 0.0] [--resolution 1920]
"""

import argparse
import math
import os
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--go2", default=str(ROOT / "third_party" / "IsaacLegs" / "isaacsim_envs" / "gui" / "go2.usd"))
parser.add_argument("--mount", default=str(ROOT / "robot" / "go2_sensor_mount" / "urdf" / "go2_sensor_mount.urdf"))
parser.add_argument("--out", default=str(ROOT / "robot" / "go2_mapping" / "go2_mapping.usda"))
parser.add_argument("--rear-x", type=float, default=None,
                    help="задний край подошвы надстройки вдоль робота от центра корпуса, м (+ вперёд); "
                         "по умолчанию — задняя кромка ровной площадки на спине")
parser.add_argument("--shift", type=float, default=0.0, help="сдвинуть надстройку от этого места, м (+ вперёд)")
parser.add_argument("--resolution", type=int, default=1920, help="сторона кадра fisheye, пикселей")
parser.add_argument("--fov", type=float, default=200.0, help="поле зрения объектива, град")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

MID360_ASSET = ROOT / "third_party" / "Livox_MID360_IsaacSim" / "assets" / "petal_scan" / "standalone" / "Livox_MID360_Petal_Scan.usd"
BASE = "/World/go2/base"


def read_stl(path):
    """Двоичный STL -> (вершины без повторов, треугольники)."""
    raw = Path(path).read_bytes()
    n = int(np.frombuffer(raw, np.uint32, 1, 80)[0])
    rec = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    tri = rec["v"].reshape(-1, 3).astype(np.float64)
    verts, idx = np.unique(np.round(tri, 6), axis=0, return_inverse=True)
    return verts, idx.reshape(-1, 3)


def origin(el):
    T = np.eye(4)
    if el is None:
        return T
    T[:3, 3] = [float(v) for v in el.get("xyz", "0 0 0").split()]
    T[:3, :3] = Rotation.from_euler("xyz", [float(v) for v in el.get("rpy", "0 0 0").split()]).as_matrix()
    return T


def parse_mount(urdf):
    robot = ET.parse(urdf).getroot()
    pkg = Path(urdf).resolve().parents[1]
    colors = {m.get("name"): [float(v) for v in m.find("color").get("rgba").split()[:3]]
              for m in robot.findall("material") if m.find("color") is not None}
    parent = {j.find("child").get("link"): (j.find("parent").get("link"), origin(j.find("origin")))
              for j in robot.findall("joint")}

    def chain(name):
        T = np.eye(4)
        while name in parent:
            name, J = parent[name]
            T = J @ T
        return T

    links = {}
    for link in robot.findall("link"):
        name = link.get("name")
        info = {"T": chain(name), "visual": None, "collision": [], "mass": 0.0, "com": np.zeros(3)}
        vis = link.find("visual")
        if vis is not None and vis.find("geometry/mesh") is not None:
            mesh = vis.find("geometry/mesh")
            fname = pkg / "meshes" / Path(mesh.get("filename")).name
            mat = vis.find("material")
            info["visual"] = (fname, float(mesh.get("scale", "1").split()[0]), origin(vis.find("origin")),
                              colors.get(mat.get("name") if mat is not None else "", [0.7, 0.7, 0.7]))
        for col in link.findall("collision"):
            g = col.find("geometry")[0]
            info["collision"].append((g.tag, dict(g.attrib), origin(col.find("origin"))))
        inert = link.find("inertial")
        if inert is not None:
            info["mass"] = float(inert.find("mass").get("value"))
            info["com"] = origin(inert.find("origin"))[:3, 3]
        links[name] = info
    root = next(n for n in links if n not in parent)
    return links, root


def set_matrix(xf, T):
    xf.ClearXformOpOrder()
    xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(T.T.tolist()))


def world(prim):
    return np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T


def body_surface(stage, roots, to_base, per_mm2=1.0, seed=0):
    """Точки по поверхности (а не вершины) визуальных сеток под roots, в кадре base."""
    rng = np.random.default_rng(seed)
    pts = []
    for r in roots:
        prim = stage.GetPrimAtPath(r)
        if not prim.IsValid():
            continue
        for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            if p.GetTypeName() != "Mesh" or "collision" in str(p.GetPath()).lower():
                continue
            if UsdGeom.Imageable(p).ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            V = np.array(p.GetAttribute("points").Get() or [], float)
            cnt = np.array(p.GetAttribute("faceVertexCounts").Get() or [], int)
            idx = np.array(p.GetAttribute("faceVertexIndices").Get() or [], int)
            if not len(V) or not len(cnt):
                continue
            first = np.r_[0, np.cumsum(cnt)[:-1]]
            T = np.vstack([np.c_[idx[first[cnt > k + 1]], idx[first[cnt > k + 1] + k], idx[first[cnt > k + 1] + k + 1]]
                           for k in range(1, cnt.max() - 1)])
            M = to_base @ world(p)
            W = V @ M[:3, :3].T + M[:3, 3]
            a, b, c = W[T[:, 0]], W[T[:, 1]], W[T[:, 2]]
            area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
            n = max(1, int(area.sum() * 1e6 * per_mm2))
            k = rng.choice(len(T), n, p=area / area.sum())
            u, v = rng.random(n), rng.random(n)
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            pts.append(a[k] + u[:, None] * (b[k] - a[k]) + v[:, None] * (c[k] - a[k]))
    return np.vstack(pts)


def add_fisheye(stage, path):
    """ftheta-камера, смотрит по +X родительского кадра, верх +Z (камера USD смотрит по -Z, верх +Y)."""
    cam = UsdGeom.Camera.Define(stage, path)
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
    R = np.column_stack([[0, -1, 0], [0, 0, 1], [-1, 0, 0]]).astype(float)  # оси камеры в кадре объектива
    T = np.eye(4)
    T[:3, :3] = R
    set_matrix(UsdGeom.Xformable(cam), T)
    prim = cam.GetPrim()
    prim.ApplyAPI("OmniLensDistortionFthetaAPI")
    half = args.resolution / 2.0
    for name, (tn, value) in {
        "omni:lensdistortion:model": (Sdf.ValueTypeNames.Token, "ftheta"),
        "omni:lensdistortion:ftheta:nominalWidth": (Sdf.ValueTypeNames.Float, float(args.resolution)),
        "omni:lensdistortion:ftheta:nominalHeight": (Sdf.ValueTypeNames.Float, float(args.resolution)),
        "omni:lensdistortion:ftheta:opticalCenter": (Sdf.ValueTypeNames.Float2, Gf.Vec2f(half, half)),
        "omni:lensdistortion:ftheta:maxFov": (Sdf.ValueTypeNames.Float, args.fov),
        "omni:lensdistortion:ftheta:k0": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k1": (Sdf.ValueTypeNames.Float, math.radians(args.fov / 2.0) / half),
        "omni:lensdistortion:ftheta:k2": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k3": (Sdf.ValueTypeNames.Float, 0.0),
        "omni:lensdistortion:ftheta:k4": (Sdf.ValueTypeNames.Float, 0.0),
    }.items():
        prim.CreateAttribute(name, tn).Set(value)
    return cam


def main():
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    links, root = parse_mount(args.mount)
    lidar_link = next(n for n in links if n == "livox_frame" or "mid360" in n)

    # Сетки надстройки — в кадре надстройки (корень URDF), отдельным двоичным слоем.
    geom_path = out.with_name("sensor_mount_geom.usdc")
    geo_stage = Usd.Stage.CreateNew(str(geom_path))
    UsdGeom.SetStageUpAxis(geo_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(geo_stage, 1.0)
    groot = UsdGeom.Xform.Define(geo_stage, "/SensorMount")
    geo_stage.SetDefaultPrim(groot.GetPrim())
    base_pts = None
    for name, info in links.items():
        if info["visual"] is None:
            continue
        fname, scale, T_vis, color = info["visual"]
        V, F = read_stl(fname)
        T = info["T"] @ T_vis
        V = (V * scale) @ T[:3, :3].T + T[:3, 3]
        if name == root:
            base_pts = V
        m = UsdGeom.Mesh.Define(geo_stage, f"/SensorMount/{name}")
        m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(V.astype(np.float32)))
        m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(F), 3, np.int32)))
        m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(F.astype(np.int32).ravel()))
        m.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        m.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        m.CreateExtentAttr([Gf.Vec3f(*V.min(0).tolist()), Gf.Vec3f(*V.max(0).tolist())])
    geo_stage.GetRootLayer().Save()

    # Подошва надстройки в её кадре (кадр крепления: центр подошвы, X вперёд робота).
    bottom = base_pts[:, 2].min()
    foot = base_pts[base_pts[:, 2] < bottom + 0.01]
    f_lo, f_hi = foot[:, :2].min(0), foot[:, :2].max(0)

    ctx = omni.usd.get_context()
    if not ctx.open_stage(args.go2):
        raise RuntimeError(f"не открылась {args.go2}")
    for _ in range(20):
        app.update()
    stage = ctx.get_stage()
    base = stage.GetPrimAtPath(BASE)
    if not base.IsValid():
        raise RuntimeError(f"в {args.go2} нет {BASE}")
    if base.IsInstance():
        raise RuntimeError(f"{BASE} — экземпляр (instanceable), к нему нельзя добавить детей")
    M_base = world(base)

    # Спина: точки по поверхности визуальных сеток корпуса (у крышки — большие плоские грани
    # с вершинами только по краям, вершин мало). Корпус Go2 — base и его обшивка base_white/black.
    P = body_surface(stage, [BASE, "/World/go2/base_white", "/World/go2/base_black"], np.linalg.inv(M_base))
    strip = P[np.abs(P[:, 1]) < 0.03]
    z_flat = float(strip[np.abs(strip[:, 0]) < 0.01, 2].max())  # высота спины над центром корпуса
    flat = strip[np.abs(strip[:, 2] - z_flat) < 0.003]
    rear_edge = float(flat[:, 0].min()) if args.rear_x is None else args.rear_x
    x_c = rear_edge - f_lo[0] + args.shift
    under = (P[:, 0] > x_c + f_lo[0]) & (P[:, 0] < x_c + f_hi[0]) & (P[:, 1] > f_lo[1]) & (P[:, 1] < f_hi[1])
    if not under.any():
        raise RuntimeError("под подошвой надстройки нет корпуса Go2 — проверьте --rear-x/--shift")
    z_top = float(P[under, 2].max())
    touch = P[under & (P[:, 2] > z_top - 0.001)]  # на что опирается подошва
    contact = (f"опирается на x {touch[:, 0].min() * 1000:+.0f}..{touch[:, 0].max() * 1000:+.0f} мм, "
               f"|y| {np.abs(touch[:, 1]).min() * 1000:.0f}..{np.abs(touch[:, 1]).max() * 1000:.0f} мм; "
               f"середина спины {z_flat * 1000:.1f} мм")
    T_mount = np.eye(4)
    T_mount[:3, 3] = [x_c, 0.0, z_top]

    # Новый слой поверх сцены IsaacLegs.
    layer = Sdf.Layer.CreateNew(str(out))
    layer.subLayerPaths.append(os.path.relpath(args.go2, out.parent))
    # Метаданные сцены (Z вверх, метры, время) USD берёт только из корневого слоя — у подслоя
    # они не наследуются; без них сцена стала бы Y-вверх в сантиметрах (и физика тоже).
    src = Sdf.Layer.FindOrOpen(args.go2)
    for key in ("upAxis", "metersPerUnit", "kilogramsPerUnit", "timeCodesPerSecond", "framesPerSecond",
                "startTimeCode", "endTimeCode", "defaultPrim"):
        if src.pseudoRoot.HasInfo(key):
            layer.pseudoRoot.SetInfo(key, src.pseudoRoot.GetInfo(key))
    if not layer.pseudoRoot.HasInfo("upAxis"):
        layer.pseudoRoot.SetInfo("upAxis", "Z")
    if not layer.pseudoRoot.HasInfo("metersPerUnit"):
        layer.pseudoRoot.SetInfo("metersPerUnit", 1.0)
    scene = Usd.Stage.Open(layer)
    scene.SetEditTarget(scene.GetRootLayer())
    for _ in range(5):
        app.update()
    mount = UsdGeom.Xform.Define(scene, f"{BASE}/sensor_mount")
    set_matrix(mount, T_mount)
    vis = UsdGeom.Xform.Define(scene, f"{BASE}/sensor_mount/visuals")
    vis.GetPrim().GetReferences().AddReference(f"./{geom_path.name}")

    # Коллизии и масса.
    parts = []
    for name, info in links.items():
        for k, (tag, attr, T_col) in enumerate(info["collision"]):
            T = info["T"] @ T_col
            path = f"{BASE}/sensor_mount/colliders/{name}_{k}"
            if tag == "box":
                size = [float(v) for v in attr["size"].split()]
                g = UsdGeom.Cube.Define(scene, path)
                g.CreateSizeAttr(1.0)
                S = np.diag(size + [1.0])
                set_matrix(g, T @ S)
            elif tag == "cylinder":
                g = UsdGeom.Cylinder.Define(scene, path)
                g.CreateRadiusAttr(float(attr["radius"]))
                g.CreateHeightAttr(float(attr["length"]))
                g.CreateAxisAttr(UsdGeom.Tokens.z)
                set_matrix(g, T)
            else:
                continue
            g.CreatePurposeAttr(UsdGeom.Tokens.guide)
            UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        if info["mass"] > 0:
            p_base = T_mount @ info["T"] @ np.r_[info["com"], 1.0]
            parts.append((info["mass"], p_base[:3]))
    base_prim = scene.GetPrimAtPath(BASE)
    mass_api = UsdPhysics.MassAPI(base_prim)
    m_b = mass_api.GetMassAttr().Get() if base_prim.HasAPI(UsdPhysics.MassAPI) else None
    axes = mass_api.GetPrincipalAxesAttr().Get() if m_b else None
    mass_note = ""
    if m_b:
        c_b = np.array(mass_api.GetCenterOfMassAttr().Get() or (0, 0, 0), float)
        I_b = np.array(mass_api.GetDiagonalInertiaAttr().Get() or (0, 0, 0), float)
        m_add = sum(m for m, _ in parts)
        M_tot = m_b + m_add
        c_new = (m_b * c_b + sum(m * p for m, p in parts)) / M_tot
        I_new = I_b.copy()
        for m, p in [(m_b, c_b)] + parts:  # параллельный перенос к новому центру масс (только диагональ)
            d = p - c_new
            I_new += m * (d @ d - d * d)
        over = UsdPhysics.MassAPI.Apply(scene.OverridePrim(BASE))
        over.CreateMassAttr(float(M_tot))
        over.CreateCenterOfMassAttr(Gf.Vec3f(*c_new.tolist()))
        over.CreateDiagonalInertiaAttr(Gf.Vec3f(*I_new.tolist()))
        mass_note = f"масса корпуса {m_b:.3f} -> {M_tot:.3f} кг, центр масс {np.round(c_b * 1000, 1)} -> {np.round(c_new * 1000, 1)} мм"
        if axes is not None and abs(abs(axes.GetReal()) - 1.0) > 1e-3:
            mass_note += " (главные оси инерции корпуса повёрнуты — добавка к инерции приближённая)"
    else:  # масса корпуса считается по коллизиям — добавляем массы нашим фигурам
        for name, info in links.items():
            cols = [p for p in Usd.PrimRange(scene.GetPrimAtPath(f"{BASE}/sensor_mount/colliders")) if p.GetName().startswith(name + "_")]
            for p in cols:
                UsdPhysics.MassAPI.Apply(p).CreateMassAttr(info["mass"] / len(cols))
        mass_note = "масса корпуса считается по коллизиям — массы надстройки заданы на её фигурах"

    # Кадры датчиков и сами датчики.
    for name, info in links.items():
        if name == root:
            continue
        fr = UsdGeom.Xform.Define(scene, f"{BASE}/sensor_mount/frames/{name}")
        set_matrix(fr, info["T"])
    lid_frame = f"{BASE}/sensor_mount/frames/livox_frame"
    mid = UsdGeom.Xform.Define(scene, f"{lid_frame}/Mid360")
    mid.GetPrim().GetReferences().AddReference(os.path.relpath(MID360_ASSET, out.parent))
    for _ in range(5):
        app.update()
    lidar = next(p for p in Usd.PrimRange(mid.GetPrim()) if p.GetTypeName() == "OmniLidar")
    L = np.linalg.inv(world(mid.GetPrim())) @ world(lidar)
    if np.abs(L[:3, :3] - np.eye(3)).max() > 1e-6:
        raise RuntimeError("оси лидара в ассете Mid-360 повёрнуты относительно ассета — нужна поправка")
    UsdGeom.Xformable(mid).AddTranslateOp().Set(Gf.Vec3d(*(-L[:3, 3]).tolist()))
    # В ассете Mid-360 своё твёрдое тело и IMU: внутри твёрдого корпуса Go2 вложенное тело
    # недопустимо (PhysX ругается, IMU ассета читает не те скорости) — тело снимаем, IMU выключаем.
    for p in Usd.PrimRange(mid.GetPrim()):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            scene.OverridePrim(p.GetPath()).RemoveAPI(UsdPhysics.RigidBodyAPI)
        if "Imu" in p.GetTypeName():
            scene.OverridePrim(p.GetPath()).SetActive(False)
    cad = scene.GetPrimAtPath(f"{lid_frame}/Mid360/torso_link/mid360_link/CAD")
    if cad.IsValid():
        UsdGeom.Imageable(scene.OverridePrim(cad.GetPath())).MakeInvisible()  # виден корпус из Fusion
    for lens in ("insta360_front", "insta360_back"):
        if scene.GetPrimAtPath(f"{BASE}/sensor_mount/frames/{lens}").IsValid():
            add_fisheye(scene, f"{BASE}/sensor_mount/frames/{lens}/fisheye")
    scene.GetRootLayer().Save()

    # Сводка: где что относительно корпуса и пола (высота корпуса над полом — по стопам в сцене).
    feet = [world(stage.GetPrimAtPath(f"/World/go2/{f}_foot"))[2, 3] for f in ("FL", "FR", "RL", "RR")
            if stage.GetPrimAtPath(f"/World/go2/{f}_foot").IsValid()]
    ground = min(feet) - 0.022 if feet else None  # радиус стопы Go2 ~2.2 см
    for nm in ("livox_frame", "insta360_front", "insta360_back"):
        if nm in links:
            p = T_mount @ links[nm]["T"][:, 3]
            hz = f", над полом {(M_base[2, 3] + p[2] - ground) * 1000:.0f} мм" if ground is not None else ""
            print(f"[go2] {nm:15s} в кадре корпуса {np.round(p[:3] * 1000, 1)} мм{hz}", flush=True)
    print(f"[go2] подошва надстройки {(f_hi[0] - f_lo[0]) * 1000:.0f} мм: от {(x_c + f_lo[0]) * 1000:+.0f} до {(x_c + f_hi[0]) * 1000:+.0f} мм "
          f"вдоль корпуса (ровная площадка спины от {rear_edge * 1000:+.0f} мм), стоит на высоте {z_top * 1000:.1f} мм "
          f"над центром корпуса ({contact})", flush=True)
    print(f"[go2] {mass_note}", flush=True)
    print(f"[go2] {out}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
