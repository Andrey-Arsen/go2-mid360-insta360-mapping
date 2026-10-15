"""Общее для езды (run_go2_isaac.py) и перерисовки (render_drive.py): сцена Go2 + окружение.

Импортировать после создания SimulationApp. Сцена — robot/go2_mapping/go2_mapping.usda
(scripts/build_go2_mapping.py): Go2 из IsaacLegs с надстройкой и нашими датчиками.
"""

from pathlib import Path

import numpy as np
import omni.usd
from isaacsim.core.utils.stage import add_reference_to_stage, is_stage_loading
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "robot" / "go2_mapping" / "go2_mapping.usda"
ROBOT = "/World/go2"
BASE = f"{ROBOT}/base"
FRAMES = f"{BASE}/sensor_mount/frames"
LENSES = {"front": f"{FRAMES}/insta360_front/fisheye", "back": f"{FRAMES}/insta360_back/fisheye"}
ENVIRONMENTS = {  # как в IsaacLegs (launch_scene.py), пути на сервере ассетов Isaac
    "none": None,
    "flat_grid": "/Isaac/Environments/Grid/default_environment.usd",
    "simple_room": "/Isaac/Environments/Simple_Room/simple_room.usd",
    "warehouse": "/Isaac/Environments/Simple_Warehouse/warehouse.usd",
    "full_warehouse": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd",
    "warehouse_shelves": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd",
    "office": "/Isaac/Environments/Office/office.usd",
    "hospital": "/Isaac/Environments/Hospital/hospital.usd",
}


def env_url(name):
    """Имя окружения (или путь к .usd, или уже полный URL) -> URL/путь; None — без окружения."""
    if name in (None, "", "none"):
        return None
    if "://" in name:
        return name
    if name in ENVIRONMENTS:
        from isaacsim.storage.native import get_assets_root_path
        root = get_assets_root_path()
        if root is None:
            raise SystemExit("нет связи с сервером ассетов Isaac — окружение не загрузить (нужен интернет)")
        return root + ENVIRONMENTS[name]
    if Path(name).exists():
        return str(Path(name).resolve())
    raise SystemExit(f"неизвестное окружение {name}: {', '.join(ENVIRONMENTS)} или путь к .usd")


def env_prim_path(url):
    return f"/World/{Path(url).stem}"


def wait_loading(app):
    app.update()
    app.update()
    while is_stage_loading():
        app.update()


def open_scene(app, env=None, scene=SCENE):
    """Открывает сцену робота и добавляет окружение; возвращает (stage, url окружения)."""
    scene = Path(scene).resolve()
    if not scene.exists():
        raise SystemExit(f"нет {scene} — сначала ~/isaacsim/python.sh scripts/build_go2_mapping.py")
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene)):
        raise SystemExit(f"не открылась {scene}")
    wait_loading(app)
    url = env_url(env)
    if url is not None:
        print(f"[go2] окружение {url} (первый раз скачивается — может занять минуты)", flush=True)
        add_reference_to_stage(url, env_prim_path(url))
        wait_loading(app)
    return ctx.get_stage(), url


def set_spawn(stage, spawn, robot_path=ROBOT):
    """Ставит робота в (X, Y, курс в градусах) на сессионном слое (файл сцены не меняется)."""
    if spawn is None:
        return
    x, y, yaw = spawn
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot = UsdGeom.Xformable(stage.GetPrimAtPath(robot_path))
        z = robot.ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()[2]
        robot.ClearXformOpOrder()
        robot.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        robot.AddRotateZOp().Set(yaw)


def disable_graphs(names=None):
    """Выключает графы OmniGraph IsaacLegs (по умолчанию все в /Graphs). Деактивация прима
    не останавливает уже загруженный граф — выключаем сам граф."""
    import omni.graph.core as og
    stage = omni.usd.get_context().get_stage()
    if names is None:
        root = stage.GetPrimAtPath("/Graphs")
        names = [str(p.GetPath()) for p in root.GetChildren()] if root.IsValid() else []
    for g in names:
        graph = og.get_graph_by_path(g)
        if graph is not None and graph.is_valid():
            graph.set_disabled(True)


def lidar_path(stage):
    lidar = next((p for p in Usd.PrimRange(stage.GetPrimAtPath(f"{FRAMES}/livox_frame"))
                  if p.GetTypeName() == "OmniLidar"), None)
    if lidar is None:
        raise SystemExit(f"нет Mid-360 под {FRAMES}/livox_frame — пересоберите scripts/build_go2_mapping.py")
    return str(lidar.GetPath())


def robot_links(stage):
    """Твёрдые тела робота (звенья) — их позы пишет запись поездки."""
    return [str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROBOT)) if p.HasAPI(UsdPhysics.RigidBodyAPI)]


def world_matrix(stage, path):
    """4x4 «столбцами»: x_world = M @ x_local."""
    return np.array(UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())).T
