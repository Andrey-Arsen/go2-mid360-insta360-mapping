#!/usr/bin/env python3
"""Сцена для Isaac Sim: картинка — гауссианы (train_gs.sh), физика — оболочка fit_shell.py.

scenes/reconstructed/<s>_gs.usda:
  /World/Gaussians       NuRec USDZ (копия out/gs/<s>_<позы>/gaussians.usdz рядом со сценой);
                         в метрах и в тех же координатах, что меш, — позы из лидара. Настройки
                         рендера NuRec (обратная тональная кривая для гауссиан) — в корневом
                         слое сцены, иначе Isaac их не применит и гауссианы выйдут белёсыми;
                         они же меняют тональную кривую сцены (op 2), поэтому скрытый цветной
                         меш здесь, если его показать, выйдет светлее, чем в <s>_rko.usda
  /World/Shell           коллайдер: оболочка, плинтус и карниз, предметы (скрыт), если она есть
  /World/Reconstruction  цветной меш (скрыт); столкновения выключены, если есть оболочка, иначе
                         коллайдер — он
  /World/Cloud           облако лидара (скрыто), если есть
  /World/RoomLight       свет для роботов и мешей; на гауссианы не влияет — они и есть свет
Нужен уже сделанный mesh_to_usd.py (файлы <s>_rko_mesh.usdc, _shell.usdc, _cloud.usdc).

Запуск: .venv/bin/python scripts/gs_scene.py tilt30gs [--poses rko|gt]
"""

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]

def nurec_render_settings(usdz):
    """Настройки рендера из слоёв самого USDZ гауссиан. Главная — invertToneMap: Isaac снимает
    тональную кривую с гауссиан, и они выходят цветом снимков (без неё — второй раз через кривую,
    белёсо: стена склада 221 вместо 166). Isaac берёт настройки только из корневого слоя
    открытой сцены, поэтому переносим их в нашу."""
    settings = {}
    with tempfile.TemporaryDirectory() as tmp, zipfile.ZipFile(usdz) as z:
        for name in z.namelist():
            if name.endswith((".usda", ".usdc", ".usd")):
                path = Path(tmp) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(z.read(name))
                settings.update(Sdf.Layer.FindOrOpen(str(path)).customLayerData.get("renderSettings", {}))
    return settings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--poses", choices=("rko", "gt"), default="rko")
    args = parser.parse_args()
    out = ROOT / "scenes" / "reconstructed"
    src = ROOT / "out" / "gs" / f"{args.session}_{args.poses}" / "gaussians.usdz"
    if not src.exists():
        raise SystemExit(f"нет {src} — сначала scripts/train_gs.sh {args.session} {args.poses}")
    name = f"{args.session}_gs" + ("" if args.poses == "rko" else "_gt")
    gs = out / f"{name}.usdz"
    shutil.copyfile(src, gs)
    render_settings = nurec_render_settings(gs)
    base = f"{args.session}_rko"
    if not (out / f"{base}_mesh.usdc").exists():
        raise SystemExit(f"нет {out / base}_mesh.usdc — сначала scripts/mesh_to_usd.py out/mesh/{base}")
    has_shell = (out / f"{base}_shell.usdc").exists()  # нет оболочки PolyFit — коллайдер из самого меша

    scene_path = out / f"{name}.usda"
    scene = Usd.Stage.CreateNew(str(scene_path))
    UsdGeom.SetStageUpAxis(scene, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(scene, 1.0)
    scene.SetTimeCodesPerSecond(60)
    scene.SetStartTimeCode(0)
    scene.SetEndTimeCode(1000000)
    world = UsdGeom.Xform.Define(scene, "/World")
    scene.SetDefaultPrim(world.GetPrim())
    UsdGeom.Xform.Define(scene, "/World/Gaussians").GetPrim().GetReferences().AddReference(f"./{gs.name}")
    if has_shell:
        shell = UsdGeom.Xform.Define(scene, "/World/Shell")
        shell.GetPrim().GetReferences().AddReference(f"./{base}_shell.usdc")
        shell.MakeInvisible()
    mesh = UsdGeom.Xform.Define(scene, "/World/Reconstruction")
    mesh.GetPrim().GetReferences().AddReference(f"./{base}_mesh.usdc")
    mesh.MakeInvisible()  # невидимый меш по-прежнему сталкивается, если оболочки нет
    if has_shell:
        UsdPhysics.CollisionAPI.Apply(scene.OverridePrim("/World/Reconstruction/mesh")).CreateCollisionEnabledAttr(False)
    if (out / f"{base}_cloud.usdc").exists():
        cloud = UsdGeom.Xform.Define(scene, "/World/Cloud")
        cloud.GetPrim().GetReferences().AddReference(f"./{base}_cloud.usdc")
        cloud.MakeInvisible()
    UsdPhysics.Scene.Define(scene, "/World/PhysicsScene")
    light = UsdLux.SphereLight.Define(scene, "/World/RoomLight")
    light.CreateIntensityAttr(30000.0)
    light.CreateRadiusAttr(0.3)
    UsdGeom.Xformable(light).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.8, 3.6))
    if render_settings:
        scene.GetRootLayer().customLayerData = {"renderSettings": render_settings}
    scene.GetRootLayer().Save()
    print(f"[gs] {scene_path}: гауссианы {gs.name} ({gs.stat().st_size / 1e6:.0f} МБ, настроек рендера NuRec "
          f"{len(render_settings)}), "
          f"коллайдер {base + '_shell.usdc' if has_shell else 'сам меш ' + base + '_mesh.usdc'}")


if __name__ == "__main__":
    main()
