#!/usr/bin/env python3
"""Как Isaac Sim переводит свечение материала в пиксель — чтобы цветная модель выглядела как снимки.

Цвет вершин модели взят со снимков, а снимки уже прошли тональную кривую рендера Isaac. Если
отдать этот цвет свечением как есть, при показе он пройдёт кривую второй раз — модель выходит
белёсой (стены склада почти белые вместо серого кирпича). Здесь рендерятся серые квадраты со
свечением известной силы тем же материалом, что у модели (UsdPreviewSurface, emissiveColor), и
той же камерой Replicator, что снимает fisheye, и записывается кривая «свечение -> пиксель sRGB».
mesh_to_usd.py и export_cloud.py берут обратную к ней: пиксель снимка -> нужное свечение.

Пишет config/isaac_color_lut.json. Перезапускать, только если меняли настройки рендера Isaac
(тональная кривая /rtx/post/tonemap/*, экспозиция).

Запуск: ~/isaacsim/python.sh scripts/isaac_color_lut.py
"""

import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdShade  # noqa: E402

GRID = 8
SIZE = 512


def patch_material(stage, path, value):
    mat = UsdShade.Material.Define(stage, path)
    surface = UsdShade.Shader.Define(stage, f"{path}/surface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    surface.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(value, value, value))
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
    surface.CreateInput("specularColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.0, 0.0, 0.0))
    surface.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(1)
    mat.CreateSurfaceOutput().ConnectToSource(surface.CreateOutput("surface", Sdf.ValueTypeNames.Token))
    return mat


def measure(values):
    """Квадраты GRID x GRID со свечением values перед камерой -> пиксель sRGB в центре каждого."""
    omni.usd.get_context().new_stage()
    app.update()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    step = 1.0
    for n, v in enumerate(values):
        i, j = divmod(n, GRID)
        x, y = (j - (GRID - 1) / 2) * step, ((GRID - 1) / 2 - i) * step
        quad = UsdGeom.Mesh.Define(stage, f"/World/P{n:03d}")
        h = step * 0.45
        quad.CreatePointsAttr([Gf.Vec3f(x - h, y - h, 0), Gf.Vec3f(x + h, y - h, 0), Gf.Vec3f(x + h, y + h, 0), Gf.Vec3f(x - h, y + h, 0)])
        quad.CreateFaceVertexCountsAttr([4])
        quad.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
        UsdShade.MaterialBindingAPI.Apply(quad.GetPrim()).Bind(patch_material(stage, f"/World/Looks/M{n:03d}", float(v)))
    cam = UsdGeom.Camera.Define(stage, "/World/Camera")
    cam.CreateFocalLengthAttr(10.0)
    cam.CreateHorizontalApertureAttr(20.955)
    cam.CreateVerticalApertureAttr(20.955)
    UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0, 0, GRID * step * 1.05))  # камера USD смотрит по −Z
    rp = rep.create.render_product("/World/Camera", (SIZE, SIZE))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach([rp])
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=8)
    img = np.asarray(rgb.get_data())[..., :3].astype(float)
    if img.ndim != 3:
        raise SystemExit(f"[lut] нет кадра с камеры: {img.shape}")
    rgb.detach()
    rp.destroy()
    # Центры квадратов в кадре: камера по центру сетки, поле зрения — по апертуре.
    fov_half = np.arctan(20.955 / 2 / 10.0)
    dist = GRID * step * 1.05
    out = []
    for n in range(len(values)):
        i, j = divmod(n, GRID)
        x, y = (j - (GRID - 1) / 2) * step, ((GRID - 1) / 2 - i) * step
        u = int(SIZE / 2 + x / (dist * np.tan(fov_half)) * SIZE / 2)
        v = int(SIZE / 2 - y / (dist * np.tan(fov_half)) * SIZE / 2)
        out.append(img[v - 3:v + 4, u - 3:u + 4].mean())
    return np.array(out)


def main():
    s = carb.settings.get_settings()
    settings = {k: s.get(k) for k in ("/rtx/post/tonemap/op", "/rtx/post/histogram/enabled", "/rtx/post/tonemap/filmIso",
                                      "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/fNumber",
                                      "/rtx/post/tonemap/whitepoint", "/rtx/post/tonemap/enableSrgbToGamma")}
    print(f"[lut] настройки рендера: {settings}", flush=True)
    values = np.r_[0.0, np.geomspace(0.002, 8.0, GRID * GRID - 1)]
    a = measure(values)
    b = measure(values[::-1])[::-1]  # те же квадраты в другом порядке: если включена автоэкспозиция,
    #                                  сумма яркости кадра та же — проверка расположения
    c = measure(values * 0.25)       # другая общая яркость кадра: должен совпасть кусок кривой
    shift = np.interp(values * 0.25, values, a) - c
    print(f"[lut] повтор с другим порядком: расхождение до {np.abs(a - b).max():.2f} из 255; "
          f"при вчетверо темнее кадре: до {np.abs(shift).max():.2f}", flush=True)
    if np.abs(shift).max() > 3:
        print("[lut] ВНИМАНИЕ: кривая зависит от яркости кадра (автоэкспозиция?) — точной обратной нет", flush=True)
    order = np.argsort(values)
    pix = np.maximum.accumulate(a[order])  # монотонно
    lut = {"emissive": values[order].round(6).tolist(), "pixel_srgb8": pix.round(3).tolist(), "render_settings": settings,
           "note": "isaac_color_lut.py: свечение UsdPreviewSurface (линейное) -> пиксель rgb Replicator (0..255)"}
    out = ROOT / "config" / "isaac_color_lut.json"
    out.write_text(json.dumps(lut, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for v in (0.05, 0.2, 0.5, 1.0):
        print(f"[lut] свечение {v}: пиксель {np.interp(v, values[order], pix):.1f} (sRGB без кривой: "
              f"{255 * (1.055 * v ** (1 / 2.4) - 0.055 if v > 0.0031308 else 12.92 * v):.1f})", flush=True)
    print(f"[lut] готово: {out}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
