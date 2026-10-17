#!/usr/bin/env python3
"""Качество модели с новых точек: Isaac рисует исходное окружение и восстановленную сцену
(меш <с>_rko.usda или гауссианы <с>_gs.usda) с ракурсов, где камер робота не было, и
сравнивает их (PSNR, SSIM). Именно оттуда видны «иголки» гауссиан и бугры меша; кадры, на
которых училась модель, этого не покажут (eval_views.py).

Ракурсы — N точек вдоль пути сессии, на каждой два: «вбок» — с высоты камер робота (0.65 м),
повернувшись на 60° от хода (поочерёдно влево и вправо), и «сверху» — с высоты человека
(1.5 м) по ходу, чуть вниз. Оригинал рисуется один раз и хранится в out/novel/<сессия>/orig/.

--align <набор COLMAP модели>: модель на позах одометрии лежит в её системе, а та целиком уплыла от
эталона (у sklad03 на ~5 см и ~0.4° по курсу); без совмещения PSNR меряет этот общий дрейф, а не
качество. Совмещение — жёсткое, по центрам камер набора модели и эталонного colmap_gt.

Пишет out/novel/<сессия>/<сцена>.json (PSNR/SSIM по ракурсам и средние) и <сцена>.png
(сверху оригинал, снизу модель).
Запуск: ~/isaacsim/python.sh scripts/eval_novel_views.py <сессия> scenes/reconstructed/<сцена>.usda [--n 6]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("session")
parser.add_argument("scene", help="восстановленная сцена .usda")
parser.add_argument("--n", type=int, default=6, help="сколько точек вдоль пути (ракурсов вдвое больше)")
parser.add_argument("--size", type=int, nargs=2, default=(640, 400), metavar=("W", "H"))
parser.add_argument("--align", help="набор COLMAP, на позах которого училась модель: совместить с эталонным colmap_gt")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import carb  # noqa: E402
import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from pxr import Gf, Sdf, UsdGeom  # noqa: E402

enable_extension("omni.usd.schema.omni_nurec_types")  # гауссианы NuRec
sys.path.insert(0, str(Path(__file__).resolve().parent))
from image_metrics import psnr, ssim  # noqa: E402


def novel_views(session, n):
    """{имя: (глаз, точка взгляда)} вдоль пути сессии (эталонная траектория лидара)."""
    p = np.load(ROOT / "data" / "sessions" / session / "gt.npz")["pos"]
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
    views = {}
    for k in range(n):
        i = int(np.searchsorted(s, (k + 0.5) / n * s[-1]))
        j = int(np.searchsorted(s, min(s[i] + 0.5, s[-1])))
        d = p[min(j, len(p) - 1), :2] - p[max(i - 1, 0), :2]
        d = d / max(np.linalg.norm(d), 1e-6)
        a = np.radians(60.0 if k % 2 == 0 else -60.0)
        side = np.array([d[0] * np.cos(a) - d[1] * np.sin(a), d[0] * np.sin(a) + d[1] * np.cos(a)])
        views[f"side_{k}"] = ((p[i, 0], p[i, 1], 0.65), (p[i, 0] + 4 * side[0], p[i, 1] + 4 * side[1], 0.5))
        views[f"high_{k}"] = ((p[i, 0], p[i, 1], 1.5), (p[i, 0] + 5 * d[0], p[i, 1] + 5 * d[1], 0.6))
    return views


def camera_centers(sparse):
    """{имя кадра: центр камеры} из images.txt набора COLMAP."""
    out = {}
    lines = [ln for ln in open(sparse / "images.txt") if ln.strip() and not ln.startswith("#")]
    for ln in lines:
        e = ln.split()
        if len(e) < 10:
            continue
        qw, qx, qy, qz, tx, ty, tz = map(float, e[1:8])
        R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                      [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                      [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
        out[e[9]] = -R.T @ np.array([tx, ty, tz])
    return out


def gt_to_model(model_sparse, gt_sparse):
    """Жёсткое преобразование эталон -> система модели по центрам камер (Umeyama без масштаба)."""
    m, g = camera_centers(model_sparse), camera_centers(gt_sparse)
    names = sorted(set(m) & set(g))
    M, G = np.array([m[n] for n in names]), np.array([g[n] for n in names])
    mu_m, mu_g = M.mean(0), G.mean(0)
    U, _, Vt = np.linalg.svd((M - mu_m).T @ (G - mu_g))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    return R, mu_m - R @ mu_g


def apply_render_settings(scene):
    """renderSettings корневого слоя сцены -> RTX (у сцены гауссиан — обратная тональная кривая NuRec)."""
    if "://" in str(scene):
        return
    for key, value in Sdf.Layer.FindOrOpen(str(scene)).customLayerData.get("renderSettings", {}).items():
        carb.settings.get_settings().set("/" + key.replace(":", "/"), value)


def render(scene, views, size):
    ctx = omni.usd.get_context()
    if not ctx.open_stage(str(scene)):
        raise SystemExit(f"не открылась {scene}")
    for _ in range(10):
        app.update()
    while is_stage_loading():
        app.update()
    apply_render_settings(scene)
    stage = ctx.get_stage()
    annots = {}
    for name, (eye, target) in views.items():
        cam = UsdGeom.Camera.Define(stage, f"/NovelCams/{name}")
        cam.CreateFocalLengthAttr(12.0)
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 200.0))
        f = (Gf.Vec3d(*target) - Gf.Vec3d(*eye)).GetNormalized()
        right = Gf.Cross(f, Gf.Vec3d(0, 0, 1)).GetNormalized()
        up = Gf.Cross(right, f)
        UsdGeom.Xformable(cam).AddTransformOp().Set(Gf.Matrix4d(*right, 0, *up, 0, *(-f), 0, *eye, 1))
        annot = rep.AnnotatorRegistry.get_annotator("rgb")
        annot.attach([rep.create.render_product(str(cam.GetPath()), tuple(size))])
        annots[name] = annot
    for _ in range(4):
        rep.orchestrator.step(rt_subframes=8)
    return {n: np.array(a.get_data())[..., :3] for n, a in annots.items()}


def main():
    meta = json.loads((ROOT / "data" / "sessions" / args.session / "meta.json").read_text(encoding="utf-8"))
    if not meta.get("environment"):
        raise SystemExit("у сессии нет окружения (эталона) — сравнивать не с чем")
    scene = Path(args.scene).resolve()
    out = ROOT / "out" / "novel" / args.session
    (out / "orig").mkdir(parents=True, exist_ok=True)
    views = novel_views(args.session, args.n)
    orig = {}
    cached = {n: out / "orig" / f"{n}_{args.size[0]}x{args.size[1]}.png" for n in views}
    if all(f.exists() for f in cached.values()):
        orig = {n: np.asarray(Image.open(f))[..., :3] for n, f in cached.items()}
    else:
        orig = render(meta["environment"], views, args.size)
        for n, img in orig.items():
            Image.fromarray(img).save(cached[n])
    recon_views = views
    if args.align:
        gt_sparse = ROOT / "data" / "sessions" / args.session / "colmap_gt" / "sparse" / "0"
        if not gt_sparse.exists():
            raise SystemExit(f"нет {gt_sparse} — scripts/session_to_colmap.py {args.session} --poses gt")
        R, t = gt_to_model(Path(args.align) / "sparse" / "0", gt_sparse)
        recon_views = {n: (tuple(R @ np.array(e) + t), tuple(R @ np.array(g) + t)) for n, (e, g) in views.items()}
    recon = render(scene, recon_views, args.size)

    rows = []
    for n in views:
        a, b = orig[n].astype(np.float64) / 255.0, recon[n].astype(np.float64) / 255.0
        m = np.ones(a.shape[:2], bool)
        rows.append({"view": n, "psnr": round(psnr(b, a, m), 2), "ssim": round(ssim(b, a, m), 4),
                     "empty_share": round(float((recon[n].sum(2) < 8).mean()), 3)})
    mean = {k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("psnr", "ssim", "empty_share")}
    side = round(float(np.mean([r["psnr"] for r in rows if r["view"].startswith("side")])), 2)
    high = round(float(np.mean([r["psnr"] for r in rows if r["view"].startswith("high")])), 2)
    report = {"session": args.session, "scene": str(scene), "aligned_with": args.align, "mean": mean, "psnr_side": side, "psnr_high": high, "views": rows}
    tag = scene.stem + ("_aligned" if args.align else "")
    (out / f"{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    w, h = args.size
    names = list(views)
    cols = min(len(names), 6)
    nrow = (len(names) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * w // 2, nrow * 2 * h // 2 + 24 * nrow), (24, 24, 28))
    for k, n in enumerate(names):
        c, r = k % cols, k // cols
        y0 = r * (h + 24)
        sheet.paste(Image.fromarray(orig[n]).resize((w // 2, h // 2)), (c * w // 2, y0 + 24))
        sheet.paste(Image.fromarray(recon[n]).resize((w // 2, h // 2)), (c * w // 2, y0 + 24 + h // 2))
        ImageDraw.Draw(sheet).text((c * w // 2 + 4, y0 + 6), f"{n}: {rows[k]['psnr']:.1f} dB", fill=(230, 230, 230))
    sheet.save(out / f"{tag}.png")
    print(f"[novel] {tag}: PSNR {mean['psnr']:.2f} дБ (вбок {side:.2f}, сверху {high:.2f}), SSIM {mean['ssim']:.3f}, "
          f"пусто {mean['empty_share']:.1%}; {out / (tag + '.png')}", flush=True)


try:
    main()
except SystemExit as e:
    print(f"[novel] {e}", flush=True)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
