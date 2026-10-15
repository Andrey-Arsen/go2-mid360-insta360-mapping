#!/usr/bin/env python3
"""Выгружает эталонную геометрию окружения в один PLY в мировых координатах — для оценки меша.

Сессия робота (render_drive.py): окружение из meta.json (склад и т. п., в т. ч. с сервера
ассетов Isaac) -> meta["gt_mesh"] (out/gt/<окружение>.ply). Берутся все видимые сетки
окружения, включая повторённые экземплярами (instanceable) — робота в файле окружения нет.
Сессия стенда Simple Room (record_session.py, без "environment" в meta) или без --session —
как раньше: сетки под /World/Environment сцены стенда -> out/gt/simple_room_world.ply.

Многоугольники режутся на треугольники веером.

Запуск: ~/isaacsim/python.sh scripts/export_scene_mesh.py [--session ИМЯ]
"""

import argparse
import json
import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--session", help="сессия: окружение и путь эталона берутся из её meta.json")
args, _ = parser.parse_known_args()

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.utils.stage import is_stage_loading  # noqa: E402
from pxr import Usd, UsdGeom  # noqa: E402

STAND_SCENE = ROOT / "scenes" / "simple_room_mid360_fisheye.usda"
STAND_GT = ROOT / "out" / "gt" / "simple_room_world.ply"


def write_ply(path, v, f):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\n"
                  f"property float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(v.astype("<f4").tobytes())
        rec = np.zeros(len(f), dtype=[("n", "u1"), ("i", "<i4", 3)])
        rec["n"], rec["i"] = 3, f
        fh.write(rec.tobytes())


def collect(stage, root_path, scale=1.0):
    verts, faces, offset, used = [], [], 0, 0
    root = stage.GetPrimAtPath(root_path) if root_path else stage.GetPseudoRoot()
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible or img.ComputePurpose() in (UsdGeom.Tokens.guide, UsdGeom.Tokens.proxy):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.array(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        idx = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if not len(pts) or not len(counts) or counts.min() < 3:
            continue
        m = np.array(cache.GetLocalToWorldTransform(prim))
        verts.append((np.c_[pts, np.ones(len(pts))] @ m)[:, :3] * scale)
        first = np.r_[0, np.cumsum(counts)[:-1]]
        tris = [np.c_[idx[first[counts > k + 1]], idx[first[counts > k + 1] + k], idx[first[counts > k + 1] + k + 1]]
                for k in range(1, counts.max() - 1)]
        faces.append(np.vstack(tris) + offset)
        offset += len(pts)
        used += 1
    if not used:
        raise SystemExit(f"в {root_path or 'сцене'} нет видимых сеток")
    return np.concatenate(verts), np.concatenate(faces), used


def main():
    meta = None
    if args.session:
        meta = json.loads((ROOT / "data" / "sessions" / args.session / "meta.json").read_text(encoding="utf-8"))
    if meta is None or "environment" not in meta:
        out = STAND_GT
        stage = Usd.Stage.Open(str(STAND_SCENE))
        v, f, used = collect(stage, "/World/Environment")
    else:
        if not meta.get("environment"):
            raise SystemExit("у сессии нет окружения — эталона нет")
        out = ROOT / meta["gt_mesh"]
        ctx = omni.usd.get_context()
        if not ctx.open_stage(meta["environment"]):
            raise SystemExit(f"не открылось окружение {meta['environment']}")
        app.update()
        app.update()
        while is_stage_loading():
            app.update()
        stage = ctx.get_stage()
        mpu = UsdGeom.GetStageMetersPerUnit(stage)
        if abs(mpu - 1.0) > 1e-6:
            print(f"[gt] ВНИМАНИЕ: окружение в {mpu} м на единицу — эталон переведён в метры", flush=True)
        v, f, used = collect(stage, None, scale=mpu)
    write_ply(out, v, f)
    print(f"[gt] {out}: мешей {used}, вершин {len(v)}, треугольников {len(f)}, "
          f"границы {v.min(0).round(2).tolist()} .. {v.max(0).round(2).tolist()}", flush=True)


try:
    main()
except SystemExit as e:
    print(f"[gt] {e}", flush=True)
    sys.exit(1)
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
