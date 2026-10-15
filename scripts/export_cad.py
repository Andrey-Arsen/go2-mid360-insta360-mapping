#!/usr/bin/env python3
"""Оболочка комнаты (fit_shell.py) -> STEP и план DXF для CAD.

room.step — твёрдые тела в миллиметрах, через ядро OpenCASCADE в gmsh: комната (объём
  воздуха из shell.obj, минус плинтус и карниз), плинтус и карниз (details/*.obj),
  ниши проёмов (details/opening_*.obj, между стеной и стеклом). Порядок и объёмы тел —
  в cad_report.json. Открывается во FreeCAD, Fusion 360, SolidWorks, AutoCAD; стены
  с толщиной в CAD делаются оболочкой (Shell/Thickness) тела комнаты наружу.
plan.dxf — план в миллиметрах (plan.png — он же картинкой):
  WALLS      срез оболочки на высоте --cut (по умолчанию 1 м, как на строительных планах);
  FURNITURE  контуры предметов ниже среза (выпуклая оболочка в проекции сверху);
  OVERHEAD   предметы выше среза (светильники) — штрихом;
  WINDOWS    проёмы, которые срез пересекает: ниша от стены до стекла;
  DIMENSIONS габариты и подпись: площадь пола, высота, объём.
Скругления у стен (в objects.ply) на план не идут — это не мебель.

Проверка: STEP читается обратно, тел столько же, объём комнаты (без вычитания деталей)
совпадает с объёмом многогранника.

Запуск: .venv/bin/python scripts/export_cad.py out/mesh/tilt30_rko [--cut 1.0]
"""

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

import ezdxf
import gmsh
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fit_shell import exact_vertices, read_obj_polygons, triangulate  # noqa: E402

MM = 1000.0


def polyhedron_volume(V, polys):
    """Объём по теореме Гаусса: сумма тетраэдров от начала координат (нормали наружу)."""
    vol = 0.0
    for poly in polys:
        for a, b, c in triangulate(V, poly):
            vol += np.dot(V[a], np.cross(V[b], V[c])) / 6.0
    return vol


def face_normal_area(V, poly):
    n = np.zeros(3)
    p = V[poly]
    for a, b in zip(p, np.roll(p, -1, 0)):
        n += np.cross(a, b)
    area = np.linalg.norm(n) / 2
    return n / (2 * area), area


@contextlib.contextmanager
def quiet_stdout():
    """Писатель STEP OpenCASCADE печатает баннеры прямо в дескриптор 1."""
    sys.stdout.flush()
    saved = os.dup(1)
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        try:
            yield
        finally:
            os.dup2(saved, 1)
            os.close(saved)


def add_body(occ, V, polys):
    pts = {v: occ.addPoint(*(V[v] * MM)) for poly in polys for v in poly}
    lines, surfaces = {}, []
    for poly in polys:
        loop = []
        for a, b in zip(poly, poly[1:] + poly[:1]):
            key = (min(a, b), max(a, b))
            if key not in lines:
                lines[key] = occ.addLine(pts[key[0]], pts[key[1]])
            loop.append(lines[key] if a == key[0] else -lines[key])
        surfaces.append(occ.addPlaneSurface([occ.addCurveLoop(loop)]))
    return occ.addVolume([occ.addSurfaceLoop(surfaces)])


def write_step(path, bodies, cut_from_room=()):
    """bodies: [(имя, V, грани)], первое — комната. Тела cut_from_room вычитаются из
    комнаты (комната — воздух), сами остаются отдельными телами."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("room")
    occ = gmsh.model.occ
    tags = {name: add_body(occ, V, polys) for name, V, polys in bodies}
    occ.synchronize()
    room = bodies[0][0]
    volumes = {name: occ.getMass(3, tag) / MM ** 3 for name, tag in tags.items()}
    tools = [(3, tags[n]) for n in cut_from_room if n in tags]
    cut = False
    if tools:
        try:
            out, _ = occ.cut([(3, tags[room])], tools, removeObject=False, removeTool=False)
            occ.synchronize()
            pieces = sorted(((occ.getMass(3, t), t) for d, t in out if d == 3), reverse=True)
            # Грани деталей лежат на гранях комнаты — OCC может оставить микроскопические осколки.
            if pieces and sum(v for v, _ in pieces[1:]) < 1e-6 * pieces[0][0]:
                occ.remove([(3, tags[room])], recursive=True)
                occ.remove([(3, t) for _, t in pieces[1:]], recursive=True)
                tags[room], cut = pieces[0][1], True
            else:
                occ.remove([(3, t) for _, t in pieces], recursive=True)
                print(f"[cad] вычитание деталей дало {len(pieces)} тел — оставляю тела внахлёст")
            occ.synchronize()
        except Exception as e:  # noqa: BLE001 — булева операция OCC может не сойтись; тогда тела внахлёст
            print(f"[cad] вычесть детали из комнаты не вышло ({e}); тела внахлёст")
    volumes[room + "_air"] = occ.getMass(3, tags[room]) / MM ** 3
    with quiet_stdout():
        gmsh.write(str(path))
    gmsh.finalize()
    # Читаем обратно — что увидит CAD.
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    with quiet_stdout():
        gmsh.open(str(path))
    vols = gmsh.model.getEntities(3)
    faces = len(gmsh.model.getEntities(2))
    gmsh.finalize()
    return volumes, len(vols), faces, cut


def section(V, polys, z):
    """Отрезки пересечения граней с плоскостью z, сцепленные в контуры."""
    segs = []
    for poly in polys:
        p = V[poly]
        h = p[:, 2] - z
        cut = []
        for i in range(len(p)):
            j = (i + 1) % len(p)
            if (h[i] < 0) != (h[j] < 0):
                t = h[i] / (h[i] - h[j])
                cut.append(p[i] + t * (p[j] - p[i]))
        if len(cut) == 2:
            segs.append((cut[0][:2], cut[1][:2]))
    loops, left = [], list(segs)
    while left:
        a, b = left.pop()
        loop = [a, b]
        grown = True
        while grown:
            grown = False
            for k, (c, d) in enumerate(left):
                if np.linalg.norm(c - loop[-1]) < 1e-6:
                    loop.append(d)
                elif np.linalg.norm(d - loop[-1]) < 1e-6:
                    loop.append(c)
                else:
                    continue
                left.pop(k)
                grown = True
                break
        loops.append(np.array(loop[:-1] if np.linalg.norm(loop[0] - loop[-1]) < 1e-6 else loop))
    return loops


def object_outlines(objects, V, polys, cut, wall_dist=0.3, join=0.3):
    """Связные куски objects.ply: у стен (скругления) — пропускаем; куски ближе join друг к
    другу (рамки в 3D) — один предмет (стол с дырой в столешнице распадается на ножки);
    контур — выпуклая оболочка сверху."""
    walls = [p for p in polys if abs(face_normal_area(V, p)[0][2]) < 0.5]
    tris = np.array([t for p in walls for t in triangulate(V, p)])
    wall_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(tris))
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(wall_mesh))
    clusters, sizes, _ = objects.cluster_connected_triangles()
    clusters = np.asarray(clusters)
    OV, OT = np.asarray(objects.vertices), np.asarray(objects.triangles)
    pieces = []
    for k in range(len(sizes)):
        p = OV[np.unique(OT[clusters == k])]
        near = scene.compute_distance(o3d.core.Tensor(p.astype(np.float32))).numpy() < wall_dist
        if near.mean() <= 0.8:
            pieces.append(p)
    group = list(range(len(pieces)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i

    boxes = [(p.min(0), p.max(0)) for p in pieces]  # по высоте тоже: стол и светильник над ним — разные
    for i in range(len(pieces)):
        for j in range(i + 1, len(pieces)):
            gap = np.maximum(boxes[i][0] - boxes[j][1], boxes[j][0] - boxes[i][1]).max()
            if gap < join:
                group[root(j)] = root(i)
    out = []
    for g in {root(i) for i in range(len(pieces))}:
        p = np.vstack([pieces[i] for i in range(len(pieces)) if root(i) == g])
        hull = ConvexHull(p[:, :2])
        out.append((p[hull.vertices, :2], p[:, 2].min(), p[:, 2].max()))
    return [(xy, zmax <= cut) for xy, zmin, zmax in out]


def draw_preview(path, loops, outlines, text, windows=(), px=1400, pad=80):
    """Картинка плана (то же, что в DXF, без размерных блоков) — посмотреть без CAD."""
    allp = np.vstack(loops)
    lo, hi = allp.min(0), allp.max(0)
    scale = (px - 2 * pad) / max(hi - lo)
    w, h = int((hi[0] - lo[0]) * scale) + 2 * pad, int((hi[1] - lo[1]) * scale) + 2 * pad + 90

    def uv(xy):
        return [(pad + (x - lo[0]) * scale, pad + (hi[1] - y) * scale) for x, y in xy]

    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for loop in loops:
        d.line(uv(np.vstack([loop, loop[:1]])), fill="black", width=5)
    for xy in windows:
        d.polygon(uv(xy), fill=(200, 230, 255), outline=(0, 120, 200), width=3)
    for xy, below in outlines:
        ring = uv(np.vstack([xy, xy[:1]]))
        if below:
            d.line(ring, fill=(30, 140, 60), width=3)
        else:
            for a, b in zip(ring[:-1], ring[1:]):  # штрих
                seg = np.linspace(a, b, max(2, int(np.hypot(b[0] - a[0], b[1] - a[1]) / 12)))
                for k in range(0, len(seg) - 1, 2):
                    d.line([tuple(seg[k]), tuple(seg[k + 1])], fill=(120, 120, 120), width=2)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except OSError:
        font = ImageFont.load_default()
    d.multiline_text((pad, h - 95), text.replace(". ", ".\n", 1), fill="black", font=font, spacing=6)
    d.text((pad + (hi[0] - lo[0]) * scale / 2 - 60, pad - 50), f"{(hi[0] - lo[0]) * MM:.0f} мм", fill="black", font=font)
    img.save(path)


def window_outlines(openings, cut):
    """Ниши проёмов, которые пересекает срез: прямоугольник от стены до стекла, вид сверху."""
    out = []
    for o in openings:
        if not o["z"][0] < cut < o["z"][1]:
            continue
        n = np.array(o["wall_normal"])
        t = np.array([-n[1], n[0], 0.0]) / np.linalg.norm(n[:2])
        out.append(np.array([(s * t - w * n)[:2] for w, s in ((o["wall_offset"], o["s"][0]), (o["back_offset"], o["s"][0]),
                                                              (o["back_offset"], o["s"][1]), (o["wall_offset"], o["s"][1]))]))
    return out


def write_dxf(path, V, polys, objects, cut, info, openings=()):
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.MM
    doc.header["$MEASUREMENT"] = 1
    for name, color, ltype in (("WALLS", 7, "CONTINUOUS"), ("FURNITURE", 3, "CONTINUOUS"),
                               ("OVERHEAD", 8, "DASHED"), ("WINDOWS", 4, "CONTINUOUS"), ("DIMENSIONS", 1, "CONTINUOUS")):
        doc.layers.add(name, color=color, linetype=ltype)
    msp = doc.modelspace()
    loops = section(V, polys, cut)
    for loop in loops:
        msp.add_lwpolyline((loop * MM).tolist(), close=True, dxfattribs={"layer": "WALLS", "lineweight": 50})
    outlines = object_outlines(objects, V, polys, cut)
    for xy, below in outlines:
        msp.add_lwpolyline((xy * MM).tolist(), close=True,
                           dxfattribs={"layer": "FURNITURE" if below else "OVERHEAD", "ltscale": 20.0})
    windows = window_outlines(openings, cut)
    for xy in windows:
        msp.add_lwpolyline((xy * MM).tolist(), close=True, dxfattribs={"layer": "WINDOWS"})
    allp = np.vstack(loops) * MM
    lo, hi = allp.min(0), allp.max(0)
    off = 600.0
    dim = msp.add_linear_dim(base=(lo[0], lo[1] - off), p1=(lo[0], lo[1]), p2=(hi[0], lo[1]),
                             dimstyle="EZDXF", override={"dimtxt": 150, "dimasz": 120, "dimdec": 0},
                             dxfattribs={"layer": "DIMENSIONS"})
    dim.render()
    dim = msp.add_linear_dim(base=(hi[0] + off, lo[1]), p1=(hi[0], lo[1]), p2=(hi[0], hi[1]), angle=90,
                             dimstyle="EZDXF", override={"dimtxt": 150, "dimasz": 120, "dimdec": 0},
                             dxfattribs={"layer": "DIMENSIONS"})
    dim.render()
    text = (f"План на высоте {info['plan_cut_m'] * MM:.0f} мм над полом. Площадь пола {info['floor_area_m2']:.2f} м2, "
            f"высота {info['height_m'] * MM:.0f} мм, объём {info['volume_m3']:.1f} м3")
    msp.add_text(text, height=150, dxfattribs={"layer": "DIMENSIONS"}).set_placement((lo[0], lo[1] - 2.2 * off))
    doc.saveas(path)
    draw_preview(Path(path).with_suffix(".png"), loops, outlines,
                 text + f"; {(hi[0] - lo[0]):.0f} x {(hi[1] - lo[1]):.0f} мм", windows)
    return len(loops)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir")
    parser.add_argument("--cut", type=float, default=1.0, help="высота среза плана над полом, м")
    args = parser.parse_args()
    mdir = Path(args.mesh_dir)
    V, polys = read_obj_polygons(mdir / "shell.obj")
    # Текст OBJ хранит микроны, а OpenCASCADE проверяет плоскость грани с точностью 1e-7 мм:
    # вершины заново — пересечения плоскостей граней, в double.
    V = exact_vertices(V, polys)
    objects = o3d.io.read_triangle_mesh(str(mdir / "objects.ply"))

    normals = [face_normal_area(V, p) for p in polys]
    floor = min((p for p, (n, _) in zip(polys, normals) if n[2] < -0.99), key=lambda p: V[p, 2].mean())
    ceiling = max((p for p, (n, _) in zip(polys, normals) if n[2] > 0.99), key=lambda p: V[p, 2].mean())
    z0 = float(V[floor, 2].mean())
    info = {
        "floor_z_m": round(z0, 4),
        "height_m": round(float(V[ceiling, 2].mean()) - z0, 4),
        "floor_area_m2": round(face_normal_area(V, floor)[1], 3),
        "volume_m3": round(polyhedron_volume(V, polys), 3),
        "faces": len(polys),
    }
    bodies = [("room", V, polys)]
    for f in sorted((mdir / "details").glob("*.obj")) if (mdir / "details").exists() else []:
        BV, bp = read_obj_polygons(f)
        bodies.append((f.stem, BV, bp))
    volumes, solids, faces, cut = write_step(mdir / "room.step", bodies, cut_from_room=("skirting", "crown"))
    info.update({"step_bodies": [b[0] for b in bodies], "step_solids_read": solids, "step_faces": faces,
                 "step_volumes_m3": {k: round(v, 4) for k, v in volumes.items()}, "details_cut_from_room": cut,
                 "step_volume_matches": bool(abs(volumes["room"] - info["volume_m3"]) < 1e-3 * info["volume_m3"])})
    shell_report = mdir / "shell_report.json"
    openings = json.loads(shell_report.read_text())["openings"] if shell_report.exists() else []
    info["openings"] = [{"width_m": o["width_m"], "height_m": o["height_m"], "depth_m": o["depth_m"],
                         "sill_m": round(o["z"][0] - z0, 3)} for o in openings]
    info["plan_cut_m"] = args.cut
    info["plan_loops"] = write_dxf(mdir / "plan.dxf", V, polys, objects, z0 + args.cut, info, openings)
    (mdir / "cad_report.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[cad] {mdir / 'room.step'}: тела {', '.join(b[0] for b in bodies)} (прочитано {solids}), граней {faces}; "
          f"объём комнаты {volumes['room']:.2f} м³ ({'совпадает' if info['step_volume_matches'] else 'НЕ совпадает'} "
          f"с моделью), воздух без деталей {volumes['room_air']:.2f} м³; plan.dxf: срез {args.cut} м, "
          f"площадь пола {info['floor_area_m2']:.2f} м², высота {info['height_m']:.3f} м, проёмов {len(openings)}")


if __name__ == "__main__":
    main()
