#!/usr/bin/env python3
"""Компактная модель комнаты (пол, стены, потолок) из плоскостей облака — PolyFit.

1. Облако (cloud.ply из build_mesh.py) прореживается, RANSAC по очереди вынимает
   плоскости; точка идёт в плоскость, только если её нормаль с плоскостью согласна.
   Плоскость делится на связные куски.
2. Мебель отсеивается лучом: от куска назад, от датчика, — если там есть меш
   (под столешницей пол, за ножкой пол), это не оболочка комнаты.
3. Параллельные куски ближе --merge-dist сливаются: потолок из-за ошибки высоты
   RKO-LIO лежит слоями в несколько сантиметров.
4. Выравнивание (--snap-deg, 0 — выкл.): почти горизонтальное — строго по гравитации
   (ось z одометрии выставлена по IMU, наклон пола — накопленный дрейф), почти
   вертикальное — строго вертикально, стены почти под прямым углом к главной — под
   прямым. Остальные плоскости (скосы, эркеры) не трогаются. Точки групп сдвигаются
   на выровненные плоскости (см. п. 5), в облаке cloud.ply они остаются как были.
5. PolyFit (Nan, Wonka, ICCV 2017; решатель SCIP) выбирает из граней, на которые эти
   плоскости режут пространство, замкнутый многогранник, лучше всего объясняющий точки.
   Свой шаг refine_planes PolyFit пропускает — он заново подогнал бы плоскости по
   точкам и снял выравнивание; слияние слоёв уже сделано в п. 3.
6. Соседние грани одной плоскости склеиваются в один многоугольник (PolyFit режет пол
   и потолок полосами по линиям других плоскостей), лишние вершины на прямых убираются.
7. Проёмы и профили у пола/потолка — room_details.py: окно — дальняя из двух
   параллельных плоскостей стены, обступленная стеной (стеной остаётся ближняя);
   плинтус и карниз — профиль, усреднённый вдоль всех стен и протянутый по контуру.
8. Всё внутри оболочки и дальше --object-dist от неё и от деталей — предметы: из меша
   вырезаются их треугольники и сглаживаются (коллайдер стола, светильника).

Пишет в папку меша: shell_planes.vg (вход PolyFit, открывается в Mapple),
shell_polyfit.obj (как выдал PolyFit), shell.obj (склеенные многоугольники, нормали
наружу — для CAD), shell.ply (треугольники, нормали внутрь — коллайдер), details/*.obj
(тела плинтуса, карниза, ниш проёмов — для CAD), details.ply (видимая поверхность
плинтуса и карниза — коллайдер), objects.ply (стол, светильник — всё, чего нет
в оболочке), shell_report.json.
Если есть эталон (export_scene_mesh.py), сравнивает длину, ширину и высоту комнаты:
лучи изнутри в обе стороны по каждой оси, медиана по сетке точек.

Запуск: .venv/bin/python scripts/fit_shell.py out/mesh/tilt30_rko
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import open3d as o3d
import polyfit
from scipy.spatial import cKDTree

import room_details as rd
import session_info

ROOT = Path(__file__).resolve().parents[1]


def ransac_plane(X, NX, dist, rng, iters=3000, sample=20_000, batch=256):
    """RANSAC со своим генератором — повторяемо (segment_plane Open3D многопоточный и от
    запуска к запуску даёт разное). Голосуют только точки с согласной нормалью."""
    k = rng.choice(len(X), min(sample, len(X)), replace=False)
    S, NS = X[k], NX[k]
    tri = rng.integers(0, len(S), (iters, 3))
    a = S[tri[:, 0]]
    n = np.cross(S[tri[:, 1]] - a, S[tri[:, 2]] - a)
    norm = np.linalg.norm(n, axis=1)
    n, a = n[norm > 1e-9] / norm[norm > 1e-9, None], a[norm > 1e-9]
    d = -(n * a).sum(1)
    best, bn, bd = -1, None, None
    for i in range(0, len(n), batch):
        nb, db = n[i:i + batch], d[i:i + batch]
        votes = ((np.abs(S @ nb.T + db) < dist) & (np.abs(NS @ nb.T) > 0.8)).sum(0)
        j = int(votes.argmax())
        if votes[j] > best:
            best, bn, bd = votes[j], nb[j], db[j]
    return bn, bd, np.nonzero(np.abs(X @ bn + bd) < dist)[0]


def extract_planes(P, N, dist, min_points, eps):
    """RANSAC раз за разом; кусок — связные точки плоскости с согласной нормалью."""
    rng = np.random.default_rng(0)
    rest = np.arange(len(P))
    pieces = []
    while len(rest) > min_points:
        n, _, inl = ransac_plane(P[rest], N[rest], dist, rng)
        if len(inl) < min_points:
            break
        idx = rest[inl[np.abs(N[rest[inl]] @ n) > 0.8]]
        rest = np.setdiff1d(rest, rest[inl])
        if len(idx) < min_points:
            continue
        lab = np.asarray(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P[idx])).cluster_dbscan(eps, 10))
        pieces += [idx[lab == k] for k in np.unique(lab[lab >= 0]) if (lab == k).sum() >= min_points]
    return pieces


def fit_plane(P, N, idx):
    """Плоскость по точкам (МНК); нормаль — в сторону датчика, как у нормалей облака."""
    c = P[idx].mean(0)
    n = np.linalg.svd(P[idx] - c, full_matrices=False)[2][2]
    n = n * np.sign(np.median(N[idx] @ n))
    return n, -n @ c


def behind_share(P, n, idx, scene, rng, offset=0.08):
    """Доля точек куска, за которыми (против нормали) есть поверхность меша."""
    s = rng.choice(idx, min(400, len(idx)), replace=False)
    rays = np.c_[P[s] - n * offset, np.tile(-n, (len(s), 1))].astype(np.float32)
    return float(np.isfinite(scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()).mean())


def regularize(groups, P, deg):
    """Горизонталь — по гравитации, вертикаль — отвесно, стены — под прямым углом к главной.
    Сдвиг плоскости после поворота — по центру её точек."""
    if deg <= 0:
        return 0
    lim = np.radians(deg)
    walls = []
    for g in groups:
        n = g[0]
        if abs(n[2]) > np.cos(lim):
            g[0] = np.array([0.0, 0.0, np.sign(n[2])])
        elif abs(n[2]) < np.sin(lim):
            g[0] = np.r_[n[:2] / np.linalg.norm(n[:2]), 0.0]
            walls.append(g)
    snapped = 0
    if walls:
        main = max(walls, key=lambda g: len(g[2]))
        phi0 = np.arctan2(main[0][1], main[0][0])
        for g in walls:
            phi = np.arctan2(g[0][1], g[0][0])
            delta = (phi - phi0 + np.pi / 4) % (np.pi / 2) - np.pi / 4  # отклонение от ближайшего прямого угла
            if abs(delta) < lim:
                g[0] = np.array([np.cos(phi - delta), np.sin(phi - delta), 0.0])
                snapped += 1
    for g in groups:
        g[1] = float(-np.mean(P[g[2]] @ g[0]))
    return snapped


def weld(V, polys, tol=1e-4):
    """PolyFit пишет одну и ту же вершину по нескольку раз — сливаем совпадающие."""
    parent = np.arange(len(V))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(V).query_pairs(tol):
        parent[root(i)] = root(j)
    rep = np.array([root(i) for i in range(len(V))])
    out = []
    for poly in polys:
        q = [int(rep[v]) for v in poly]
        q = [v for i, v in enumerate(q) if v != q[i - 1]]  # подряд одинаковые
        if len(q) >= 3:
            out.append(q)
    return out


def open_edges(polys):
    """Рёбра без встречной пары — дыры; у замкнутого многогранника их нет."""
    edges = {(a, b) for poly in polys for a, b in zip(poly, poly[1:] + poly[:1])}
    return sum((b, a) not in edges for a, b in edges)


def merge_coplanar(V, polys, tol=1e-3, straight_m=1e-3):
    """Склеивает соседние грани одной плоскости: общие рёбра (идут навстречу) убираются,
    оставшиеся сцепляются в контур. Группа, у которой выходит не один контур (дырка),
    остаётся как была. Затем убираются вершины, прямые во всех гранях, где они есть."""
    planes = []
    for poly in polys:
        p = V[poly]
        n = np.zeros(3)
        for a, b in zip(p, np.roll(p, -1, 0)):
            n += np.cross(a, b)
        n /= np.linalg.norm(n)
        planes.append((n, -n @ p.mean(0)))
    group = [-1] * len(polys)
    reps = []
    for i, (n, d) in enumerate(planes):
        for k, (rn, rd) in enumerate(reps):
            if n @ rn > 1 - 1e-6 and abs(d - rd) < tol:
                group[i] = k
                break
        else:
            group[i] = len(reps)
            reps.append((n, d))
    out = []
    for k in range(len(reps)):
        members = [polys[i] for i in range(len(polys)) if group[i] == k]
        if len(members) == 1:
            out.append(members[0])
            continue
        edges = {(a, b) for poly in members for a, b in zip(poly, poly[1:] + poly[:1])}
        border = {e for e in edges if (e[1], e[0]) not in edges}
        nxt = {}
        for a, b in border:
            nxt.setdefault(a, []).append(b)
        if any(len(v) != 1 for v in nxt.values()):
            out += members  # касание в вершине — не трогаем
            continue
        loops, left = [], dict((a, v[0]) for a, v in nxt.items())
        while left:
            start = next(iter(left))
            loop, a = [start], left.pop(start)
            while a != start:
                loop.append(a)
                a = left.pop(a)
            loops.append(loop)
        out += loops if len(loops) == 1 else members
    # Вершина нужна, если хоть в одной грани она отходит от прямой через соседей дальше
    # straight_m (PolyFit пишет координаты во float32 — на прямой бывают изломы в доли мм).
    corner = np.zeros(len(V), bool)
    for poly in out:
        for i, v in enumerate(poly):
            a, c = V[poly[i - 1]], V[poly[(i + 1) % len(poly)]]
            ac = c - a
            if np.linalg.norm(np.cross(ac, V[v] - a)) > straight_m * np.linalg.norm(ac):
                corner[v] = True
    return [[v for v in poly if corner[v]] for poly in out]


def exact_vertices(V, polys):
    """Вершина — точное пересечение плоскостей её граней (МНК, если граней больше трёх):
    после float32 PolyFit грани плоские лишь приблизительно, а CAD (OpenCASCADE) требует
    точной плоскости."""
    incident = {}
    for poly in polys:
        p = V[poly]
        n = np.zeros(3)
        for a, b in zip(p, np.roll(p, -1, 0)):
            n += np.cross(a, b)
        n /= np.linalg.norm(n)
        for v in poly:
            incident.setdefault(v, []).append((n, -n @ p.mean(0)))
    out = V.copy()
    for v, planes in incident.items():
        A = np.array([n for n, _ in planes])
        if np.linalg.matrix_rank(A, tol=1e-3) == 3:
            out[v] = np.linalg.lstsq(A, -np.array([d for _, d in planes]), rcond=None)[0]
    return out


def newell(V, poly):
    p = V[poly]
    n = np.zeros(3)
    for a, b in zip(p, np.roll(p, -1, 0)):
        n += np.cross(a, b)
    return n / np.linalg.norm(n)


def write_obj(path, V, polys):
    used = sorted({v for poly in polys for v in poly})
    remap = {v: i + 1 for i, v in enumerate(used)}
    with open(path, "w") as f:
        f.write("# оболочка комнаты: fit_shell.py (PolyFit), метры, z вверх\n")
        for v in used:
            f.write("v {:.12f} {:.12f} {:.12f}\n".format(*V[v]))  # OpenCASCADE ждёт плоские грани до 1e-7 мм
        for poly in polys:
            f.write("f " + " ".join(str(remap[v]) for v in poly) + "\n")


def write_vg(path, P, N, groups):
    """Формат vertex group PolyFit/Mapple (ASCII): точки, нормали, группы с плоскостями."""
    rng = np.random.default_rng(1)
    with open(path, "w") as f:
        f.write(f"num_points: {len(P)}\n")
        np.savetxt(f, P, fmt="%.4f")
        f.write("num_colors: 0\n")
        f.write(f"num_normals: {len(N)}\n")
        np.savetxt(f, N, fmt="%.4f")
        f.write(f"num_groups: {len(groups)}\n")
        for k, (n, d, idx) in enumerate(groups):
            r, g, b = rng.random(3)
            f.write(f"group_type: 0\nnum_group_parameters: 4\ngroup_parameters: {n[0]:.6f} {n[1]:.6f} {n[2]:.6f} {d:.6f}\n"
                    f"group_label: plane_{k}\ngroup_color: {r:.3f} {g:.3f} {b:.3f}\ngroup_num_point: {len(idx)}\n")
            f.write(" ".join(map(str, idx)) + "\nnum_children: 0\n")


def read_obj_polygons(path):
    V, F = [], []
    for line in open(path):
        tok = line.split()
        if tok and tok[0] == "v":
            V.append([float(x) for x in tok[1:4]])
        elif tok and tok[0] == "f":
            F.append([int(x.split("/")[0]) - 1 for x in tok[1:]])
    return np.array(V), F


def cross2(a, b):
    return a[0] * b[1] - a[1] * b[0]


def triangulate(V, poly):
    """Отрезание ушей: грань PolyFit — плоский многоугольник, не обязательно выпуклый."""
    pts = V[poly]
    n = np.zeros(3)
    for a, b in zip(pts, np.roll(pts, -1, 0)):  # нормаль по Ньюэллу — задаёт обход
        n += np.cross(a, b)
    u = np.cross(n, [1, 0, 0] if abs(n[0]) < 0.9 * np.linalg.norm(n) else [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n / np.linalg.norm(n), u)
    xy = np.c_[pts @ u, pts @ v]  # в этих осях обход против часовой
    ids = list(range(len(poly)))
    tris = []
    while len(ids) > 3:
        for k in range(len(ids)):
            i, j, m = ids[k - 1], ids[k], ids[(k + 1) % len(ids)]
            a, b, c = xy[i], xy[j], xy[m]
            if cross2(b - a, c - b) <= 1e-12:
                continue  # вогнутая вершина
            inside = False
            for q in ids:
                if q in (i, j, m):
                    continue
                p = xy[q]
                if cross2(b - a, p - a) >= 0 and cross2(c - b, p - b) >= 0 and cross2(a - c, p - c) >= 0:
                    inside = True
                    break
            if not inside:
                tris.append([poly[i], poly[j], poly[m]])
                ids.pop(k)
                break
        else:
            break  # вырожденная грань — остаток веером
    tris += [[poly[ids[0]], poly[ids[k]], poly[ids[k + 1]]] for k in range(1, len(ids) - 1)]
    return tris


def raycast(mesh, origins, dirs):
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    rays = np.c_[origins, np.broadcast_to(dirs, origins.shape)].astype(np.float32)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()


def room_dimensions(meshes, lo, hi, openings=(), step=0.25):
    """Длина/ширина/высота: лучи в обе стороны из сетки точек внутри комнаты, медиана.
    Лучи, попавшие в проём модели (окно), не считаются — у эталона там стекло глубже стены,
    у модели закрытая стена: сравниваем стену со стеной."""
    out = {name: {} for name in meshes}
    first = next(iter(meshes.values()))
    for axis, label in ((0, "length_x_m"), (1, "width_y_m"), (2, "height_z_m")):
        other = [k for k in range(3) if k != axis]
        # Отступ от краёв: по высоте 0.3 м (над плинтусом, под карнизом), по горизонтали 0.6 м
        # (мимо скосов углов), для высоты комнаты — 1 м от стен.
        margin = {k: (0.3 if k == 2 else 0.6 if axis != 2 else 1.0) for k in other}
        grids = [np.arange(lo[k] + margin[k], hi[k] - margin[k] + 1e-9, step) for k in other]
        g = np.stack(np.meshgrid(*grids, indexing="ij"), -1).reshape(-1, 2)
        origins = np.zeros((len(g), 3))
        origins[:, other] = g
        origins[:, axis] = (lo[axis] + hi[axis]) / 2
        d = np.zeros(3)
        d[axis] = 1
        ok = np.ones(len(origins), bool)
        for sgn in (1, -1):
            hit = origins + raycast(first, origins, sgn * d)[:, None] * sgn * d
            for o in openings:
                n = np.array(o["wall_normal"])
                t, up = rd.wall_axes(n)
                s, z = hit @ t, hit @ up
                near = np.abs(hit @ n + o["wall_offset"]) < 0.15
                inside = (s > o["s"][0] - 0.2) & (s < o["s"][1] + 0.2) & (z > o["z"][0] - 0.2) & (z < o["z"][1] + 0.2)
                ok &= ~(near & inside)
        for name, mesh in meshes.items():
            t = raycast(mesh, origins[ok], d) + raycast(mesh, origins[ok], -d)
            out[name][label] = round(float(np.median(t[np.isfinite(t)])), 4)
    return out


def check_shell(V, polys, holes, P, min_size=1.5, min_inside=0.85):
    """Оболочка годится, если она замкнута, не меньше min_size по всем осям и внутри неё
    (или на её стенках) лежит большая часть облака. Иначе — выход с кодом 3: помещение не
    одна «коробка» (склад без видимой крыши, несколько комнат, улица), коллайдер — сам меш."""
    if holes:
        raise SystemExit(f"[shell] оболочка не замкнута ({holes} открытых рёбер) — не годится")
    size = V.max(0) - V.min(0)
    if size.min() < min_size:
        raise SystemExit(f"[shell] оболочка вырождена: размеры {np.round(size, 2).tolist()} м — не годится")
    tris = np.array([t for poly in polys for t in triangulate(V, poly)])
    solid = o3d.t.geometry.RaycastingScene()
    solid.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(
        o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(tris))))
    pts = o3d.core.Tensor(P[:: max(1, len(P) // 200_000)].astype(np.float32))
    inside = (solid.compute_occupancy(pts).numpy() > 0.5) | (solid.compute_distance(pts).numpy() < 0.15)
    if inside.mean() < min_inside:
        raise SystemExit(f"[shell] в оболочке только {inside.mean():.0%} облака — помещение не одна коробка, не годится")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_dir")
    parser.add_argument("--voxel", type=float, default=0.03)
    parser.add_argument("--ransac-dist", type=float, default=0.04)
    parser.add_argument("--min-points", type=int, default=800, help="меньше точек — не плоскость (при вокселе 3 см ~0.7 м²)")
    parser.add_argument("--merge-dist", type=float, default=0.07, help="сливать параллельные куски ближе, м")
    parser.add_argument("--weights", type=float, nargs=3, default=(0.43, 0.27, 0.3),
                        help="PolyFit: данные, покрытие, сложность")
    parser.add_argument("--snap-deg", type=float, default=3.0, help="выравнивать плоскости, отклонённые меньше, град; 0 — нет")
    parser.add_argument("--object-dist", type=float, default=0.06, help="предметы — дальше этого от оболочки, м")
    parser.add_argument("--profile-depth", type=float, default=0.35, help="зона профиля у пола/потолка от угла, м")
    parser.add_argument("--no-details", action="store_true", help="без проёмов и плинтусов — только PolyFit")
    parser.add_argument("--smooth-iters", type=int, default=10, help="сглаживание предметов (Тобин), итераций; 0 — нет")
    args = parser.parse_args()

    mdir = Path(args.mesh_dir)
    t0 = time.time()
    pcd = o3d.io.read_point_cloud(str(mdir / "cloud.ply")).voxel_down_sample(args.voxel)
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    colored = mdir / "mesh_colored.ply"
    mesh = o3d.io.read_triangle_mesh(str(colored if colored.exists() else mdir / "mesh.ply"))
    if len(mesh.triangles) > 500_000:
        mesh = mesh.simplify_quadric_decimation(500_000)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    pieces = extract_planes(P, N, args.ransac_dist, args.min_points, 3.5 * args.voxel)
    rng = np.random.default_rng(0)
    cos5 = np.cos(np.radians(5))
    merged, furniture = [], []  # [n, d, idx]
    for idx in sorted(pieces, key=len, reverse=True):
        n, d = fit_plane(P, N, idx)
        if behind_share(P, n, idx, scene, rng) > 0.5:
            furniture.append((n, d, idx))
            continue
        for g in merged:
            if g[0] @ n > cos5 and abs(g[1] - d) < args.merge_dist:
                g[2] = np.r_[g[2], idx]
                g[0], g[1] = fit_plane(P, N, g[2])
                break
        else:
            merged.append([n, d, idx])
    tilt = {i: np.degrees(np.arccos(min(1.0, abs(g[0][2])))) for i, g in enumerate(merged) if abs(g[0][2]) > 0.95}
    fitted = [(g[0].copy(), g[1]) for g in merged]
    snapped = regularize(merged, P, args.snap_deg)
    openings = []
    if not args.no_details:
        # Окно/ниша: дальняя из двух параллельных плоскостей стены — не стена, а проём.
        cloud_full = np.asarray(o3d.io.read_point_cloud(str(mdir / "cloud.ply")).points)
        drop, openings = rd.find_recesses(merged, P, full=cloud_full, voxel=args.voxel)
        merged = [g for i, g in enumerate(merged) if i not in drop]
        fitted = [f for i, f in enumerate(fitted) if i not in drop]
    # PolyFit сам перефитывает плоскость по точкам группы (collect_valid_planes), поэтому
    # выравнивание переносим на точки: каждая встаёт на выровненную плоскость с тем же
    # отступом, что был от подогнанной, — разброс (толщина слоя) сохраняется.
    Q = P.copy()
    for (n0, d0), (n, d, idx) in zip(fitted, merged):
        resid = P[idx] @ n0 + d0
        Q[idx] = P[idx] - (P[idx] @ n + d)[:, None] * n + resid[:, None] * n

    # Вход PolyFit — только точки оболочки, с новой нумерацией.
    used = np.concatenate([g[2] for g in merged])
    remap = np.full(len(P), -1)
    remap[used] = np.arange(len(used))
    groups = [(n, d, remap[idx]) for n, d, idx in merged]
    vg = mdir / "shell_planes.vg"
    write_vg(vg, Q[used], N[used], groups)

    t1 = time.time()
    polyfit.initialize()
    pset = polyfit.read_point_set(str(vg))
    hypothesis = polyfit.HypothesisGenerator(pset)  # без refine_planes — см. п. 5 описания
    result = hypothesis.generate()
    if not result:
        raise RuntimeError("PolyFit не нарезал грани — проверьте плоскости в shell_planes.vg (Mapple)")
    hypothesis.compute_confidences(result, False)
    polyfit.FaceSelection(pset, result).optimize(hypothesis, polyfit.SCIP, *args.weights)
    if result.size_of_facets() == 0:
        raise RuntimeError("PolyFit не собрал многогранник — попробуйте другие --weights")
    raw = mdir / "shell_polyfit.obj"
    polyfit.save_mesh(str(raw), result)
    t_fit = time.time() - t1

    V, raw_polys = read_obj_polygons(raw)
    polys = merge_coplanar(V, weld(V, raw_polys))
    V = exact_vertices(V, polys)
    holes = open_edges(polys)
    check_shell(V, polys, holes, P)
    write_obj(mdir / "shell.obj", V, polys)
    # В shell.obj нормали наружу (твёрдое тело для CAD), в shell.ply — внутрь комнаты:
    # всё происходит внутри, и PhysX выталкивает тела на лицевую сторону треугольника.
    tris = [t[::-1] for poly in polys for t in triangulate(V, poly)]
    shell = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.array(tris)))
    shell.compute_triangle_normals()
    shell.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mdir / "shell.ply"), shell)

    # Детали: профиль у пола и у потолка (плинтус, карниз), ниши проёмов. Каждая — своё
    # тело в details/*.obj (для CAD); видимая поверхность профилей — в details.ply (коллайдер).
    ddir = mdir / "details"
    if ddir.exists():
        for f in ddir.glob("*.obj"):
            f.unlink()
    ddir.mkdir(exist_ok=True)
    detail_meshes, moldings = [], {}
    if not args.no_details:
        horizontal = [(p, newell(V, p)) for p in polys]
        floor_poly = min((p for p, n in horizontal if n[2] < -0.99), key=lambda p: V[p, 2].mean())
        ceil_poly = max((p for p, n in horizontal if n[2] > 0.99), key=lambda p: V[p, 2].mean())
        for name, face, sign in (("skirting", floor_poly, 1), ("crown", ceil_poly, -1)):
            loop = V[face]
            z0 = float(loop[:, 2].mean())
            sec = rd.section_points(cloud_full, loop, z0, sign, args.profile_depth, 0.4)
            poly, _ = rd.regular_profile(sec, args.profile_depth)
            if poly is None:
                continue
            MV, faces, mtris = rd.loft_ring(loop, z0, sign, poly)
            # Тело для CAD заходит в стену и пол на 0.5 мм: при совпадающих гранях булевы
            # операции OpenCASCADE (вычесть плинтус из воздуха комнаты) неустойчивы.
            CV, cfaces, _ = rd.loft_ring(loop, z0, sign, np.where(poly == 0.0, -0.0005, poly))
            write_obj(ddir / f"{name}.obj", CV, cfaces)
            detail_meshes.append(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(MV), o3d.utility.Vector3iVector(mtris)))
            moldings[name] = {"profile_mm": np.round(poly * 1000).astype(int).tolist(), "points": int(len(sec)),
                              "height_mm": int(round(poly[:, 1].max() * 1000)), "depth_mm": int(round(poly[:, 0].max() * 1000))}
        for k, o in enumerate(openings, 1):
            BV, bfaces = rd.opening_box(o)
            write_obj(ddir / f"opening_{k}.obj", BV, bfaces)
    details = o3d.geometry.TriangleMesh()
    for m in detail_meshes:
        details += m
    if detail_meshes:
        details.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(mdir / "details.ply"), details)
    elif (mdir / "details.ply").exists():
        (mdir / "details.ply").unlink()

    # Предметы: треугольники меша внутри оболочки и дальше --object-dist от неё и от деталей.
    sscene = o3d.t.geometry.RaycastingScene()
    sscene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(shell))
    if detail_meshes:
        sscene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(details))
    tri = np.asarray(mesh.triangles)
    cent = np.asarray(mesh.vertices)[tri].mean(1).astype(np.float32)
    near = sscene.compute_distance(o3d.core.Tensor(cent)).numpy() < args.object_dist
    solid = o3d.t.geometry.RaycastingScene()  # для «внутри»: оболочка с нормалями наружу
    solid.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(np.array([t[::-1] for t in tris])))))
    outside = solid.compute_occupancy(o3d.core.Tensor(cent)).numpy() < 0.5
    objects = o3d.geometry.TriangleMesh(mesh)
    objects.remove_triangles_by_mask(near | outside)
    clusters, sizes, _ = objects.cluster_connected_triangles()
    objects.remove_triangles_by_mask(np.asarray(sizes)[np.asarray(clusters)] < 200)
    objects.remove_unreferenced_vertices()
    # Шум лидара (±2 см) делает поверхность бугристой; Тобин сглаживает, почти не сжимая.
    objects = objects.filter_smooth_taubin(number_of_iterations=args.smooth_iters)
    objects.compute_vertex_normals()
    o3d.io.write_triangle_mesh(str(mdir / "objects.ply"), objects)

    lo, hi = V.min(0), V.max(0)
    report = {
        "planes_shell": [{"normal": np.round(n, 4).tolist(), "offset_m": round(float(d), 4), "points": int(len(i))}
                         for n, d, i in merged],
        "planes_furniture": len(furniture),
        "snap_deg": args.snap_deg,
        "horizontal_tilt_before_snap_deg": [round(float(t), 3) for t in tilt.values()],
        "walls_snapped_to_right_angle": snapped,
        "polyfit_weights": list(args.weights),
        "polyfit_s": round(t_fit, 1),
        "polyfit_faces": len(raw_polys),
        "shell_faces": len(polys),
        "shell_open_edges": holes,
        "shell_triangles": len(tris),
        "shell_bounds_m": [np.round(lo, 3).tolist(), np.round(hi, 3).tolist()],
        "objects_triangles": len(objects.triangles),
        "openings": openings,
        "moldings": moldings,
    }
    # Ровность пола: насколько меш-пол гуляет вокруг плоскости пола оболочки.
    floor = min((g for g in merged if g[0][2] > 0.95), key=lambda g: -g[1] / g[0][2], default=None)  # нижняя, нормаль вверх
    if floor is not None:
        mv = np.asarray(mesh.vertices)
        h = mv @ floor[0] + floor[1]
        on = np.abs(h) < 0.1
        report["floor_plane_z_m"] = round(float(-floor[1] / floor[0][2]), 4)
        report["mesh_floor_std_mm"] = round(float(h[on].std()) * 1000, 1)
        report["mesh_floor_p95_abs_mm"] = round(float(np.percentile(np.abs(h[on]), 95)) * 1000, 1)
    GT = session_info.gt_mesh(session_info.session_of(mdir))
    if GT is not None and GT.exists():
        gt = o3d.io.read_triangle_mesh(str(GT))
        dims = room_dimensions({"shell": shell, "gt": gt}, lo, hi, openings)
        report["dimensions"] = dims
        report["dimension_error_mm"] = {k: round((dims["shell"][k] - dims["gt"][k]) * 1000, 1) for k in dims["gt"]}
        acc = o3d.t.geometry.RaycastingScene()
        acc.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(gt))
        sp = np.asarray(shell.sample_points_uniformly(200_000).points).astype(np.float32)
        e = acc.compute_distance(o3d.core.Tensor(sp)).numpy()
        report["shell_to_gt_median_mm"] = round(float(np.median(e)) * 1000, 1)
        report["shell_to_gt_p90_mm"] = round(float(np.percentile(e, 90)) * 1000, 1)
        if detail_meshes:
            dp = np.asarray(details.sample_points_uniformly(100_000).points).astype(np.float32)
            e = acc.compute_distance(o3d.core.Tensor(dp)).numpy()
            report["details_to_gt_median_mm"] = round(float(np.median(e)) * 1000, 1)
            report["details_to_gt_p90_mm"] = round(float(np.percentile(e, 90)) * 1000, 1)
    report["total_s"] = round(time.time() - t0, 1)
    (mdir / "shell_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[shell] плоскостей оболочки {len(merged)}, мебели {len(furniture)}; PolyFit {t_fit:.1f} с: "
          f"граней {len(polys)}, треугольников {len(tris)}, {'замкнута' if holes == 0 else f'открытых рёбер {holes}'}; "
          f"предметы {len(objects.triangles)} треуг.")
    if "dimension_error_mm" in report:
        print(f"[shell] размеры {dims['shell']} против эталона {dims['gt']}, ошибка мм {report['dimension_error_mm']}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
