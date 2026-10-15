"""Детали комнаты поверх плоской оболочки fit_shell.py: проёмы и погонажные профили.

Проём (окно, ниша). Две параллельные плоскости стены на 2–40 см друг от друга: если
точки дальней лежат в прямоугольнике, который ближняя обступает со всех четырёх
сторон, дальняя — стекло или дно ниши. Стеной оболочки остаётся ближняя (для робота
окно и так стена), проём записывается прямоугольником: вдоль стены, по высоте, глубина.
Если наоборот, обступают ближнюю, это выступ (пилястра) — его не трогаем.

Профиль у пола и у потолка (плинтус, карниз). Точки всех стен у стыка со стеной
переводятся в сечение: d — от стены в комнату, h — от пола вверх (от потолка вниз);
углы комнаты (0.4 м с каждого конца стены) не берутся. Шум лидара ±2 см сравним с
самой деталью, но профиль одинаков вдоль всех стен — медиана по лучам из точки
(D, D) усредняет десятки тысяч точек. Медианная кривая упрощается до ломаной (Дуглас–
Пекер, 4 мм) и концами ложится на стену (d = 0) и пол (h = 0). Углы не притягиваются
к 45/90°: в вогнутом углу медиана смешивает две грани, излом немного скруглён — это
честный усреднённый замер.
Профиль протягивается по контуру пола (потолка): при сдвиге d контур смещается внутрь
с углами «на ус», соседние слои соединяются плоскими трапециями — кольцо-тело.
"""

import numpy as np


def cross2(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def wall_axes(n):
    """Вдоль стены (горизонталь) и вверх для вертикальной плоскости с нормалью n."""
    t = np.array([-n[1], n[0], 0.0])
    return t / np.linalg.norm(t), np.array([0.0, 0.0, 1.0])


def strip_coverage(s, z, s0, s1, z0, z1, cell=0.05, need=2):
    """Доля клеток cell x cell в полосе [s0,s1]x[z0,z1], где есть точки: рама проёма —
    сплошная стена, а не тонкая линия точек (скругление угла и т. п.)."""
    m = (s >= s0) & (s <= s1) & (z >= z0) & (z <= z1)
    ns, nz = max(1, int((s1 - s0) / cell)), max(1, int((z1 - z0) / cell))
    counts = np.histogram2d(s[m], z[m], bins=(ns, nz), range=((s0, s1), (z0, z1)))[0]
    return float((counts >= need).mean())


def hole_bounds(s, z, rect, cell=0.05, pad=0.6):
    """Края проёма по дыре в точках стены: края стекла лидар видит плохо (под углом), а
    обрез стены — чётко. Пустые клетки вокруг прямоугольника стекла, связные с его
    центром; если дыра не замкнута рамой — остаётся прямоугольник стекла."""
    from scipy import ndimage
    s0, s1, z0, z1 = rect
    lo_s, hi_s, lo_z, hi_z = s0 - pad, s1 + pad, z0 - pad, z1 + pad
    ns, nz = int((hi_s - lo_s) / cell), int((hi_z - lo_z) / cell)
    m = (s >= lo_s) & (s < hi_s) & (z >= lo_z) & (z < hi_z)
    raw = np.histogram2d(s[m], z[m], bins=(ns, nz), range=((lo_s, hi_s), (lo_z, hi_z)))[0] > 0
    occ = ndimage.binary_closing(raw, iterations=1)  # редкие пропуски в стене — не дыра
    lab, _ = ndimage.label(~occ)
    c = lab[ns // 2, nz // 2]
    if c == 0:
        return rect
    region = ndimage.binary_dilation(lab == c) & ~raw  # вернуть клетку, съеденную закрытием
    if region[0, :].any() or region[-1, :].any() or region[:, 0].any() or region[:, -1].any():
        return rect  # дыра уходит за край — рамы нет
    i, k = np.nonzero(region)
    return lo_s + i.min() * cell, lo_s + (i.max() + 1) * cell, lo_z + k.min() * cell, lo_z + (k.max() + 1) * cell


def find_recesses(groups, P, full=None, min_depth=0.02, max_depth=0.40, strip=0.2, min_cover=0.5, min_area=0.3,
                  max_share=0.85, voxel=0.03):
    """groups: [n, d, idx] по точкам P (нормаль к датчику). full — облако без прореживания:
    по нему ищется край проёма (в вокселе 3 см на краю окна усредняются откос, стекло и
    стена, и среднее ложится на плоскость стены внутри проёма). Возвращает (индексы,
    которые убрать из оболочки, список проёмов)."""
    full = P if full is None else full
    drop, openings = set(), []
    cos5 = np.cos(np.radians(5))
    for i, (ni, di, ii) in enumerate(groups):
        if abs(ni[2]) > 0.1:
            continue
        for j, (nj, dj, ij) in enumerate(groups):
            if j == i or nj @ ni < cos5 or abs(nj[2]) > 0.1:
                continue
            depth = dj - di  # j дальше от комнаты, чем i
            if not min_depth < depth < max_depth:
                continue
            t, up = wall_axes(ni)
            sj, zj = P[ij] @ t, P[ij] @ up
            if len(ij) * voxel ** 2 < min_area:
                continue
            s0, s1 = np.percentile(sj, [1, 99])
            z0, z1 = np.percentile(zj, [1, 99])
            si, zi = P[ii] @ t, P[ii] @ up
            # Проём — часть стены: заметно меньше её (иначе это сама стена за выступом).
            wall_area = np.ptp(np.percentile(si, [1, 99])) * np.ptp(np.percentile(zi, [1, 99]))
            if (s1 - s0) * (z1 - z0) > max_share * wall_area:
                continue
            cover = [
                strip_coverage(si, zi, s0 - strip, s0, z0, z1),
                strip_coverage(si, zi, s1, s1 + strip, z0, z1),
                strip_coverage(si, zi, s0, s1, z0 - strip, z0),
                strip_coverage(si, zi, s0, s1, z1, z1 + strip),
            ]
            if min(cover) >= min_cover:
                # Край дыры — по всем точкам у плоскости стены (у самого края нормали
                # смешанные, и в группу RANSAC эти точки не попали), но не по шумным
                # точкам стекла у краёв, где лидар видит его под острым углом.
                dist_wall, dist_back = np.abs(full @ ni + di), np.abs(full @ ni + dj)
                near = (dist_wall < 0.025) & (dist_wall < dist_back)
                s0, s1, z0, z1 = hole_bounds(full[near] @ t, full[near] @ up, (s0, s1, z0, z1))
                drop.add(j)
                openings.append({"wall_normal": ni.tolist(), "wall_offset": float(di), "back_offset": float(dj),
                                 "depth_m": round(float(depth), 3), "s": [float(s0), float(s1)], "z": [float(z0), float(z1)],
                                 "width_m": round(float(s1 - s0), 3), "height_m": round(float(z1 - z0), 3),
                                 "coverage": [round(c, 2) for c in cover]})
    return drop, openings


def opening_box(o):
    """8 вершин и 6 граней (нормали наружу) ниши проёма между стеной и стеклом."""
    n = np.array(o["wall_normal"])
    t, up = wall_axes(n)
    V = []
    for w in (o["wall_offset"], o["back_offset"]):
        for s in o["s"]:
            for z in o["z"]:
                V.append(s * t + z * up - w * n)
    V = np.array(V)
    faces = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    return V, orient_outward(V, faces)


def polygon_signed_volume(V, polys):
    vol = 0.0
    for poly in polys:
        for k in range(1, len(poly) - 1):
            a, b, c = V[poly[0]], V[poly[k]], V[poly[k + 1]]
            vol += np.dot(a, np.cross(b, c)) / 6.0
    return vol


def orient_outward(V, polys):
    return polys if polygon_signed_volume(V, polys) > 0 else [p[::-1] for p in polys]


def section_points(C, loop, z0, sign, depth, margin, low=-0.05):
    """Точки облака у стыка: (d, h) по всем рёбрам контура длиннее 2*margin + 0.3 м."""
    L = loop[:, :2]
    ccw = np.sum(cross2(L, np.roll(L, -1, 0))) > 0
    out = []
    for i in range(len(L)):
        a, b = L[i], L[(i + 1) % len(L)]
        t = b - a
        length = np.linalg.norm(t)
        if length < 2 * margin + 0.3:
            continue
        t /= length
        n = np.array([-t[1], t[0]]) if ccw else np.array([t[1], -t[0]])
        rel = C[:, :2] - a
        s, d, h = rel @ t, rel @ n, (C[:, 2] - z0) * sign
        m = (s > margin) & (s < length - margin) & (d > low) & (d < depth) & (h > low) & (h < depth)
        out.append(np.c_[d[m], h[m]])
    return np.vstack(out) if out else np.zeros((0, 2))


def median_profile(Q, depth, step_deg=0.5, min_count=30):
    """Медиана расстояния по лучам из (depth, depth): от стены (вверху) до пола."""
    c = np.array([depth, depth])
    ang = np.degrees(np.arctan2(Q[:, 1] - c[1], Q[:, 0] - c[0]))
    r = np.hypot(Q[:, 0] - c[0], Q[:, 1] - c[1])
    prof = []
    for a0 in np.arange(-180.0, -90.0, step_deg):
        m = (ang >= a0) & (ang < a0 + step_deg)
        if m.sum() >= min_count:
            a = np.radians(a0 + step_deg / 2)
            prof.append(c + np.median(r[m]) * np.array([np.cos(a), np.sin(a)]))
    return np.array(prof)


def intersect(l1, l2):
    """Прямые (точка, направление) -> точка пересечения или None."""
    (p, u), (q, v) = l1, l2
    den = cross2(u, v)
    if abs(den) < 1e-9:
        return None
    return p + u * cross2(q - p, v) / den


def douglas_peucker(pts, eps):
    if len(pts) < 3:
        return [0, len(pts) - 1]
    ab = pts[-1] - pts[0]
    dist = np.abs(cross2(ab, pts - pts[0])) / max(np.linalg.norm(ab), 1e-12)
    k = int(dist.argmax())
    if dist[k] <= eps:
        return [0, len(pts) - 1]
    left, right = douglas_peucker(pts[:k + 1], eps), douglas_peucker(pts[k:], eps)
    return left[:-1] + [k + r for r in right]


def regular_profile(Q, depth, eps=0.004, attach=0.012, min_area=5e-4):
    """Точки сечения -> многоугольник (d, h): угол (0,0), по полу до профиля, по профилю
    вверх, по стене вниз. Профиль — медианная кривая, упрощённая до ломаной с точностью
    eps; где кривая ближе attach к стене или полу, она на них и лежит. None — деталь
    меньше min_area (острый угол) или точек мало. Вторым — сама медианная кривая."""
    if len(Q) < 1000:
        return None, None
    prof = median_profile(Q, depth)
    if len(prof) < 20:
        return None, prof
    v = prof[douglas_peucker(prof, eps)].copy()
    i = 0  # последняя вершина ещё на стене
    while i + 1 < len(v) and v[i + 1][0] < attach:
        i += 1
    k = len(v) - 1  # первая вершина уже на полу
    while k - 1 > i and v[k - 1][1] < attach:
        k -= 1
    if k - i < 1:
        return None, prof
    v = v[i:k + 1]
    v[0][0], v[-1][1] = 0.0, 0.0
    if np.any(v < -attach):
        return None, prof
    v = np.maximum(v, 0.0)
    poly = np.vstack([[0.0, 0.0], v[::-1]])
    area = 0.5 * abs(np.sum(cross2(poly, np.roll(poly, -1, 0))))
    return (poly if area >= min_area else None), prof


def offset_loop(loop, d):
    """Контур (горизонтальный, xy) сдвигается внутрь на d, углы «на ус»."""
    L = loop[:, :2]
    ccw = np.sum(cross2(L, np.roll(L, -1, 0))) > 0
    t = np.roll(L, -1, 0) - L
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    n = np.c_[-t[:, 1], t[:, 0]] if ccw else np.c_[t[:, 1], -t[:, 0]]
    n_prev = np.roll(n, 1, 0)
    miter = (n_prev + n) / (1.0 + np.sum(n_prev * n, axis=1, keepdims=True))
    return L + d * miter


def loft_ring(loop, z0, sign, poly):
    """Кольцо-тело профиля poly (d, h) вдоль контура: вершины, грани (наружу) и
    треугольники видимой поверхности (не по полу/стене) — для коллайдера."""
    n_loop, n_poly = len(loop), len(poly)
    V = np.array([np.r_[xy, z0 + sign * h] for d, h in poly for xy in offset_loop(loop, d)])

    def vid(j, i):
        return (j % n_poly) * n_loop + (i % n_loop)

    faces, visible = [], []
    for j in range(n_poly):
        on_floor = abs(poly[j][1]) < 1e-9 and abs(poly[(j + 1) % n_poly][1]) < 1e-9
        on_wall = abs(poly[j][0]) < 1e-9 and abs(poly[(j + 1) % n_poly][0]) < 1e-9
        for i in range(n_loop):
            q = [vid(j, i), vid(j, i + 1), vid(j + 1, i + 1), vid(j + 1, i)]
            faces.append(q)
            visible.append(not (on_floor or on_wall))
    faces = orient_outward(V, faces)  # наружу из тела — на видимых гранях это в комнату
    tris = [t for q, vis in zip(faces, visible) if vis for t in ([q[0], q[1], q[2]], [q[0], q[2], q[3]])]
    return V, faces, np.array(tris)
