#!/usr/bin/env python3
"""Уточняет позы камер по самим снимкам (COLMAP) перед обучением гауссиан.

Гауссианы требуют поз точнее, чем даёт лидар: на sklad03 с эталонными позами — 32.3 дБ на
отложенных кадрах, с позами RKO-LIO даже после refine_trajectory.py — 26.0 дБ (1 см ошибки на
расстоянии метра — это уже ~6 пикселей кадра). Лидарные позы — отличное начало, а последний
сантиметр снимают сами кадры:
1. точки SIFT в каждом кадре (маски — робот и край объектива, как у 3DGRUT);
2. пары кадров — только соседние по нашим позам (до --neighbors ближайших, взгляд в ту же
   сторону): без перебора всех со всеми, минуты вместо часов;
3. триангуляция точек от наших поз, затем пучковая оптимизация поз (внутренние параметры
   камер закреплены);
4. результат подгоняется обратно к лидарным позам подобием по центрам камер — метры и система
   координат те же, что у меша и коллайдера; сдвиги поз печатаются.
Пишет позы в <набор>/sparse/0/images.txt (лидарные — в images_lidar.txt), рабочие файлы — в
<набор>_ba/. На траекторию лидара поправки не переносятся: снимков 4 в секунду, колебания шага
(2–3 Гц) между ними не видны, и меш от этого не лучше (проверено на sklad03).

Проверено на sklad03 (1054 кадра, ~8 мин): ошибка поз камер против эталона 0.73 -> 0.47 см и
0.238° -> 0.031°; гауссианы на отложенных кадрах 26.0 -> 31.8 дБ (на эталонных позах 32.3).

Запуск: .venv/bin/python scripts/refine_cameras.py <сессия> [--data data/sessions/<с>/colmap]
"""

import argparse
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("GLOG_minloglevel", "2")  # COLMAP пишет в журнал каждую мелочь

import numpy as np  # noqa: E402
import pycolmap as pc  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def centers_and_axes(rec, names):
    """Центры камер и оптические оси (мир) по именам кадров."""
    by_name = {im.name: im for im in rec.images.values()}
    C, A = [], []
    for n in names:
        T = by_name[n].cam_from_world()
        R, t = T.rotation.matrix(), T.translation
        C.append(-R.T @ t)
        A.append(R.T @ np.array([0.0, 0.0, 1.0]))
    return np.array(C), np.array(A)


def write_text_model(path, cams, images):
    """cameras.txt / images.txt / points3D.txt (старый текстовый формат COLMAP)."""
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "cameras.txt", "w") as f:
        for cid, (w, h, params) in cams.items():
            f.write(f"{cid} OPENCV_FISHEYE {w} {h} " + " ".join(f"{p:.9g}" for p in params) + "\n")
    with open(path / "images.txt", "w") as f:
        for iid, (q, t, cid, name) in images.items():
            f.write(f"{iid} " + " ".join(f"{v:.12g}" for v in (*q, *t)) + f" {cid} {name}\n\n")
    (path / "points3D.txt").write_text("")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("--data", help="набор COLMAP (session_to_colmap.py); по умолчанию data/sessions/<с>/colmap")
    parser.add_argument("--neighbors", type=int, default=20, help="сколько соседних кадров сопоставлять с каждым")
    parser.add_argument("--max-dist", type=float, default=5.0, help="соседи — не дальше этого, м")
    parser.add_argument("--features", type=int, default=6000, help="точек SIFT на кадр")
    parser.add_argument("--rounds", type=int, default=2, help="кругов: оптимизация -> подгонка к лидару -> чистка")
    parser.add_argument("--keep", action="store_true", help="оставить базу COLMAP и рабочие файлы в <набор>_ba/")
    args = parser.parse_args()
    data = Path(args.data) if args.data else ROOT / "data" / "sessions" / args.session / "colmap"
    sparse = data / "sparse" / "0"
    if not (sparse / "images.txt").exists():
        raise SystemExit(f"нет набора {data} — сначала scripts/session_to_colmap.py {args.session}")
    work = data.parent / (data.name + "_ba")
    shutil.rmtree(work, ignore_errors=True)
    (work / "masks").mkdir(parents=True)
    t0 = time.time()

    lidar_txt = sparse / "images_lidar.txt"
    if lidar_txt.exists():  # повторный запуск — начинаем снова с лидарных поз
        shutil.copyfile(lidar_txt, sparse / "images.txt")
    rec0 = pc.Reconstruction(str(sparse))
    names = sorted(im.name for im in rec0.images.values())
    by_name = {im.name: im for im in rec0.images.values()}

    # 1. Точки SIFT — по камерам (у каждой свои параметры), маска — та же, что у 3DGRUT.
    for n in names:
        mask = data / "images" / (Path(n).with_suffix("").as_posix() + "_mask.png")
        if mask.exists():
            (work / "masks" / Path(n).parent).mkdir(parents=True, exist_ok=True)
            os.symlink(mask.resolve(), work / "masks" / f"{n}.png")
    db_path = work / "database.db"
    eo = pc.FeatureExtractionOptions()
    eo.sift.max_num_features = args.features
    eo.num_threads = -1
    for cid, cam in rec0.cameras.items():
        ro = pc.ImageReaderOptions()
        ro.camera_model = "OPENCV_FISHEYE"
        ro.camera_params = ",".join(f"{p:.9g}" for p in cam.params)
        ro.mask_path = str(work / "masks")
        subset = [n for n in names if by_name[n].camera_id == cid]
        pc.extract_features(db_path, data / "images", image_names=subset, camera_mode=pc.CameraMode.SINGLE,
                            reader_options=ro, extraction_options=eo)
    print(f"[ba] точки SIFT: кадров {len(names)}, {time.time() - t0:.0f} с", flush=True)

    # 2. Пары — соседи по нашим позам с похожим направлением взгляда.
    C, A = centers_and_axes(rec0, names)
    tree = cKDTree(C)
    pairs = set()
    for i in range(len(names)):
        d, j = tree.query(C[i], k=min(len(names), 4 * args.neighbors), distance_upper_bound=args.max_dist)
        cand = [k for dk, k in zip(d, j) if np.isfinite(dk) and k != i and A[i] @ A[k] > 0.0][: args.neighbors]
        pairs.update((min(i, k), max(i, k)) for k in cand)
    (work / "pairs.txt").write_text("".join(f"{names[a]} {names[b]}\n" for a, b in sorted(pairs)))
    po = pc.ImportedPairingOptions()
    po.match_list_path = str(work / "pairs.txt")
    pc.match_image_pairs(db_path, pairing_options=po)
    db = pc.Database.open(str(db_path))
    db_images = {im.name: im for im in db.read_all_images()}
    db_cams = {c.camera_id: c for c in db.read_all_cameras()}
    verified = db.num_verified_image_pairs()
    db.close()
    print(f"[ba] пар {len(pairs)}, с подтверждённой геометрией {verified}; {time.time() - t0:.0f} с", flush=True)

    # 3. Модель с нашими позами и номерами из базы -> триангуляция.
    cams = {cid: (c.width, c.height, list(c.params)) for cid, c in db_cams.items()}
    images = {}
    for n in names:
        T = by_name[n].cam_from_world()
        q = T.rotation.quat  # x, y, z, w
        images[db_images[n].image_id] = ((q[3], q[0], q[1], q[2]), T.translation, db_images[n].camera_id, n)
    write_text_model(work / "known", cams, images)
    rec = pc.Reconstruction(str(work / "known"))
    rec = pc.triangulate_points(rec, db_path, data / "images", work / "tri", clear_points=True)
    print(f"[ba] триангуляция: точек {rec.num_points3D()}, ошибка перепроекции {rec.compute_mean_reprojection_error():.2f} px", flush=True)

    # 4. Пучковая оптимизация -> обратно к лидару подобием -> чистка выбросов; несколько кругов.
    C0, _ = centers_and_axes(rec0, names)
    for r in range(args.rounds):
        ba = pc.BundleAdjustmentOptions()
        ba.refine_focal_length = ba.refine_principal_point = ba.refine_extra_params = False
        pc.bundle_adjustment(rec, ba)
        C1, _ = centers_and_axes(rec, names)
        res = pc.estimate_sim3d_robust(C1, C0)
        if not res:
            raise SystemExit("[ba] не удалось подогнать к лидарным позам")
        sim = res["tgt_from_src"]
        rec.transform(sim)
        removed = pc.ObservationManager(rec).filter_all_points3D(2.0, 1.0)
        err = rec.compute_mean_reprojection_error()
        print(f"[ba] круг {r + 1}: ошибка перепроекции {err:.2f} px, масштаб к лидару {sim.scale:.5f}, "
              f"убрано выбросов {removed}; {time.time() - t0:.0f} с", flush=True)

    # 5. Сдвиги поз и запись в набор (имена кадров — как были; лидарные позы — в images_lidar.txt).
    C1, A1 = centers_and_axes(rec, names)
    _, A0 = centers_and_axes(rec0, names)
    move = np.linalg.norm(C1 - C0, axis=1)
    turn = np.degrees(np.arccos(np.clip(np.einsum("ij,ij->i", A0, A1), -1, 1)))
    if not lidar_txt.exists():
        shutil.copyfile(sparse / "images.txt", lidar_txt)
    by_new = {im.name: im for im in rec.images.values()}
    lines = []
    for im in sorted(rec0.images.values(), key=lambda m: m.image_id):
        T = by_new[im.name].cam_from_world()
        q, t = T.rotation.quat, T.translation
        lines.append(f"{im.image_id} {q[3]:.12g} {q[0]:.12g} {q[1]:.12g} {q[2]:.12g} {t[0]:.12g} {t[1]:.12g} {t[2]:.12g} "
                     f"{im.camera_id} {im.name}\n\n")
    (sparse / "images.txt").write_text("# позы уточнены refine_cameras.py; лидарные — images_lidar.txt\n" + "".join(lines))
    report = {"images": len(names), "pairs": len(pairs), "verified_pairs": verified, "points3D": rec.num_points3D(),
              "reprojection_error_px": round(rec.compute_mean_reprojection_error(), 3),
              "move_cm_median": round(float(np.median(move)) * 100, 2), "move_cm_p95": round(float(np.percentile(move, 95)) * 100, 2),
              "turn_deg_median": round(float(np.median(turn)), 3), "turn_deg_p95": round(float(np.percentile(turn, 95)), 3),
              "seconds": round(time.time() - t0)}
    (work / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    if not args.keep:  # база COLMAP — 1–2 ГБ, нужна только для отладки
        for f in ("database.db", "masks", "known", "tri", "pairs.txt"):
            shutil.rmtree(work / f, ignore_errors=True) if (work / f).is_dir() else (work / f).unlink(missing_ok=True)
    print(f"[ba] готово: позы сдвинуты на {report['move_cm_median']:.2f} см (95%: {report['move_cm_p95']:.2f}), "
          f"повёрнуты на {report['turn_deg_median']:.3f}° (95%: {report['turn_deg_p95']:.3f}°); ошибка перепроекции "
          f"{report['reprojection_error_px']:.2f} px, точек {report['points3D']}; {report['seconds']} с -> {sparse / 'images.txt'}", flush=True)


if __name__ == "__main__":
    main()
