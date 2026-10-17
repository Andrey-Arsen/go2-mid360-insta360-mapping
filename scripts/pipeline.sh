#!/usr/bin/env bash
# Сессия -> цветная 3D-модель в Isaac Sim, одной командой (без участия человека).
#   scripts/pipeline.sh tilt30                 # запись уже есть: bag, RKO-LIO, меш, оценка, цвет, USD
#   scripts/pipeline.sh walk03 --record        # стенд: сначала записать (сцена с наклоном 30°)
#   SCENE=simple_room_mid360_fisheye.usda scripts/pipeline.sh walk04 --record   # стенд, другая сцена
# Сессии робота делает scripts/render_drive.py (обычно всё вместе: scripts/process_drive.sh).
# Настройки: DEPTH=11 — детальнее меш (глубина Poisson, по умолчанию 10); TRIANGLES=2000000 —
# сколько треугольников в цветной модели (по умолчанию 500000); ROOM_SHELL=0 — без оболочки PolyFit,
# ROOM_SHELL=1 — строить всегда (по умолчанию строится, если план меньше 400 м²: на большом складе
# PolyFit считает 10–30 минут и всё равно отказывается; коллайдер тогда — сам меш); RKO_VOXEL=1.0 — воксель
# карты RKO-LIO вручную (по умолчанию — по размеру помещения, см. scripts/rko_config.py);
# REFINE_PASSES=6 — проходов уточнения траектории после RKO-LIO (0 — без него);
# MESH_DENOISE=10, MESH_TAUBIN=10 — сглаживание меша (билатеральное по нормалям: ровные плоскости
# с острыми рёбрами, и Таубин); 0 — выключить.
# Итог: scenes/reconstructed/<сессия>_rko.usda (цветной меш, коллайдер, скрытые слои облака и
#       коллайдера), out/mesh/<сессия>_rko/: cloud.laz (CloudCompare), mesh_colored.ply, room.step,
#       plan.dxf/plan.png (CAD, если есть оболочка), cloud.e57; out/compare_<сессия>_rko.png.
# Журналы шагов — out/logs/<сессия>/. Гауссианы — scripts/process_drive.sh с GS=1 или вручную:
#   .venv/bin/python scripts/session_to_colmap.py <с> && .venv/bin/python scripts/refine_cameras.py <с>
#   scripts/train_gs.sh <с>
#   .venv/bin/python scripts/gs_scene.py <с> && .venv/bin/python scripts/eval_views.py <с>
set -e
S=${1:?укажите имя сессии}
cd "$(dirname "$0")/.."
ROOT=$PWD
source /opt/ros/humble/setup.bash
LOGS=out/logs/$S
mkdir -p "$LOGS" out/rko
M=out/mesh/${S}_rko

# run ИМЯ ФИЛЬТР команда... — шаг с полным журналом; на экран — строки по фильтру.
run() {
    local name=$1 filter=$2
    shift 2
    if ! "$@" > "$LOGS/$name.log" 2>&1; then
        echo "!! шаг «$name» не удался. Конец журнала $LOGS/$name.log:"
        tail -15 "$LOGS/$name.log"
        exit 1
    fi
    grep -E "$filter" "$LOGS/$name.log" | cut -c1-200 || true
}
# run_isaac ИМЯ ФИЛЬТР ПРИЗНАК_УСПЕХА команда... — скрипты Isaac выходят с кодом 0 и при ошибке,
# поэтому успех — по итоговой строке.
run_isaac() {
    local name=$1 filter=$2 ok=$3
    shift 3
    "$@" > "$LOGS/$name.log" 2>&1 || true
    if ! grep -qE "$ok" "$LOGS/$name.log"; then
        echo "!! шаг «$name» не удался. Журнал $LOGS/$name.log, последнее важное:"
        grep -E "^\[|Traceback|Error:|Exception" "$LOGS/$name.log" | grep -v "Warp CUDA" | tail -12
        exit 1
    fi
    grep -E "$filter" "$LOGS/$name.log" | cut -c1-200 || true
}

if [ "$2" = "--record" ]; then
    echo "== запись в Isaac Sim (~5 мин)"
    run_isaac record '^\[record\] готово' '^\[record\] готово' \
        ~/isaacsim/python.sh scripts/record_session.py --name "$S" --scene "${SCENE:-simple_room_handheld_tilt30.usda}"
fi
[ -f "data/sessions/$S/lidar.npz" ] || { echo "нет сессии data/sessions/$S"; exit 1; }
GT=$(.venv/bin/python -c "import sys; sys.path.insert(0, 'scripts'); import session_info as s; g = s.gt_mesh('$S'); print(g or '')")

echo "== rosbag"
run bag '^\[bag\]' python3 scripts/session_to_bag.py "$S" --images

echo "== RKO-LIO (офлайн)"
run rko_config '^\[rko\]' .venv/bin/python scripts/rko_config.py "$S" "$LOGS/rko_lio.yaml"
run rko_lio 'x^' ros2 launch rko_lio odometry.launch.py mode:=offline bag_path:="data/sessions/$S/bag" \
    config_file:="$ROOT/$LOGS/rko_lio.yaml" dump_results:=true results_dir:="$ROOT/out/rko" run_name:="$S"
# RKO-LIO нумерует и папку, и файл (<сессия>_N/<сессия>_tum_N.txt) — берём самый свежий.
TUM=$(ls -t out/rko/"$S"_*/"$S"_tum_*.txt 2>/dev/null | grep -v "refined\|_level" | head -1)
[ -n "$TUM" ] || { echo "!! RKO-LIO не выдал траекторию — журнал $LOGS/rko_lio.log"; exit 1; }
# Низ по силе тяжести за всю запись (без покоя в начале карта иначе наклонена на ~1°).
run level '^\[level\]' .venv/bin/python scripts/level_trajectory.py "$S" "$TUM"
TUM=${TUM%.txt}_level.txt
# Скан к карте из других проходов: убирает «двойные стены» и дрожь поз (ошибка 2.8 -> 1.1 см на складе).
if [ "${REFINE_PASSES:-6}" != 0 ]; then
    run refine '^\[refine\] проход '"${REFINE_PASSES:-6}"':' .venv/bin/python scripts/refine_trajectory.py "$S" "$TUM" --iters "${REFINE_PASSES:-6}"
    TUM=${TUM%.txt}_ref.txt
fi
run_isaac eval_odometry 'walked|ate_rmse_m|final_error_m' 'ate_rmse_m' \
    ~/isaacsim/python.sh scripts/eval_odometry.py "$S" "$TUM"

echo "== меш и сравнение с эталоном"
run build_mesh 'mesh_triangles|floor_filled' .venv/bin/python scripts/build_mesh.py "$S" --tum "$TUM" --depth "${DEPTH:-10}" \
    --denoise "${MESH_DENOISE:-10}" --taubin "${MESH_TAUBIN:-10}"
if [ -n "$GT" ]; then
    [ -f "$GT" ] || run_isaac export_gt '^\[gt\]' '^\[gt\] .*треугольников' \
        ~/isaacsim/python.sh scripts/export_scene_mesh.py --session "$S"
    run eval_mesh 'median|within_5cm|fscore|^\[eval\]' .venv/bin/python scripts/eval_mesh.py "$M"
else
    echo "   эталона нет (сессия без окружения) — оценка точности пропущена"
fi

echo "== цвет и облако (LAZ, E57, USD)"
rm -f "$M/mesh_colored.ply"
run colorize '^\[color\]' .venv/bin/python scripts/colorize_mesh.py "$M" "$S" --tum "$TUM" --triangles "${TRIANGLES:-500000}"
run cloud '^\[cloud\]' .venv/bin/python scripts/export_cloud.py "$M" "$S" --tum "$TUM"
run e57 '^\[e57\]' .venv/bin/python scripts/export_e57.py "$M"

echo "== оболочка помещения (PolyFit), STEP и план DXF"
rm -rf "$M/shell.ply" "$M/shell.obj" "$M/objects.ply" "$M/details.ply" "$M/details" "$M/room.step" "$M/plan.dxf" "$M/plan.png"
PLAN_M2=$(.venv/bin/python -c "import json; b = json.load(open('$M/report.json'))['bounds_m']; print(round((b[1][0] - b[0][0]) * (b[1][1] - b[0][1])))")
SHELL_MODE=${ROOM_SHELL:-auto}
if [ "$SHELL_MODE" = auto ] && [ "$PLAN_M2" -gt 400 ]; then
    echo "   пропущено: план ${PLAN_M2} м² — не одна комната (ROOM_SHELL=1 — строить всё равно); коллайдер — сам меш"
elif [ "$SHELL_MODE" != 0 ]; then
    if .venv/bin/python scripts/fit_shell.py "$M" > "$LOGS/fit_shell.log" 2>&1; then
        grep -E '^\[shell\]' "$LOGS/fit_shell.log" | cut -c1-200 || true
        run export_cad '^\[cad\]' .venv/bin/python scripts/export_cad.py "$M"
    else
        rm -f "$M/shell.ply" "$M/objects.ply" "$M/details.ply"
        echo "   оболочка не годится: $(grep -m1 '^\[shell\]' "$LOGS/fit_shell.log" | cut -c1-150)"
        echo "   коллайдер будет из самого меша (журнал $LOGS/fit_shell.log)"
    fi
else
    echo "   пропущено (ROOM_SHELL=0) — коллайдер будет из самого меша"
fi

echo "== USD для Isaac Sim и сравнение с оригиналом"
run usd '^\[usd\]' .venv/bin/python scripts/mesh_to_usd.py "$M"
run_isaac compare '^\[compare\]' '^\[compare\]' \
    ~/isaacsim/python.sh scripts/render_compare.py "$ROOT/scenes/reconstructed/${S}_rko.usda" --session "$S"
echo "== готово: scenes/reconstructed/${S}_rko.usda (открыть в Isaac Sim по полному пути), $M/"
