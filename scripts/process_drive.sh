#!/usr/bin/env bash
# Записанная поездка Go2 -> 3D-модель, без участия человека (можно оставить на ночь).
#   scripts/process_drive.sh sklad01             # перерисовка поездки + модель
#   GS=1 scripts/process_drive.sh sklad01        # + Gaussian splatting (ещё ~30 мин, видеокарта целиком)
#   scripts/process_drive.sh sklad01 --skip-render   # перерисовка уже есть — только обработка
# Поездку записывают так: scripts/go2_warehouse.sh full_warehouse --record sklad01
# Настройки обработки (DEPTH, TRIANGLES, ROOM_SHELL) — см. шапку scripts/pipeline.sh;
# перерисовки: SHUTTER=0.25 (снимок раз в N с), IMAGE_SIZE=1920; гауссиан: CAMERA_BA=0 — без уточнения
# поз камер по снимкам (быстрее, но заметно мутнее).
# Итог:
#   scenes/reconstructed/<имя>_rko.usda   цветной меш + коллайдер — открыть в Isaac Sim
#   scenes/reconstructed/<имя>_gs.usda    гауссианы + коллайдер (с GS=1)
#   out/mesh/<имя>_rko/                   cloud.laz, mesh_colored.ply, room.step, plan.dxf ...
#   out/compare_<имя>_rko.png             оригинал и модель с одних точек
# Журналы шагов — out/logs/<имя>/.
set -e
N=${1:?укажите имя поездки (data/drives/<имя>)}
cd "$(dirname "$0")/.."
LOGS=out/logs/$N
mkdir -p "$LOGS"
T0=$(date +%s)
[ -f "data/drives/$N/motion.npz" ] || { echo "нет записи data/drives/$N — сначала scripts/go2_warehouse.sh <окружение> --record $N"; exit 1; }

if [ "$2" != "--skip-render" ]; then
    echo "== перерисовка поездки с полными датчиками (долго: ~6 кадров/с, т. е. минута поездки ~ 10 мин)"
    ~/isaacsim/python.sh scripts/render_drive.py "$N" --force --shutter-every "${SHUTTER:-0.25}" \
        --image-size "${IMAGE_SIZE:-1920}" > "$LOGS/render.log" 2>&1 || true
    grep -E "^\[render\]" "$LOGS/render.log" | grep -v "кадр/с" | cut -c1-200 || true
    grep -q "^\[render\] готово" "$LOGS/render.log" || {
        echo "!! перерисовка не удалась — журнал $LOGS/render.log:"
        grep -E "^\[|Traceback|Error:" "$LOGS/render.log" | grep -v "Warp CUDA" | tail -12; exit 1; }
fi

scripts/pipeline.sh "$N"

if [ "${GS:-0}" != 0 ]; then
    echo "== Gaussian splatting (около 40 мин: уточнение поз камер по снимкам ~10, обучение ~20; видеокарта занята)"
    .venv/bin/python scripts/session_to_colmap.py "$N" > "$LOGS/colmap.log" 2>&1 || {
        echo "!! набор для обучения не собран — журнал $LOGS/colmap.log:"; tail -5 "$LOGS/colmap.log"; exit 1; }
    grep "^\[colmap\]" "$LOGS/colmap.log" | tail -1 | cut -c1-200
    if [ "${CAMERA_BA:-1}" != 0 ]; then  # позы камер — по самим снимкам (COLMAP): +6 дБ на складе
        .venv/bin/python scripts/refine_cameras.py "$N" > "$LOGS/refine_cameras.log" 2>&1 || {
            echo "!! уточнение поз камер не удалось — журнал $LOGS/refine_cameras.log:"; grep -E "^\[ba\]|Error|error" "$LOGS/refine_cameras.log" | tail -5; exit 1; }
        grep "^\[ba\] готово" "$LOGS/refine_cameras.log" | cut -c1-200
    fi
    rm -f "out/gs/${N}_rko/gaussians.usdz"  # чтобы старый файл не сошёл за новый
    scripts/train_gs.sh "$N" > "$LOGS/train_gs.log" 2>&1 || true
    [ -f "out/gs/${N}_rko/gaussians.usdz" ] || {
        echo "!! обучение не удалось — журнал $LOGS/train_gs.log:"; tail -8 "$LOGS/train_gs.log"; exit 1; }
    .venv/bin/python scripts/gs_scene.py "$N" > "$LOGS/gs_scene.log" 2>&1 || {
        echo "!! сцена с гауссианами не собрана — журнал $LOGS/gs_scene.log:"; tail -5 "$LOGS/gs_scene.log"; exit 1; }
    grep "^\[gs\]" "$LOGS/gs_scene.log" | cut -c1-200
    if .venv/bin/python scripts/eval_views.py "$N" > "$LOGS/eval_views.log" 2>&1; then
        grep "^\[eval\]" "$LOGS/eval_views.log" | cut -c1-200
    else
        echo "   оценка картинки не удалась (журнал $LOGS/eval_views.log) — сама сцена готова"
    fi
fi
echo "== всё: $(( ($(date +%s) - T0) / 60 )) мин. Модель: scenes/reconstructed/${N}_rko.usda$([ "${GS:-0}" != 0 ] && echo ", гауссианы: scenes/reconstructed/${N}_gs.usda")"
