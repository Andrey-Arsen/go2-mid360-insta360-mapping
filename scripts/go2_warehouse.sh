#!/usr/bin/env bash
# Go2 с надстройкой (Orin, Mid-360, Insta360) в Isaac Sim: склад, ходьба политикой IsaacLegs,
# наши датчики в ROS 2, RViz; рулить WASD в этом терминале.
#   scripts/go2_warehouse.sh                          # Full Warehouse
#   scripts/go2_warehouse.sh warehouse                # другое окружение (см. run_go2_isaac.py --help)
#   scripts/go2_warehouse.sh full_warehouse --spawn 2 -3 90   # где поставить робота: X Y угол
#   scripts/go2_warehouse.sh full_warehouse --record sklad01  # записать поездку -> data/drives/sklad01
#     потом без участия человека: scripts/process_drive.sh sklad01 (перерисовка + модель)
# Нужно один раз: robot/go2_mapping (scripts/build_go2_mapping.py), сборка IsaacLegs
# (colcon build в third_party/IsaacLegs) и torch для контроллера (third_party/pydeps).
# Журналы — out/go2_logs/. Выход: Ctrl+C здесь (закроет всё) или закрыть окно Isaac Sim.
# WITH_RVIZ=0 — без RViz; аргументы после окружения уходят в run_go2_isaac.py (--image-size, --headless ...).
set -e
set -m  # каждый фоновый процесс — своя группа: Ctrl+C здесь их не задевает, гасим группой целиком
cd "$(dirname "$0")/.."
ROOT=$PWD
ENV_NAME=${1:-full_warehouse}
[ $# -gt 0 ] && shift
for need in robot/go2_mapping/go2_mapping.usda third_party/IsaacLegs/install/setup.bash third_party/pydeps/torch; do
    [ -e "$need" ] || { echo "нет $need — см. шапку этого скрипта"; exit 1; }
done
source scripts/ros_env.sh
source third_party/IsaacLegs/install/setup.bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/third_party/pydeps"  # torch для контроллера, в конец — системные пакеты главнее
LOG=out/go2_logs
mkdir -p "$LOG"
RECORD=""
prev=""
for a in "$@"; do [ "$prev" = "--record" ] && RECORD=$a; prev=$a; done

# python.sh Isaac запускает python дочерним процессом — поэтому гасим всю группу задания.
# disown — чтобы bash не печатал «Terminated» про свои фоновые задания при выходе.
~/isaacsim/python.sh scripts/run_go2_isaac.py --env "$ENV_NAME" "$@" > "$LOG/isaac.log" 2>&1 &
ISAAC_PID=$!
disown
CTRL_PID=""
RVIZ_PID=""
cleanup() {
    set +e  # уже закрытый процесс — не повод бросить уборку на полпути
    # Isaac — первым и мягко: SIGTERM всей группе (python.sh умирает сразу, а сам Isaac ловит
    # сигнал и дописывает запись поездки), ждём, пока из группы не выйдут все.
    if kill -0 -- "-$ISAAC_PID" 2>/dev/null; then
        [ -n "$RECORD" ] && echo "сохраняю запись поездки и закрываю Isaac Sim..."
        kill -TERM -- "-$ISAAC_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 -- "-$ISAAC_PID" 2>/dev/null || break; sleep 1; done
    fi
    for pid in $CTRL_PID $RVIZ_PID; do kill -- "-$pid" 2>/dev/null; done
    sleep 1
    for pid in $CTRL_PID $RVIZ_PID $ISAAC_PID; do kill -9 -- "-$pid" 2>/dev/null; done
    if [ -n "$RECORD" ]; then
        grep "^\[go2\] поездка записана" "$LOG/isaac.log" \
            || echo "!! не вижу, что поездка записана, — проверьте $LOG/isaac.log и data/drives/$RECORD"
    fi
}
trap cleanup EXIT
echo "Isaac Sim грузится (ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}); склад в первый раз скачивается несколько минут..."
until grep -q "\[go2\] готово" "$LOG/isaac.log"; do
    if ! kill -0 $ISAAC_PID 2>/dev/null || grep -q "^\[go2\] нет\|^\[go2\] не\|Traceback" "$LOG/isaac.log"; then
        echo "Isaac Sim не запустился:"; grep -E "^\[go2\]|Traceback|Error:" "$LOG/isaac.log" | tail -5
        echo "весь журнал: $LOG/isaac.log"; exit 1
    fi
    sleep 2
done
grep "^\[go2\]\|^\[ros2_sensors\]\|^\[mid360_driver\]" "$LOG/isaac.log"

ros2 launch fullbody_controller go2.launch.py > "$LOG/controller.log" 2>&1 &
CTRL_PID=$!
disown
if [ "${WITH_RVIZ:-1}" != 0 ]; then  # WITH_RVIZ=0 scripts/go2_warehouse.sh — без RViz
    rviz2 -d rviz/go2.rviz --ros-args -p use_sim_time:=true > "$LOG/rviz.log" 2>&1 &
    RVIZ_PID=$!
    disown
fi
echo "Контроллер ходьбы${RVIZ_PID:+ и RViz} запущен${RVIZ_PID:+ы}. Управление — здесь (окно терминала должно быть активным):"
if [ -n "$RECORD" ]; then
    echo "ИДЁТ ЗАПИСЬ. Сначала постойте 3 секунды (по покою выставится IMU), потом езжайте."
    echo "Объезжайте всё, что должно попасть в модель; Ctrl+C — конец записи."
fi
python3 scripts/teleop_wasd.py --watch-pid $ISAAC_PID
