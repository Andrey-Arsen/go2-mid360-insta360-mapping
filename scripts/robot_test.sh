#!/usr/bin/env bash
# Робот с GitHub (IsaacLegs: Go2, G1, G1 29-DOF) с его политикой ходьбы — в нашей модели или
# любом окружении Isaac; рулить WASD в этом терминале (как scripts/go2_warehouse.sh).
#   scripts/robot_test.sh g1 scenes/reconstructed/sklad03_rko.usda        # гуманоид G1 в модели склада
#   scripts/robot_test.sh go2 scenes/reconstructed/sklad03_gs.usda        # Go2 среди гауссиан
#   scripts/robot_test.sh g1_29dof scenes/reconstructed/sklad03_rko.usda --spawn 2 -3 90
#   scripts/robot_test.sh g1 full_warehouse                               # для сравнения — оригинал
# Роботы: go2, g1, g1_29dof. Аргументы после окружения уходят в run_robot_isaac.py
# (--spawn X Y КУРС, --headless, --no-play). Журналы — out/robot_logs/. Выход: Ctrl+C здесь.
set -e
set -m  # каждый фоновый процесс — своя группа (как в go2_warehouse.sh)
cd "$(dirname "$0")/.."
ROOT=$PWD
ROBOT=${1:?укажите робота: go2, g1 или g1_29dof}
ENV_NAME=${2:?укажите окружение: путь к модели (scenes/reconstructed/<имя>_rko.usda) или имя (full_warehouse)}
shift 2
case "$ROBOT" in go2|g1|g1_29dof) ;; *) echo "неизвестный робот $ROBOT: go2, g1 или g1_29dof"; exit 1;; esac
for need in third_party/IsaacLegs/install/setup.bash third_party/pydeps/torch; do
    [ -e "$need" ] || { echo "нет $need — см. установку в README"; exit 1; }
done
source scripts/ros_env.sh
source third_party/IsaacLegs/install/setup.bash
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$ROOT/third_party/pydeps"
LOG=out/robot_logs
mkdir -p "$LOG"

# Контроллер — раньше симуляции: гуманоид падает, если политика включится не сразу (README IsaacLegs).
ros2 launch fullbody_controller "$ROBOT.launch.py" > "$LOG/controller.log" 2>&1 &
CTRL_PID=$!
disown
~/isaacsim/python.sh scripts/run_robot_isaac.py --robot "$ROBOT" --env "$ENV_NAME" "$@" > "$LOG/isaac.log" 2>&1 &
ISAAC_PID=$!
disown
cleanup() {
    set +e
    if kill -0 -- "-$ISAAC_PID" 2>/dev/null; then
        kill -TERM -- "-$ISAAC_PID" 2>/dev/null
        for _ in $(seq 1 60); do kill -0 -- "-$ISAAC_PID" 2>/dev/null || break; sleep 1; done
    fi
    kill -- "-$CTRL_PID" 2>/dev/null
    sleep 1
    for pid in $CTRL_PID $ISAAC_PID; do kill -9 -- "-$pid" 2>/dev/null; done
}
trap cleanup EXIT
echo "Isaac Sim грузится ($ROBOT в $ENV_NAME, ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0})..."
until grep -q "^\[robot\] готово" "$LOG/isaac.log"; do
    if ! kill -0 $ISAAC_PID 2>/dev/null || grep -q "^\[robot\] нет\|^\[robot\] не\|^\[go2\] не\|Traceback" "$LOG/isaac.log"; then
        echo "Isaac Sim не запустился:"; grep -E "^\[robot\]|^\[go2\]|Traceback|Error:" "$LOG/isaac.log" | tail -5
        echo "весь журнал: $LOG/isaac.log"; exit 1
    fi
    sleep 2
done
grep "^\[robot\]" "$LOG/isaac.log"
kill -0 "$CTRL_PID" 2>/dev/null || { echo "контроллер не запустился — журнал $LOG/controller.log"; tail -5 "$LOG/controller.log"; exit 1; }
echo "Робот ходит по политике. Управление — здесь (окно терминала должно быть активным):"
python3 scripts/teleop_wasd.py --watch-pid $ISAAC_PID
