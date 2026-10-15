#!/usr/bin/env bash
# Проигрывает записанную сессию: RKO-LIO считает одометрию живьём, RViz показывает
# скан, карту RKO-LIO, его путь (оранжевый) и эталон (зелёный).
#   scripts/play_session.sh walk01 [скорость, по умолчанию 1.0]   # RViz не закрывается после конца записи
# Сначала соберите bag: python3 scripts/session_to_bag.py walk01 --images
set -e
SESSION=${1:?укажите сессию, например walk01}
RATE=${2:-1.0}
cd "$(dirname "$0")/.."
source scripts/ros_env.sh
# Свой домен: иначе проигрывание смешается с живой симуляцией (run_ros2.py тоже пишет /livox/lidar).
export ROS_DOMAIN_ID=${PLAY_DOMAIN:-42}
BAG=data/sessions/$SESSION/bag
[ -d "$BAG" ] || { echo "нет $BAG — сначала python3 scripts/session_to_bag.py $SESSION"; exit 1; }

CFG=$(mktemp --suffix .yaml)
cat config/rko_lio_sim.yaml > "$CFG"
echo "use_sim_time: true" >> "$CFG"  # время из bag (--clock)
trap 'rm -f "$CFG"; kill 0' EXIT

ros2 launch rko_lio odometry.launch.py mode:=online autodetect:=false config_file:="$CFG" &
rviz2 -d rviz/session.rviz --ros-args -p use_sim_time:=true &
RVIZ=$!
sleep 5
ros2 bag play "$BAG" --clock --rate "$RATE"
echo "запись кончилась; RViz остаётся открытым — закройте окно, чтобы выйти"
wait $RVIZ
