#!/usr/bin/env bash
# RViz для стенда: облако Mid-360, кадры обоих объективов, TF.
# Время — симуляции (/clock от Isaac Sim). ROS_DOMAIN_ID — из окружения, как у Isaac Sim.
set -e
cd "$(dirname "$0")/.."
source scripts/ros_env.sh
exec rviz2 -d rviz/stend.rviz --ros-args -p use_sim_time:=true
