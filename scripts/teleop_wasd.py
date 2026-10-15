#!/usr/bin/env python3
"""Управление Go2 с клавиатуры (WASD) в этом терминале -> /cmd_vel (geometry_msgs/Twist).

  W / S   вперёд / назад          A / D   поворот влево / вправо
  Q / E   шаг влево / вправо      пробел  стоп
  + / -   быстрее / медленнее     Ctrl+C  выход

Клавиша включает движение по своей оси и оно держится (можно вместе: W и A — дуга);
та же клавиша ещё раз — эта ось стоп, пробел — стоп всё. Так удобнее, когда симуляция
идёт медленнее реального времени. Публикует 20 Гц. Окно терминала должно быть активным.

Запуск: source /opt/ros/humble/setup.bash && python3 scripts/teleop_wasd.py [--speed 0.6] [--turn 1.0]
"""

import argparse
import os
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.signals import SignalHandlerOptions

KEYS = {"w": (0, 1), "s": (0, -1), "q": (1, 1), "e": (1, -1), "a": (2, 1), "d": (2, -1)}  # ось, знак


def alive(pid):
    """Работает ли процесс. Зомби (вышел, но лаунчер ещё не забрал его код — bash не забирает,
    пока ждёт этот телеоп) — уже не работает, хотя kill(pid, 0) для него проходит."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.6, help="м/с вперёд")
    parser.add_argument("--turn", type=float, default=1.0, help="рад/с поворот")
    parser.add_argument("--topic", default="cmd_vel")
    parser.add_argument("--watch-pid", type=int, help="выйти, когда этот процесс завершится (Isaac Sim)")
    args = parser.parse_args()
    # Свой Ctrl+C: обработчик rclpy закрыл бы контекст раньше, чем уйдёт команда «стоп».
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("teleop_wasd")
    pub = node.create_publisher(Twist, args.topic, 10)
    speed, turn = args.speed, args.turn
    cmd = [0, 0, 0]  # вперёд, вбок, поворот: -1, 0, +1
    if not sys.stdin.isatty():
        raise SystemExit("teleop_wasd: нужен терминал (клавиши читаются из него) — запустите в окне терминала")
    old = termios.tcgetattr(sys.stdin)
    print(__doc__.split("\n\n")[1], flush=True)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if args.watch_pid and not alive(args.watch_pid):
                print("\nIsaac Sim закрыт — выхожу", flush=True)
                break
            if select.select([sys.stdin], [], [], 0.05)[0]:
                k = sys.stdin.read(1).lower()
                if k in KEYS:
                    axis, sign = KEYS[k]
                    cmd[axis] = 0 if cmd[axis] == sign else sign
                elif k == " ":
                    cmd = [0, 0, 0]
                elif k in "+=":  # политика училась на ±1 м/с и ±1 рад/с — выше не пускаем
                    speed, turn = min(speed * 1.2, 1.0), min(turn * 1.2, 1.0)
                elif k in "-_":
                    speed, turn = speed / 1.2, turn / 1.2
                print(f"\r вперёд {cmd[0] * speed:+.2f} м/с  вбок {cmd[1] * speed * 0.5:+.2f} м/с  "
                      f"поворот {cmd[2] * turn:+.2f} рад/с   (макс {speed:.2f} м/с)   ", end="", flush=True)
            t = Twist()
            t.linear.x, t.linear.y, t.angular.z = cmd[0] * speed, cmd[1] * speed * 0.5, cmd[2] * turn
            pub.publish(t)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        pub.publish(Twist())  # стоп: контроллер держит последнюю команду, таймаута у него нет
        node.destroy_node()
        rclpy.shutdown()
        print()


if __name__ == "__main__":
    main()
