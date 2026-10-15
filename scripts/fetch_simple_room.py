#!/usr/bin/env python3
"""Скачивает Simple Room из ассетов Isaac Sim со всеми зависимостями в assets/isaac/.

Структура папок повторяет облако NVIDIA, поэтому относительные ссылки внутри
USD продолжают работать. После копирования проверяет, что локальная копия не
ссылается ни на что в сети.

Запуск: ~/isaacsim/python.sh scripts/fetch_simple_room.py
"""

import sys
import traceback
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import omni.client  # noqa: E402
from isaacsim.storage.native import get_assets_root_path  # noqa: E402
from pxr import UsdUtils  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / "assets" / "isaac"
ROOM = "Isaac/Environments/Simple_Room/simple_room.usd"


def main():
    remote_root = get_assets_root_path()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(f"{remote_root}/{ROOM}")
    if unresolved:
        raise RuntimeError(f"не разрешились ссылки: {unresolved}")
    urls = [layer.identifier for layer in layers] + list(assets)
    copied = skipped = 0
    for url in urls:
        if not url.startswith(remote_root):
            skipped += 1  # MDL из самой установки Isaac Sim
            continue
        dst = LOCAL / url[len(remote_root) + 1 :]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        result = omni.client.copy(url, str(dst))
        if result != omni.client.Result.OK:
            raise RuntimeError(f"{url}: {result}")
        copied += 1
    print(f"[fetch] скопировано {copied}, взято из установки {skipped}", flush=True)

    _, local_assets, local_unresolved = UsdUtils.ComputeAllDependencies(str(LOCAL / ROOM))
    remote = [a for a in local_assets if "://" in a] + list(local_unresolved)
    if remote:
        raise RuntimeError(f"локальная копия всё ещё тянет из сети: {remote[:5]}")
    print(f"[fetch] готово: {LOCAL / ROOM}", flush=True)


try:
    main()
except BaseException:
    traceback.print_exc()
    sys.stderr.flush()
finally:
    app.close()
