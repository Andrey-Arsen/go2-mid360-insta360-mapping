"""Что известно о сессии: meta.json, эталонная геометрия, имя сессии по папке меша.

Сессии стенда Simple Room (record_session.py) окружения в meta не пишут — их эталон
out/gt/simple_room_world.ply. Сессии робота (render_drive.py) пишут "environment" и
"gt_mesh"; без окружения эталона нет (None).
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STAND_GT = ROOT / "out" / "gt" / "simple_room_world.ply"
MESH_SUFFIXES = ("_rko_refined", "_gt_refined", "_rko", "_gt")


def meta(session):
    path = ROOT / "data" / "sessions" / session / "meta.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def session_of(mesh_dir):
    """out/mesh/<сессия>_rko -> <сессия>."""
    name = Path(mesh_dir).name
    for suf in MESH_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def gt_mesh(session):
    """Путь эталонного меша сессии (может ещё не быть выгружен) или None, если эталона нет."""
    m = meta(session)
    if "environment" not in m:
        return STAND_GT
    return ROOT / m["gt_mesh"] if m.get("gt_mesh") else None
