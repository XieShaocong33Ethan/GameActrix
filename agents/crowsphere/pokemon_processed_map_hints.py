from __future__ import annotations

import importlib.util
from collections import deque
from pathlib import Path
import re


def render_processed_map_hints(
    *,
    map_info_header: str,
    player_pos: tuple[int, int],
) -> list[str] | None:
    """
    从官方 processed_map/<MapName>.py 读取 coll_map，给出 WarpPoint 位置与最短路径提示。

    目标：让模型在室内地图（如 RedsHouse2f）能快速找到楼梯/出口。
    """

    m = re.search(r"Map Name:\s*([^,]+)", map_info_header)
    if not m:
        return None
    name = m.group(1).strip()

    coll_map = _load_coll_map_case_insensitive(name)
    if not coll_map:
        return None

    warps = {(x, y) for y, row in enumerate(coll_map) for x, v in enumerate(row) if v == "WarpPoint"}
    if not warps:
        return None

    path = _shortest_path_to_any_warp(coll_map, start=player_pos, goals=warps)
    warp_list = sorted(warps)
    if not path:
        return ["WarpPoints (processed_map): " + ", ".join([f"({x},{y})" for x, y in warp_list])]

    dest = _apply_path(player_pos, path)
    preview = " ".join(path[:12])
    return [
        "WarpPoints (processed_map): " + ", ".join([f"({x},{y})" for x, y in warp_list]),
        f"Nearest WarpPoint: ({dest[0]},{dest[1]}), steps={len(path)}",
        f"Path to WarpPoint (first 12): {preview}",
    ]


def _eval_root() -> Path:
    """
    尽量在不同运行布局下都能找到官方 starter kit 根目录：
    - 在线/官方：<eval_root>/evaluation_utils/...
    - 本仓库：<repo_root>/eval/orak-2025-starter-kit/evaluation_utils/...
    """

    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "evaluation_utils").is_dir():
            return parent

    repo_root = here.parents[2]
    candidate = repo_root / "eval" / "orak-2025-starter-kit"
    if (candidate / "evaluation_utils").is_dir():
        return candidate

    return repo_root


def _load_coll_map_case_insensitive(map_name: str) -> list[list[str]] | None:
    eval_root = _eval_root()
    processed_dir = (
        eval_root
        / "evaluation_utils"
        / "mcp_game_servers"
        / "pokemon_red"
        / "game"
        / "processed_map"
    )
    want = (map_name + ".py").lower()
    picked = None
    for p in processed_dir.glob("*.py"):
        if p.name.lower() == want:
            picked = p
            break
    if picked is None:
        return None

    spec = importlib.util.spec_from_file_location(picked.stem, picked)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    coll_map = getattr(mod, "coll_map", None)
    if not isinstance(coll_map, list) or not coll_map:
        return None
    if not isinstance(coll_map[0], list):
        return None

    out: list[list[str]] = []
    for row in coll_map:
        if not isinstance(row, list):
            return None
        out.append([str(v) for v in row])
    return out


def _apply_path(start: tuple[int, int], path: list[str]) -> tuple[int, int]:
    x, y = start
    for step in path:
        if step == "up":
            y -= 1
        elif step == "down":
            y += 1
        elif step == "left":
            x -= 1
        elif step == "right":
            x += 1
    return x, y


def _shortest_path_to_any_warp(
    coll_map: list[list[str]],
    *,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
) -> list[str] | None:
    max_y = len(coll_map)
    max_x = len(coll_map[0]) if max_y else 0

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < max_x and 0 <= y < max_y

    def walkable(x: int, y: int, is_goal: bool) -> bool:
        v = coll_map[y][x]
        if v in {"O", ".", "G"}:
            return True
        if v == "WarpPoint":
            return is_goal
        return False

    sx, sy = start
    if not in_bounds(sx, sy):
        return None

    q = deque([(sx, sy)])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
    move_taken: dict[tuple[int, int], str] = {}

    while q:
        x, y = q.popleft()
        if (x, y) in goals:
            path: list[str] = []
            cur = (x, y)
            while parent[cur] is not None:
                path.append(move_taken[cur])
                cur = parent[cur]  # type: ignore[assignment]
            path.reverse()
            return path

        for step, (dx, dy) in (
            ("up", (0, -1)),
            ("down", (0, 1)),
            ("left", (-1, 0)),
            ("right", (1, 0)),
        ):
            nx, ny = x + dx, y + dy
            if not in_bounds(nx, ny):
                continue
            if (nx, ny) in parent:
                continue
            is_goal = (nx, ny) in goals
            if not walkable(nx, ny, is_goal):
                continue
            parent[(nx, ny)] = (x, y)
            move_taken[(nx, ny)] = step
            q.append((nx, ny))

    return None
