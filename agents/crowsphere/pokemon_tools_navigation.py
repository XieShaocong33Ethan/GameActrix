from __future__ import annotations

from typing import Any


def _nav_from_processed_map(
    *,
    map_name: str,
    pos: tuple[int, int] | None,
    max_preview: int,
) -> dict[str, Any]:
    """
    Compute navigation hints from official processed_map coll_map if available.
    Returns dict with keys:
      - has_processed_map, is_on_warppoint, edge_warp_hint_dir
      - nearest_warppoint, path_to_nearest_warppoint
      - next_keys_to_nearest_warppoint
      - exit_warppoint, path_to_exit_warppoint, next_keys_to_exit_warppoint
    """
    if not map_name or pos is None:
        return {"has_processed_map": False}

    coll_map = _load_coll_map_case_insensitive(map_name)
    if not coll_map:
        return {"has_processed_map": False}

    x, y = pos
    max_y = len(coll_map)
    max_x = len(coll_map[0]) if max_y else 0

    def in_bounds(xx: int, yy: int) -> bool:
        return 0 <= xx < max_x and 0 <= yy < max_y

    if not in_bounds(x, y):
        return {"has_processed_map": True, "in_bounds": False}

    warps = {(xx, yy) for yy, row in enumerate(coll_map) for xx, v in enumerate(row) if str(v) == "WarpPoint"}
    is_on_warppoint = str(coll_map[y][x]) == "WarpPoint"

    edge_dir = None
    if is_on_warppoint:
        if y == 0:
            edge_dir = "up"
        elif y == max_y - 1:
            edge_dir = "down"
        elif x == 0:
            edge_dir = "left"
        elif x == max_x - 1:
            edge_dir = "right"

    nav: dict[str, Any] = {
        "has_processed_map": True,
        "is_on_warppoint": is_on_warppoint,
        "edge_warp_hint_dir": edge_dir,
        "num_warppoints": len(warps),
    }

    # nearest warppoint path
    path_nearest = _shortest_path_to_any_warp(coll_map, start=pos, goals=warps) if warps else None
    # NOTE: allow empty path [] when already standing on a WarpPoint.
    if path_nearest is not None:
        dest = _apply_path(pos, path_nearest)
        nav["nearest_warppoint"] = {"x": dest[0], "y": dest[1], "steps": len(path_nearest)}
        nav["path_to_nearest_warppoint"] = path_nearest[: max(0, int(max_preview))]
        nav["next_keys_to_nearest_warppoint"] = path_nearest[: max(1, int(max_preview))]

    # exit warppoint heuristic: for early indoor maps, bottom-most warp is often the exit.
    exit_warp = None
    if warps:
        if "RedsHouse1f" in map_name:
            exit_warp = max(warps, key=lambda p: p[1])
        elif "House" in map_name or map_name.lower().endswith(("1f", "2f", "b1f", "b2f")):
            exit_warp = max(warps, key=lambda p: p[1])

    if exit_warp:
        path_exit = _shortest_path_to_any_warp(coll_map, start=pos, goals={exit_warp})
        # NOTE: allow empty path [] when already standing on the exit WarpPoint.
        if path_exit is not None:
            nav["exit_warppoint"] = {"x": exit_warp[0], "y": exit_warp[1], "steps": len(path_exit)}
            nav["path_to_exit_warppoint"] = path_exit[: max(0, int(max_preview))]
            nav["next_keys_to_exit_warppoint"] = path_exit[: max(1, int(max_preview))]

    # Overworld navigation helper: shortest path to a map edge (avoids WarpPoint tiles).
    # Useful for "go north/south" subtasks, especially when standing in front of a house door.
    def _edge_goals(direction: str) -> set[tuple[int, int]]:
        d = str(direction or "").strip().lower()
        goals: set[tuple[int, int]] = set()
        if d == "north":
            y0 = 0
            for xx in range(max_x):
                if str(coll_map[y0][xx]) in {"O", ".", "G"}:
                    goals.add((xx, y0))
        elif d == "south":
            y0 = max_y - 1
            for xx in range(max_x):
                if str(coll_map[y0][xx]) in {"O", ".", "G"}:
                    goals.add((xx, y0))
        elif d == "west":
            x0 = 0
            for yy in range(max_y):
                if str(coll_map[yy][x0]) in {"O", ".", "G"}:
                    goals.add((x0, yy))
        elif d == "east":
            x0 = max_x - 1
            for yy in range(max_y):
                if str(coll_map[yy][x0]) in {"O", ".", "G"}:
                    goals.add((x0, yy))
        return goals

    for d in ("north", "south", "west", "east"):
        goals = _edge_goals(d)
        if not goals:
            continue
        path_edge = _shortest_path_to_any_warp(coll_map, start=pos, goals=goals)
        if path_edge is None:
            continue
        dest = _apply_path(pos, path_edge)
        nav[f"{d}_edge"] = {"x": dest[0], "y": dest[1], "steps": len(path_edge)}
        nav[f"path_to_{d}_edge"] = path_edge[: max(0, int(max_preview))]
        nav[f"next_keys_to_{d}_edge"] = path_edge[: max(1, int(max_preview))]

    return nav


def _eval_root() -> Any:
    from pathlib import Path

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
    from pathlib import Path
    import importlib.util

    eval_root = _eval_root()
    processed_dir = (
        Path(eval_root)
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
    from collections import deque

    max_y = len(coll_map)
    max_x = len(coll_map[0]) if max_y else 0

    def in_bounds(x: int, y: int) -> bool:
        return 0 <= x < max_x and 0 <= y < max_y

    def walkable(x: int, y: int, is_goal: bool) -> bool:
        v = str(coll_map[y][x])
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


def _extract_map_on_screen(obs_text: str) -> str:
    """
    从 obs_text 中提取 'Map on Screen:' 后面的地图数据。
    返回地图文本（每行是一行地图字符）。
    """
    if not obs_text:
        return ""

    lines = obs_text.split("\n")
    map_start_idx = -1

    # 找到 "Map on Screen:" 行
    for i, line in enumerate(lines):
        if "Map on Screen:" in line:
            map_start_idx = i + 1
            break

    if map_start_idx < 0 or map_start_idx >= len(lines):
        return ""

    # 提取地图行，直到遇到空行或新的 section 标记
    map_lines: list[str] = []
    for i in range(map_start_idx, len(lines)):
        line = lines[i].strip()
        # 停止条件：空行、[Section] 标记、或其他明显的非地图内容
        if not line or line.startswith("[") or line.startswith("WarpPoints"):
            break
        map_lines.append(line)

    return "\n".join(map_lines)


def _find_northernmost_reachable_in_explored_map(
    *,
    explored_map_text: str,
    current_pos: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """
    基于 Map on Screen，从当前位置用 BFS 找到所有可达点，返回最北（Y 最小）的点。
    用于野外导航时绕过障碍物。

    Map on Screen 格式：
    ( 1,  2): O	( 2,  2): O	( 3,  2): O
    ( 1,  3): O	( 2,  3): C	...
    """
    if not explored_map_text or not current_pos:
        return None

    # 解析坐标-值对：( x,  y): tile
    import re

    tile_map: dict[tuple[int, int], str] = {}

    # 正则匹配：\(\s*(\d+),\s*(\d+)\):\s*([A-Z?~.]+)
    pattern = r"\(\s*(\d+),\s*(\d+)\):\s*([A-Z?~.COG]+)"
    for match in re.finditer(pattern, explored_map_text):
        x = int(match.group(1))
        y = int(match.group(2))
        tile = match.group(3).strip()
        tile_map[(x, y)] = tile

    if not tile_map:
        return None

    cx, cy = current_pos

    def walkable(x: int, y: int) -> bool:
        tile = tile_map.get((x, y))
        if not tile:
            return False
        # O = walkable, G = walkable (grass)
        # C = collision, ~ = water (not walkable without surf), ? = unexplored
        return tile in {"O", "G"}

    if not walkable(cx, cy):
        return None

    # BFS 找到所有可达点
    from collections import deque

    q = deque([(cx, cy)])
    visited = {(cx, cy)}
    reachable_points: list[tuple[int, int]] = [(cx, cy)]

    while q:
        x, y = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not walkable(nx, ny):
                continue
            visited.add((nx, ny))
            reachable_points.append((nx, ny))
            q.append((nx, ny))

    if not reachable_points:
        return None

    # 选择最北的点（Y 最小），如果有多个则选最接近当前 X 坐标的
    reachable_points.sort(key=lambda p: (p[1], abs(p[0] - cx)))
    return reachable_points[0]


def _find_horizontal_extreme_reachable_in_map_on_screen(
    *,
    explored_map_text: str,
    current_pos: tuple[int, int] | None,
    prefer: str,  # "west" | "east"
) -> tuple[int, int] | None:
    """
    基于 Map on Screen 的可达区域，选择一个水平方向上的“极值点”，用于绕开北向被障碍封死的情况。

    - prefer="west": 选择最小 x 的可达点
    - prefer="east": 选择最大 x 的可达点
    """
    if not explored_map_text or not current_pos:
        return None

    import re

    tile_map: dict[tuple[int, int], str] = {}
    pattern = r"\(\s*(\d+),\s*(\d+)\):\s*([A-Z?~.COG]+)"
    for match in re.finditer(pattern, explored_map_text):
        x = int(match.group(1))
        y = int(match.group(2))
        tile = match.group(3).strip()
        tile_map[(x, y)] = tile

    if not tile_map:
        return None

    cx, cy = current_pos

    def walkable(x: int, y: int) -> bool:
        tile = tile_map.get((x, y))
        if not tile:
            return False
        return tile in {"O", "G"}

    if not walkable(cx, cy):
        return None

    from collections import deque

    q = deque([(cx, cy)])
    visited = {(cx, cy)}
    reachable_points: list[tuple[int, int]] = [(cx, cy)]

    while q:
        x, y = q.popleft()
        for dx, dy in [(0, -1), (0, 1), (-1, 0), (1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in visited:
                continue
            if not walkable(nx, ny):
                continue
            visited.add((nx, ny))
            reachable_points.append((nx, ny))
            q.append((nx, ny))

    if not reachable_points:
        return None

    prefer_norm = (prefer or "").strip().lower()
    if prefer_norm == "east":
        reachable_points.sort(key=lambda p: (-p[0], abs(p[1] - cy)))
    else:
        reachable_points.sort(key=lambda p: (p[0], abs(p[1] - cy)))

    best = reachable_points[0]
    return None if best == (cx, cy) else best
