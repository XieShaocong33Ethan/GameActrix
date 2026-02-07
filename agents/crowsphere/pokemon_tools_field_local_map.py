from __future__ import annotations

import re


def _best_move_target_from_local_map(
    *,
    map_info_text: str,
    pos: tuple[int, int] | None,
    direction: str,
) -> tuple[int, int] | None:
    """
    从 obs_text_preproc_pokemon 注入的 Local Map（dy=... 行）中，挑一个“当前视野内可走的落脚点”。
    目的：在没有原始 Map on Screen 坐标列表时，也能给 move_to 提供合法的可见坐标点。
    """
    if not map_info_text or pos is None:
        return None

    cx, cy = pos
    rows: list[tuple[int, str]] = []
    for m in re.finditer(r"^dy=\s*([+-]?\d+):\s*([P.#WTS?]+)\s*$", map_info_text, flags=re.MULTILINE):
        try:
            dy = int(m.group(1))
        except Exception:
            continue
        row = str(m.group(2) or "")
        if not row:
            continue
        rows.append((dy, row))

    if not rows:
        return None

    # Infer radius from row width (should be 2r+1).
    r = (len(rows[0][1]) - 1) // 2
    if r < 1:
        return None

    grid: dict[tuple[int, int], str] = {}
    for dy, row in rows:
        if len(row) != 2 * r + 1:
            continue
        for i, ch in enumerate(row):
            dx = i - r
            grid[(dx, dy)] = ch

    if not grid:
        return None

    def is_walkable(ch: str) -> bool:
        # '.' 是可落脚；'P' 是当前位置；其余符号都视为不可穿越（#/?/W/T/S）。
        return ch in {".", "P"}

    # BFS: only keep locally reachable walkable tiles within the Local Map window.
    from collections import deque

    start = (0, 0)
    if not is_walkable(grid.get(start, "P")):
        return None
    q = deque([start])
    visited = {start}
    while q:
        x0, y0 = q.popleft()
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nxt = (x0 + dx, y0 + dy)
            if nxt in visited:
                continue
            ch = grid.get(nxt)
            if ch is None or not is_walkable(ch):
                continue
            visited.add(nxt)
            q.append(nxt)

    reachable: list[tuple[int, int]] = []
    for (dx, dy), ch in grid.items():
        if (dx, dy) not in visited:
            continue
        if ch != ".":
            continue
        reachable.append((cx + dx, cy + dy))

    if not reachable:
        return None

    d = str(direction or "").strip().lower()
    if d == "north":
        candidates = [(x, y) for (x, y) in reachable if y < cy]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (p[1], abs(p[0] - cx)))
        return candidates[0]
    if d == "south":
        candidates = [(x, y) for (x, y) in reachable if y > cy]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (-p[1], abs(p[0] - cx)))
        return candidates[0]
    if d == "west":
        candidates = [(x, y) for (x, y) in reachable if x < cx]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (p[0], abs(p[1] - cy)))
        return candidates[0]
    if d == "east":
        candidates = [(x, y) for (x, y) in reachable if x > cx]
        if not candidates:
            return None
        candidates.sort(key=lambda p: (-p[0], abs(p[1] - cy)))
        return candidates[0]

    return None

