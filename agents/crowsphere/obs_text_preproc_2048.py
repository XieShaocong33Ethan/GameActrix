from __future__ import annotations

import re

from agents.crowsphere.twenty_fourty_eight_board import (
    board_to_grid,
    empty_count,
    max_tile as board_max_tile,
    parse_board_from_obs_text,
)
from agents.crowsphere.twenty_fourty_eight_expected_score import (
    estimate_expected_scores_all_directions,
)


def preprocess_twenty_fourty_eight_obs_text(obs_text: str) -> str:
    current_score = _parse_twenty_fourty_eight_score(obs_text)
    board = parse_board_from_obs_text(obs_text)
    if board is None:
        return obs_text.strip() or "N/A"

    grid = board_to_grid(board)
    empties = empty_count(board)
    max_tile = board_max_tile(board)

    analyses = []
    valid_dirs: list[str] = []
    move_stats = estimate_expected_scores_all_directions(board)
    for m in move_stats:
        d = m.direction
        changed = m.changed
        expected_score = m.expected_score_estimate if changed else float("-inf")
        analyses.append(
            f"- {d}: change={'yes' if changed else 'no'}, merge_gain={m.immediate_score_gain}, "
            f"empty_after={m.empty_after}, max_after={m.max_tile_after}, expected_score={expected_score:.3f}"
        )
        if changed:
            valid_dirs.append(d)

    preferred_order = {"left": 0, "down": 1, "right": 2, "up": 3}
    candidates = [m for m in move_stats if m.changed]
    candidates.sort(
        key=lambda m: (
            -float(m.expected_score_estimate),
            -int(m.immediate_score_gain),
            -int(m.empty_after),
            preferred_order.get(m.direction, 9),
        )
    )
    recommended = candidates[0].direction if candidates else "NONE"

    score_line = (
        f"Current Score: {current_score}"
        if current_score is not None
        else "Current Score: N/A"
    )

    grid_lines = [" ".join(f"{v:4d}" for v in row) for row in grid]
    blocks = [
        "[2048]",
        "Grid:",
        *grid_lines,
        score_line,
        f"Empty Cells: {empties}",
        f"Max Tile: {max_tile}",
        "Move Analysis (expectimax expected score estimate):",
        *analyses,
        f"Valid Dirs (change=yes): {', '.join(valid_dirs) if valid_dirs else 'NONE'}",
        f"Recommended Dir: {recommended}",
    ]
    return "\n".join(blocks).strip()


def _parse_twenty_fourty_eight_score(obs_text: str) -> int | None:
    m = re.search(r"Score:\s*(\d+)", obs_text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None
