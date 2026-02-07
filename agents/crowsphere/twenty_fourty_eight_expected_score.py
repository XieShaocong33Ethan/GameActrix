from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

from agents.crowsphere.twenty_fourty_eight_board import (
    ALLOWED_DIRECTIONS,
    Board,
    empty_count,
    max_tile,
    move_board,
    spawn_distribution,
    heuristic_value,
)


@dataclass(frozen=True)
class ExpectedScoreEstimate:
    direction: str
    changed: bool
    immediate_score_gain: int
    empty_after: int
    max_tile_after: int
    search_depth: int
    expected_score_gain: float
    expected_score_estimate: float


def pick_search_depth(board: Board) -> int:
    # For NxM boards, avoid deep expectimax (branching explodes with empty cells).
    # Keep the original 4x4 behavior unchanged for stable scores/regression tests.
    if (board.rows, board.cols) != (4, 4):
        return 1
    empties = empty_count(board)
    if empties >= 10:
        return 2
    if empties >= 6:
        return 3
    return 4


def estimate_expected_score_for_direction(
    board: Board,
    direction: str,
    *,
    depth: int | None = None,
    heuristic_weight: float = 10.0,
) -> ExpectedScoreEstimate:
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"invalid direction: {direction}")

    search_depth = pick_search_depth(board) if depth is None else int(depth)
    search_depth = max(1, min(6, search_depth))
    if not math.isfinite(heuristic_weight) or heuristic_weight < 0:
        raise ValueError("heuristic_weight must be a finite non-negative number")

    after, gain = move_board(board, direction)
    changed = after != board
    if not changed:
        return ExpectedScoreEstimate(
            direction=direction,
            changed=False,
            immediate_score_gain=0,
            empty_after=empty_count(after),
            max_tile_after=max_tile(after),
            search_depth=search_depth,
            expected_score_gain=float("-inf"),
            expected_score_estimate=float("-inf"),
        )

    remaining_moves = search_depth - 1
    future_score_gain, future_tail_estimate = _chance_node(
        after,
        remaining_moves,
        heuristic_weight,
    )

    expected_score_gain = float(gain) + future_score_gain
    expected_score_estimate = float(gain) + future_tail_estimate

    return ExpectedScoreEstimate(
        direction=direction,
        changed=True,
        immediate_score_gain=gain,
        empty_after=empty_count(after),
        max_tile_after=max_tile(after),
        search_depth=search_depth,
        expected_score_gain=expected_score_gain,
        expected_score_estimate=expected_score_estimate,
    )


def estimate_expected_scores_all_directions(
    board: Board,
    *,
    depth: int | None = None,
    heuristic_weight: float = 10.0,
) -> list[ExpectedScoreEstimate]:
    return [
        estimate_expected_score_for_direction(
            board,
            d,
            depth=depth,
            heuristic_weight=heuristic_weight,
        )
        for d in ALLOWED_DIRECTIONS
    ]


@lru_cache(maxsize=300_000)
def _max_node(
    board: Board,
    remaining_moves: int,
    heuristic_weight: float,
) -> tuple[float, float]:
    if remaining_moves <= 0:
        tail = heuristic_weight * heuristic_value(board)
        return 0.0, float(tail)

    best_score_gain = float("-inf")
    best_tail = float("-inf")
    any_move = False
    for d in ALLOWED_DIRECTIONS:
        after, gain = move_board(board, d)
        if after == board:
            continue
        any_move = True
        score_gain, tail_estimate = _chance_node(after, remaining_moves - 1, heuristic_weight)
        score_gain += float(gain)
        tail_estimate += float(gain)
        if tail_estimate > best_tail:
            best_tail = tail_estimate
            best_score_gain = score_gain
    if not any_move:
        return 0.0, 0.0
    return best_score_gain, best_tail


@lru_cache(maxsize=300_000)
def _chance_node(
    after_board: Board,
    remaining_moves: int,
    heuristic_weight: float,
) -> tuple[float, float]:
    if remaining_moves <= 0:
        tail = heuristic_weight * heuristic_value(after_board)
        return 0.0, float(tail)

    expected_score_gain = 0.0
    expected_tail = 0.0
    for b2, prob in spawn_distribution(after_board):
        score_gain, tail_estimate = _max_node(b2, remaining_moves, heuristic_weight)
        expected_score_gain += prob * score_gain
        expected_tail += prob * tail_estimate
    return expected_score_gain, expected_tail
