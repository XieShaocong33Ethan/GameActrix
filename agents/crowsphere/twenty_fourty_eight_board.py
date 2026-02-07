from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable


@dataclass(frozen=True)
class BoardState:
    """2048 board state (row-major) with explicit dimensions."""

    rows: int
    cols: int
    cells: tuple[int, ...]  # row-major, length = rows * cols

    def __post_init__(self) -> None:
        if self.rows <= 0 or self.cols <= 0:
            raise ValueError("rows/cols must be positive")
        if len(self.cells) != self.rows * self.cols:
            raise ValueError("cells length must equal rows*cols")


Board = BoardState

ALLOWED_DIRECTIONS = ("up", "down", "left", "right")

_BRACKET_ROW_RE = re.compile(r"^\s*\[(?:\s*\d+\s*,)*\s*\d+\s*\]\s*$")


def parse_board_from_obs_text(obs_text: str) -> Board | None:
    """Parse board from obs_text.

    Supported formats:
    - Raw env obs: lines like `[0, 2, 0, 2]`
    - Preprocessed obs: a `Grid:` block with whitespace-separated numbers
    """

    # 1) Prefer raw env format: consecutive bracket rows.
    bracket_rows: list[list[int]] = []
    for ln in (obs_text or "").splitlines():
        if not _BRACKET_ROW_RE.match(ln):
            continue
        nums = [int(x) for x in re.findall(r"\d+", ln)]
        if nums:
            bracket_rows.append(nums)

    if bracket_rows:
        cols = len(bracket_rows[0])
        if cols >= 2 and all(len(r) == cols for r in bracket_rows):
            cells = tuple(v for r in bracket_rows for v in r)
            if len(bracket_rows) >= 2:
                return BoardState(rows=len(bracket_rows), cols=cols, cells=cells)

    # 2) Fallback: parse `Grid:` block from preprocessed obs.
    lines = (obs_text or "").splitlines()
    grid_start = None
    for i, ln in enumerate(lines):
        if ln.strip().lower() == "grid:":
            grid_start = i + 1
            break
    if grid_start is None:
        return None

    grid_rows: list[list[int]] = []
    for ln in lines[grid_start:]:
        if not ln.strip():
            break
        # Stop at metadata lines like "Current Score: ..."
        if ":" in ln:
            break
        nums = [int(x) for x in re.findall(r"\d+", ln)]
        if not nums:
            break
        grid_rows.append(nums)

    if not grid_rows:
        return None
    cols = len(grid_rows[0])
    if cols < 2 or not all(len(r) == cols for r in grid_rows):
        return None
    if len(grid_rows) < 2:
        return None
    cells = tuple(v for r in grid_rows for v in r)
    return BoardState(rows=len(grid_rows), cols=cols, cells=cells)


def board_to_grid(board: Board) -> list[list[int]]:
    return [
        list(board.cells[r * board.cols : (r + 1) * board.cols])
        for r in range(board.rows)
    ]


def empty_indices(board: Board) -> tuple[int, ...]:
    return tuple(i for i, v in enumerate(board.cells) if v == 0)


def empty_count(board: Board) -> int:
    return sum(1 for v in board.cells if v == 0)


def max_tile(board: Board) -> int:
    return max(board.cells, default=0)


@lru_cache(maxsize=200_000)
def move_board(board: Board, direction: str) -> tuple[Board, int]:
    if direction not in ALLOWED_DIRECTIONS:
        raise ValueError(f"invalid direction: {direction}")

    if direction == "left":
        return _move_left(board)
    if direction == "right":
        return _move_right(board)
    if direction == "up":
        return _move_up(board)
    return _move_down(board)


def _move_left(board: Board) -> tuple[Board, int]:
    out: list[int] = []
    total_gain = 0
    for y in range(board.rows):
        start = y * board.cols
        line = board.cells[start : start + board.cols]
        merged, gain = _merge_line(tuple(line))
        out.extend(list(merged))
        total_gain += gain
    return BoardState(rows=board.rows, cols=board.cols, cells=tuple(out)), total_gain


def _move_right(board: Board) -> tuple[Board, int]:
    out: list[int] = []
    total_gain = 0
    for y in range(board.rows):
        start = y * board.cols
        line = tuple(reversed(board.cells[start : start + board.cols]))
        merged, gain = _merge_line(line)
        out.extend(list(reversed(merged)))
        total_gain += gain
    return BoardState(rows=board.rows, cols=board.cols, cells=tuple(out)), total_gain


def _move_up(board: Board) -> tuple[Board, int]:
    out = [0] * (board.rows * board.cols)
    total_gain = 0
    for x in range(board.cols):
        col = tuple(board.cells[y * board.cols + x] for y in range(board.rows))
        merged, gain = _merge_line(col)
        total_gain += gain
        for y in range(board.rows):
            out[y * board.cols + x] = merged[y]
    return BoardState(rows=board.rows, cols=board.cols, cells=tuple(out)), total_gain


def _move_down(board: Board) -> tuple[Board, int]:
    out = [0] * (board.rows * board.cols)
    total_gain = 0
    for x in range(board.cols):
        col = tuple(
            board.cells[y * board.cols + x]
            for y in range(board.rows - 1, -1, -1)
        )
        merged, gain = _merge_line(col)
        total_gain += gain
        merged_rev = tuple(reversed(merged))
        for y in range(board.rows):
            out[y * board.cols + x] = merged_rev[y]
    return BoardState(rows=board.rows, cols=board.cols, cells=tuple(out)), total_gain


@lru_cache(maxsize=200_000)
def _merge_line(line: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    nonzero = [v for v in line if v != 0]
    merged: list[int] = []
    gain = 0
    i = 0
    while i < len(nonzero):
        if i + 1 < len(nonzero) and nonzero[i] == nonzero[i + 1]:
            nv = nonzero[i] * 2
            merged.append(nv)
            gain += nv
            i += 2
        else:
            merged.append(nonzero[i])
            i += 1
    merged.extend([0] * (len(line) - len(merged)))
    return tuple(merged[: len(line)]), gain


def spawn_distribution(board: Board) -> Iterable[tuple[Board, float]]:
    empties = empty_indices(board)
    if not empties:
        return [(board, 1.0)]

    total = sum(board.cells)
    if total in (0, 2):
        tile_probs = ((2, 1.0),)
    else:
        tile_probs = ((2, 0.9), (4, 0.1))

    per_cell = 1.0 / float(len(empties))
    out: list[tuple[Board, float]] = []
    for idx in empties:
        for tile, prob in tile_probs:
            b = list(board.cells)
            b[idx] = tile
            out.append((BoardState(rows=board.rows, cols=board.cols, cells=tuple(b)), per_cell * prob))
    return out


def heuristic_value(board: Board) -> float:
    empty = empty_count(board)
    m = max_tile(board)
    log_max = float(_log2(m))

    smooth = _smoothness(board)
    mono = _monotonicity(board)
    # Strongly prefer keeping the current max tile in a corner. A constant bonus (e.g. 0/1)
    # becomes negligible once tiles grow; scale by log2(max_tile) instead.
    corner_bonus = float(_log2(m)) if _max_in_corner(board, m) else 0.0

    return (
        2.7 * float(empty)
        + 1.0 * float(mono)
        + 0.1 * float(smooth)
        + 1.0 * float(log_max)
        + 2.0 * float(corner_bonus)
    )


def _log2(value: int) -> int:
    if value <= 0:
        return 0
    return value.bit_length() - 1


def _max_in_corner(board: Board, m: int) -> bool:
    if m <= 0:
        return False
    return m in (
        board.cells[0],
        board.cells[board.cols - 1],
        board.cells[(board.rows - 1) * board.cols],
        board.cells[board.rows * board.cols - 1],
    )


def _smoothness(board: Board) -> float:
    penalty = 0.0
    for y in range(board.rows):
        for x in range(board.cols):
            v = board.cells[y * board.cols + x]
            if v == 0:
                continue
            lv = _log2(v)
            if x + 1 < board.cols:
                nv = board.cells[y * board.cols + (x + 1)]
                if nv != 0:
                    penalty += abs(float(lv - _log2(nv)))
            if y + 1 < board.rows:
                nv = board.cells[(y + 1) * board.cols + x]
                if nv != 0:
                    penalty += abs(float(lv - _log2(nv)))
    return -penalty


def _monotonicity(board: Board) -> float:
    logg = [_log2(v) for v in board.cells]
    totals = [0.0, 0.0, 0.0, 0.0]  # up, down, left, right

    for x in range(board.cols):
        for y in range(board.rows - 1):
            a = logg[y * board.cols + x]
            b = logg[(y + 1) * board.cols + x]
            if a > b:
                totals[0] += float(b - a)
            elif b > a:
                totals[1] += float(a - b)

    for y in range(board.rows):
        for x in range(board.cols - 1):
            a = logg[y * board.cols + x]
            b = logg[y * board.cols + (x + 1)]
            if a > b:
                totals[2] += float(b - a)
            elif b > a:
                totals[3] += float(a - b)

    return max(totals[0], totals[1]) + max(totals[2], totals[3])
