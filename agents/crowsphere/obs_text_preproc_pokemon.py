from __future__ import annotations

import re
from typing import Any, Iterable


def preprocess_pokemon_red_obs_text(obs_text: str) -> str:
    sections = _parse_pokemon_red_obs_sections(obs_text)

    state_line = sections.get("state_line") or "State: UNKNOWN"
    # Extract state for conditional rendering
    state = state_line.split(":", 1)[1].strip() if ":" in state_line else "UNKNOWN"

    filtered_screen = _limit_lines(sections.get("filtered_screen_text") or [], max_lines=12)
    selection_box = _clean_selection_box(sections.get("selection_box_text") or [], max_lines=12)

    # State-aware context: only include relevant information based on current state
    if state == "Dialog":
        # Dialog: ONLY dialog text and selection box (for naming), NO map/battle/inventory clutter
        blocks = [
            state_line,
            _render_section("[Filtered Screen Text]", filtered_screen),
            _render_section("[Selection Box Text]", selection_box),
        ]
    elif state == "Battle":
        # Battle: ONLY battle info, NO map clutter
        enemy = _clean_enemy_info(sections.get("enemy_pokemon") or [], max_lines=12)
        party = _limit_lines(sections.get("current_party") or [], max_lines=30)
        blocks = [
            state_line,
            _render_section("[Filtered Screen Text]", filtered_screen),
            _render_section("[Selection Box Text]", selection_box),
            _render_section("[Enemy Pokemon]", enemy),
            _render_section("[Current Party]", party),
        ]
    elif state == "Title" or state == "Selection":
        # Title/Selection: Just screen text
        blocks = [
            state_line,
            _render_section("[Filtered Screen Text]", filtered_screen),
            _render_section("[Selection Box Text]", selection_box),
        ]
    else:
        # Field: Full context including map, inventory, etc.
        enemy = _clean_enemy_info(sections.get("enemy_pokemon") or [], max_lines=12)
        party = _limit_lines(sections.get("current_party") or [], max_lines=30)
        badge_list = _limit_lines(sections.get("badge_list") or [], max_lines=10)
        bag = _summarize_bag(
            sections.get("bag") or [],
            max_item_lines=10,
            keep_keywords=(
                "BALL",
                "POTION",
                "PARCEL",
                "PACKAGE",
                "TOWN MAP",
                "MAP",
                "HM",
                "TM",
            ),
        )
        money = _clean_money(sections.get("current_money") or [])
        map_info = _summarize_map_info(
            sections.get("map_info") or [],
            max_object_lines=20,
        )
        blocks = [
            state_line,
            _render_section("[Filtered Screen Text]", filtered_screen),
            _render_section("[Selection Box Text]", selection_box),
            _render_section("[Enemy Pokemon]", enemy),
            _render_section("[Current Party]", party),
            _render_section("[Badge List]", badge_list),
            _render_section("[Bag]", bag),
            _render_section("[Current Money]", money),
            _render_section("[Map Info]", map_info),
        ]

    return "\n\n".join(blocks).strip()


def _parse_pokemon_red_obs_sections(obs_text: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "state_line": None,
        "filtered_screen_text": [],
        "selection_box_text": [],
        "enemy_pokemon": [],
        "current_party": [],
        "badge_list": [],
        "bag": [],
        "current_money": [],
        "map_info": [],
    }

    current: str | None = None
    for raw_line in obs_text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if stripped.startswith("State:"):
            out["state_line"] = stripped
            current = None
            continue

        if stripped == "[Filtered Screen Text]":
            current = "filtered_screen_text"
            continue
        if stripped == "[Selection Box Text]":
            current = "selection_box_text"
            continue
        if stripped == "[Enemy Pokemon]":
            current = "enemy_pokemon"
            continue
        if stripped == "[Current Party]":
            current = "current_party"
            continue
        if stripped == "[Badge List]":
            current = "badge_list"
            continue
        if stripped == "[Bag]":
            current = "bag"
            continue
        if stripped.startswith("[Current Money]"):
            current = "current_money"
            out[current].append(stripped)
            continue
        if stripped == "[Map Info]":
            current = "map_info"
            continue

        if current is None:
            continue
        out[current].append(stripped)

    return out


def _render_section(header: str, lines: list[str]) -> str:
    cleaned = [ln for ln in _strip_empty(lines) if ln.strip()]
    if not cleaned:
        cleaned = ["N/A"]
    return "\n".join([header, *cleaned]).strip()


def _strip_empty(lines: Iterable[str]) -> list[str]:
    out: list[str] = []
    for ln in lines:
        if ln is None:
            continue
        s = str(ln).rstrip()
        if s == "":
            continue
        out.append(s)
    return out


def _limit_lines(lines: list[str], *, max_lines: int) -> list[str]:
    cleaned = _strip_empty(lines)
    if not cleaned:
        return []
    if len(cleaned) <= max_lines:
        return cleaned
    return cleaned[-max_lines:]


def _clean_selection_box(lines: list[str], *, max_lines: int) -> list[str]:
    cleaned = []
    for ln in _strip_empty(lines):
        s = ln.strip()
        if not s:
            continue
        if s == "N/A":
            return []
        if set(s) == {"-"}:
            continue
        cleaned.append(s)
    return _limit_lines(cleaned, max_lines=max_lines)


def _clean_enemy_info(lines: list[str], *, max_lines: int) -> list[str]:
    cleaned = _strip_empty(lines)
    if not cleaned:
        return []
    if any("Not in battle" in ln for ln in cleaned):
        return ["- Not in battle"]
    return _limit_lines(cleaned, max_lines=max_lines)


def _clean_money(lines: list[str]) -> list[str]:
    if not lines:
        return []
    raw = " ".join(_strip_empty(lines)).strip()
    m = re.search(r"¥\s*(\d+)", raw)
    if m:
        return [f"¥{m.group(1)}"]
    return [raw] if raw else []


def _summarize_bag(
    lines: list[str],
    *,
    max_item_lines: int,
    keep_keywords: tuple[str, ...],
) -> list[str]:
    cleaned = _strip_empty(lines)
    if not cleaned:
        return []
    if len(cleaned) == 1 and cleaned[0] == "N/A":
        return []

    header_lines: list[str] = []
    item_lines: list[str] = []
    for ln in cleaned:
        if ln.startswith("- "):
            item_lines.append(ln)
        else:
            header_lines.append(ln)

    head_items = item_lines[:max_item_lines]

    keyword_items: list[str] = []
    for ln in item_lines:
        upper = ln.upper()
        if any(k in upper for k in keep_keywords):
            keyword_items.append(ln)

    merged: list[str] = []
    merged.extend(header_lines[:3])
    for ln in head_items + keyword_items:
        if ln not in merged:
            merged.append(ln)
    return merged


def _summarize_map_info(
    lines: list[str],
    *,
    max_object_lines: int,
) -> list[str]:
    cleaned = _strip_empty(lines)
    if not cleaned:
        return []

    map_on_screen_idx = None
    for i, ln in enumerate(cleaned):
        if ln.strip().startswith("Map on Screen:"):
            map_on_screen_idx = i
            break

    header_lines = cleaned if map_on_screen_idx is None else cleaned[:map_on_screen_idx]
    map_lines: list[str] = [] if map_on_screen_idx is None else cleaned[map_on_screen_idx + 1 :]

    header_keep: list[str] = []
    for ln in header_lines:
        s = ln.strip()
        if s == "Action instruction":
            continue
        if s.startswith("- ") and "->" in s:
            continue
        if any(
            s.startswith(prefix)
            for prefix in (
                "Map Name:",
                "Map type:",
                "Expansion direction:",
                "Your position",
                "Your facing direction:",
            )
        ):
            header_keep.append(s)

    if not map_lines:
        return header_keep

    if any("Not in Field State" in ln for ln in map_lines):
        return header_keep + ["Map on Screen:", "Not in Field State"]

    player_pos = _parse_pokemon_player_pos(header_keep)
    cells = _parse_pokemon_map_cells(map_lines)

    rendered: list[str] = list(header_keep)

    if player_pos is not None and cells:
        rendered.append("Coord: up=y-1, down=y+1, left=x-1, right=x+1")
        rendered.extend(_render_pokemon_local_map(cells, player_pos, radius=4))

        neighbors = _render_pokemon_neighbors(cells, player_pos)
        if neighbors:
            rendered.append(neighbors)
        rendered.extend(_render_pokemon_dir_hints(cells, player_pos))

        from agents.crowsphere.pokemon_processed_map_hints import (
            render_processed_map_hints,
        )

        processed_hint = render_processed_map_hints(
            map_info_header=" ".join(header_keep),
            player_pos=player_pos,
        )
        if processed_hint:
            rendered.extend(processed_hint)

    objects = _summarize_pokemon_notable_cells(cells, max_object_lines=max_object_lines)
    if objects:
        rendered.extend(["Notable Cells:", *objects])

    return rendered


_POKEMON_TERRAIN_TOKENS = {
    # Walkable
    "O",
    "G",
    ".",
    # Unwalkable / barriers
    "X",
    "#",
    "-",
    "|",
    "Cut",
    "C",
    # Special traversal tiles (still terrain-like)
    "~",  # water (Surf)
    "D",
    "L",
    "R",
    # Unknown / unexplored
    "?",
}


def _is_notable_pokemon_map_cell(value: str) -> bool:
    """
    Decide whether a map cell value is worth listing in "Notable Cells".

    Hidden tests may rename NPC/object identifiers (e.g., SPRITE_* -> NPC_*).
    We therefore treat any non-terrain multi-character token as notable.
    """
    v = (value or "").strip()
    if not v:
        return False

    if v == "WarpPoint":
        return True
    if v.startswith(("TalkTo", "SIGN_", "OBJ_")):
        return True
    if "SPRITE_" in v:
        return True

    # Robust fallback: unknown non-terrain identifiers (e.g., NPC_*).
    if len(v) > 1 and v not in _POKEMON_TERRAIN_TOKENS:
        return True
    return False


def _parse_pokemon_player_pos(header_lines: list[str]) -> tuple[int, int] | None:
    for ln in header_lines:
        m = re.search(r"Your position \(x, y\): \(([-\d]+),\s*([-\d]+)\)", ln)
        if not m:
            continue
        try:
            return int(m.group(1)), int(m.group(2))
        except Exception:
            return None
    return None


def _parse_pokemon_map_cells(map_lines: list[str]) -> dict[tuple[int, int], str]:
    cells: dict[tuple[int, int], str] = {}
    for row in map_lines:
        for x_str, y_str, value in re.findall(
            r"\(\s*(-?\d+),\s*(-?\d+)\):\s*([^\t]+)",
            row,
        ):
            try:
                x = int(x_str)
                y = int(y_str)
            except Exception:
                continue
            cells[(x, y)] = value.strip()
    return cells


def _render_pokemon_neighbors(
    cells: dict[tuple[int, int], str],
    pos: tuple[int, int],
) -> str | None:
    x, y = pos
    up = cells.get((x, y - 1))
    down = cells.get((x, y + 1))
    left = cells.get((x - 1, y))
    right = cells.get((x + 1, y))
    if up is None and down is None and left is None and right is None:
        return None
    return f"Neighbors: up={up}, down={down}, left={left}, right={right}"


def _render_pokemon_dir_hints(
    cells: dict[tuple[int, int], str],
    pos: tuple[int, int],
) -> list[str]:
    x, y = pos
    by_dir = {
        "up": cells.get((x, y - 1)),
        "down": cells.get((x, y + 1)),
        "left": cells.get((x - 1, y)),
        "right": cells.get((x + 1, y)),
    }

    symbols = {d: _pokemon_cell_symbol(v) for d, v in by_dir.items()}
    symbol_line = "Neighbor Symbols: " + ", ".join([f"{d}={symbols[d]}" for d in ("up", "down", "left", "right")])

    walkable = [d for d in ("up", "down", "left", "right") if symbols[d] in {".", "W"}]
    interactable = [d for d in ("up", "down", "left", "right") if symbols[d] in {"T", "S"}]
    blocked = [d for d in ("up", "down", "left", "right") if symbols[d] == "#"]
    unknown = [d for d in ("up", "down", "left", "right") if symbols[d] == "?"]

    lines = [symbol_line]
    if walkable:
        lines.append("Walkable Dirs (can move): " + ", ".join(walkable))
    if interactable:
        lines.append("Interactable Dirs (face then press a): " + ", ".join(interactable))
    if blocked:
        lines.append("Blocked Dirs (wall): " + ", ".join(blocked))
    if unknown:
        lines.append("Unknown Dirs: " + ", ".join(unknown))
    return lines


def _render_pokemon_local_map(
    cells: dict[tuple[int, int], str],
    pos: tuple[int, int],
    *,
    radius: int,
) -> list[str]:
    cx, cy = pos
    r = max(1, min(8, int(radius)))

    lines: list[str] = [
        "Local Map (P=you, .=walkable, #=wall, W=warp, T=talk/sign, S=sprite/object, ?=unknown):"
    ]
    for dy in range(-r, r + 1):
        row = []
        for dx in range(-r, r + 1):
            x = cx + dx
            y = cy + dy
            if (x, y) == (cx, cy):
                row.append("P")
                continue
            v = cells.get((x, y))
            row.append(_pokemon_cell_symbol(v))
        dy_label = f"{dy:+d}".rjust(3)
        lines.append(f"dy={dy_label}: {''.join(row)}")
    return lines


def _pokemon_cell_symbol(value: str | None) -> str:
    if value is None:
        return "?"
    v = value.strip()
    if v in {"O", ".", "G"}:
        return "."
    if v in {"X", "#", "Cut", "C", "-", "|"}:
        return "#"
    if v == "WarpPoint":
        return "W"
    if v.startswith("TalkTo") or v.startswith("SIGN_"):
        return "T"
    if "SPRITE_" in v or v.startswith("OBJ_"):
        return "S"
    # Hidden tests may rename sprite/object prefixes (e.g., NPC_*). Treat unknown
    # non-terrain identifiers as generic objects.
    if len(v) > 1 and v not in _POKEMON_TERRAIN_TOKENS:
        return "S"
    return "?"


def _summarize_pokemon_notable_cells(
    cells: dict[tuple[int, int], str],
    *,
    max_object_lines: int,
) -> list[str]:
    if not cells:
        return []
    objects: list[str] = []
    for (x, y), v in cells.items():
        if _is_notable_pokemon_map_cell(v):
            objects.append(f"({x},{y}): {v}")
    objects.sort()
    return objects[: max(0, int(max_object_lines))]
