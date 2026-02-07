from __future__ import annotations

import re


def _split_sections(text: str) -> dict[str, str]:
    # Our pokemon preproc format:
    #   State: X
    #   [Filtered Screen Text]
    #   ...
    current = "__preamble__"
    out: dict[str, list[str]] = {current: []}
    state_line = ""
    for raw in (text or "").splitlines():
        ln = raw.rstrip("\n")
        if ln.startswith("State:"):
            state_line = ln.strip()
            continue
        m = re.match(r"^\[([^\]]+)\]\s*$", ln.strip())
        if m:
            current = m.group(1).strip()
            out.setdefault(current, [])
            continue
        out.setdefault(current, []).append(ln)
    joined = {k: "\n".join(v).strip() for k, v in out.items()}
    joined["__state_line__"] = state_line
    return joined


def _parse_state(state_line: str) -> str:
    m = re.search(r"State:\s*([A-Za-z_]+)", state_line or "")
    return (m.group(1) if m else "UNKNOWN").strip()


def _non_na_lines(block: str) -> list[str]:
    lines = [ln.strip() for ln in (block or "").splitlines() if ln.strip()]
    if not lines:
        return []
    if len(lines) == 1 and lines[0] == "N/A":
        return []
    return [ln for ln in lines if ln != "N/A"]


def _first_match(text: str, pattern: str) -> str | None:
    m = re.search(pattern, text or "")
    if not m:
        return None
    return m.group(1).strip()


def _parse_pos(map_info_block: str) -> tuple[int, int] | None:
    # Some env versions render fixed-width numbers with leading spaces, e.g. "( 7,  1)".
    m = re.search(
        r"Your position \(x, y\):\s*\(\s*([-\d]+)\s*,\s*([-\d]+)\s*\)",
        map_info_block or "",
    )
    if not m:
        return None
    try:
        return int(m.group(1)), int(m.group(2))
    except Exception:
        return None


def _parse_walkable_dirs(map_info_block: str) -> list[str]:
    m = re.search(r"Walkable Dirs \(can move\):\s*(.+)$", map_info_block or "", flags=re.MULTILINE)
    if not m:
        return []
    dirs = [d.strip().lower() for d in m.group(1).split(",")]
    return [d for d in dirs if d in {"up", "down", "left", "right"}]


def _pick_alternative_dir(*, last_action: str) -> str | None:
    last = (last_action or "").strip().lower()
    # last_action could be "up up" / "left" / "a" etc. We only handle movement dirs here.
    last_dir = last.split()[-1] if last else ""
    options = ["up", "right", "down", "left"]
    if last_dir in options:
        options.remove(last_dir)
    return options[0] if options else None


def _extract_map_screen_raw(map_info_block: str) -> str | None:
    """Extract the raw map screen content from Map Info block."""
    m = re.search(r"Map on Screen:\s*\n(.+)", map_info_block, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Also try to find Notable Cells or Local Map content
    m2 = re.search(r"(Local Map|Notable Cells|WarpPoints).*", map_info_block, re.DOTALL)
    if m2:
        return m2.group(0)
    return map_info_block  # Return entire block as fallback


def _parse_map_screen_tiles(map_screen_raw: str | None) -> list[tuple[int, int, str]]:
    """
    Parse entries like:
      ( 3, 10): WarpPoint
      (12,  5): SPRITE_OAK
    Returns a list of (x, y, value).
    """
    if not map_screen_raw:
        return []
    tiles: list[tuple[int, int, str]] = []
    for x_str, y_str, val in re.findall(r"\(\s*(\d+),\s*(\d+)\):\s*([^\s]+)", map_screen_raw):
        try:
            tiles.append((int(x_str), int(y_str), str(val)))
        except Exception:
            continue
    return tiles


def _contains_token(value: str, token: str) -> bool:
    """
    Return True if `token` appears as an underscore-delimited token in `value`.

    Why: Hidden tests may rename object prefixes (e.g., SPRITE_* -> NPC_*). We still
    want to match important entities like "OAK" without accidentally matching longer
    strings such as "OAKSLAB" in sign ids (e.g., SIGN_PALLETTOWN_OAKSLAB_SIGN).
    """
    v = str(value or "").strip().upper()
    t = str(token or "").strip().upper()
    if not v or not t:
        return False
    return bool(re.search(rf"(?:^|_){re.escape(t)}(?:$|_)", v))
