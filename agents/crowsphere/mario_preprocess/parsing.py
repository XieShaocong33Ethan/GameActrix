from __future__ import annotations

import re

from agents.crowsphere.mario_preprocess.types import PipeInfo, Position


def parse_env_int(obs_text: str, key: str) -> int | None:
    """Parse an integer field from an injected `[Env Info]` block.

    Example:
      [Env Info]
      x_pos: 2751
      time: 305
    """
    text = str(obs_text or "")
    k = str(key or "").strip()
    if not k:
        return None
    m = re.search(rf"\b{re.escape(k)}\s*:\s*(-?\d+)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_mario_position(obs_text: str) -> Position | None:
    # y 可能在“坠落/死亡动画”阶段为负值（例如 -20），这里要允许负数，否则会丢失关键的距离计算依据。
    match = re.search(r"Position of Mario:\s*\((-?\d+),\s*(-?\d+)\)", obs_text or "")
    if match:
        return Position(int(match.group(1)), int(match.group(2)))
    return None


def parse_positions(obs_text: str, object_type: str) -> list[Position]:
    positions: list[Position] = []
    pattern = rf"-\s*{re.escape(object_type)}:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, obs_text or "", re.IGNORECASE)
    if not match:
        return positions

    coords_str = match.group(1)
    coord_matches = re.findall(r"\((-?\d+),\s*(-?\d+)(?:,\s*-?\d+)?\)", coords_str)
    for x, y in coord_matches:
        positions.append(Position(int(x), int(y)))
    return positions


def parse_monsters(obs_text: str) -> dict[str, list[Position]]:
    """
    解析所有 `- Monster ...:` 行，返回 {monster_name -> [Position, ...]}。

    目标：
    - 兼容已知的 Goomba/Koopa
    - 对未知怪物也能解析并纳入后续威胁/风险逻辑（不再“看不见”）

    说明：
    - monster_name 会做轻量归一化：
      - 含 goomba -> Goomba
      - 含 koopa -> Koopa
      - 其他保持原样（去除首尾空白）
    """
    text = str(obs_text or "")
    out: dict[str, list[Position]] = {}

    # 逐行匹配，避免跨行误伤；坐标串可能是 "None" 或 "(x,y), (x,y), ..."
    pattern = r"^\s*-\s*Monster\s+([^:]+):\s*(.+?)\s*$"
    for match in re.finditer(pattern, text, flags=re.IGNORECASE | re.MULTILINE):
        raw_name = (match.group(1) or "").strip()
        coords_str = (match.group(2) or "").strip()
        if not raw_name:
            continue

        name_norm = raw_name
        lower = raw_name.lower()
        if "goomba" in lower:
            name_norm = "Goomba"
        elif "koopa" in lower:
            name_norm = "Koopa"

        if re.search(r"\bNone\b", coords_str, flags=re.IGNORECASE):
            out.setdefault(name_norm, [])
            continue

        coord_matches = re.findall(r"\((-?\d+),\s*(-?\d+)(?:,\s*-?\d+)?\)", coords_str)
        positions: list[Position] = [Position(int(x), int(y)) for x, y in coord_matches]
        if not positions:
            # 兜底：有些格式可能只给纯数字
            nums = re.findall(r"-?\d+", coords_str)
            if len(nums) >= 2:
                try:
                    positions = [Position(int(nums[0]), int(nums[1]))]
                except Exception:
                    positions = []

        if positions:
            out.setdefault(name_norm, []).extend(positions)
        else:
            out.setdefault(name_norm, [])

    return out


def parse_pipes(obs_text: str) -> list[PipeInfo]:
    pipes: list[PipeInfo] = []
    pattern = r"-\s*Warp Pipe:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, obs_text or "", re.IGNORECASE)
    if not match:
        return pipes

    coords_str = match.group(1)
    coord_matches_3 = re.findall(r"\((\d+),\s*(\d+),\s*(\d+)\)", coords_str)
    for x, y, h in coord_matches_3:
        pipes.append(PipeInfo(int(x), int(y), int(h)))

    if not coord_matches_3:
        coord_matches_2 = re.findall(r"\((\d+),\s*(\d+)\)", coords_str)
        for x, y in coord_matches_2:
            pipes.append(PipeInfo(int(x), int(y), 47))
    return pipes


def parse_pit(obs_text: str) -> tuple[int, int] | None:
    """
    解析坑的位置。
    支持三种格式：
    1. 旧格式: "Pit: start at 171, end at 204"
    2. 完整坐标: "Pit: start at (171,31), (171,15), end at (204,32), (204,16)"
    3. 部分坐标: "Pit: start at (220,31), (220,15), end at None"
    """
    DEFAULT_PIT_WIDTH = 40

    text = str(obs_text or "")
    mario_pos = parse_mario_position(text)
    mario_x = int(mario_pos.x) if mario_pos is not None else None

    # 尽量从单行里解析（to_text 会把 pit_1start 与 pit_2end 拼到同一行）。
    pit_line_match = re.search(r"-\s*Pit:\s*start at\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    pit_line = pit_line_match.group(1) if pit_line_match else ""

    start_part = pit_line
    end_part = ""
    if re.search(r"\bend at\b", pit_line, re.IGNORECASE):
        parts = re.split(r"\bend at\b", pit_line, maxsplit=1, flags=re.IGNORECASE)
        start_part = parts[0]
        end_part = parts[1] if len(parts) > 1 else ""

    def _extract_xs(part: str) -> list[int]:
        if not part:
            return []
        if re.search(r"\bNone\b", part, re.IGNORECASE):
            return []
        xs = [int(x) for x in re.findall(r"\((\d+),\s*\d+(?:,\s*\d+)?\)", part)]
        if xs:
            return xs
        # 兼容旧格式：start/end 是纯数字
        nums = [int(x) for x in re.findall(r"\d+", part)]
        return nums[:1]

    start_xs = sorted({int(x) for x in _extract_xs(start_part)})
    end_xs = sorted({int(x) for x in _extract_xs(end_part)})

    if not start_xs:
        return None

    # 如果在屏幕里能定位 Mario，则优先选取“在 Mario 右侧的最近坑”。
    start_candidates = [x for x in start_xs if (mario_x is None or x > mario_x)]
    if not start_candidates:
        start_candidates = start_xs

    start_x = int(min(start_candidates))
    if end_xs:
        end_candidates = [x for x in end_xs if x > start_x]
        if end_candidates:
            end_x = int(min(end_candidates))
            # Heuristic: in RandomStages / castle lava, template matching for pit-end can spuriously
            # latch onto the screen's right edge, producing an unrealistically wide pit (often > 80px).
            # Treat such cases as "end not reliably observed" and fall back to a conservative default width.
            if int(end_x) >= 250 and int(end_x - start_x) >= 80:
                return (start_x, start_x + DEFAULT_PIT_WIDTH)
            return (start_x, int(end_x))

    return (start_x, start_x + DEFAULT_PIT_WIDTH)
