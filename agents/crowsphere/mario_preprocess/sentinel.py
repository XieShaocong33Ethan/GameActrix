from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class VLMSentinelInfo:
    """解析后的 VLM Sentinel 信息"""

    on_platform: bool = False  # Mario 是否在高处（管道/砖块/台阶）
    enemy_below_ahead: bool = False  # 前方低处是否有地面敌人
    enemy_ahead: bool = False  # 前方是否有敌人
    enemy_count_ahead: str = "unknown"  # none/one/two_plus/unknown
    dense_enemies_ahead: bool = False  # 前方是否密集敌人（>=2 且距离近）
    pipe_ahead: bool = False  # 前方是否有管道
    pipe_distance: str = "unknown"  # very_close/close/mid/far/unknown
    pit_ahead: bool = False  # 前方是否有坑
    pit_distance: str = "unknown"  # very_close/close/mid/far/unknown
    pit_start_dx: int | None = None  # 屏幕坐标：pit 起点相对 Mario 的 dx（像素）
    pit_end_dx: int | None = None  # 屏幕坐标：pit 终点相对 Mario 的 dx（像素）
    stairs_ahead: bool = False  # 前方是否有台阶
    stairs_pit_ahead: bool = False  # 台阶后紧跟坑
    low_ceiling: bool = False  # 低天花板（视觉判断）
    present: bool = False  # obs_text 中是否包含 sentinel 块


def _slice_block(obs_text: str, header: str) -> str:
    text = str(obs_text or "")
    if header not in text:
        return ""
    block = text.split(header, 1)[1]
    m = re.search(r"\n\[[^\]]+\]", block)
    if m:
        block = block[: m.start()]
    return block


def parse_vlm_sentinel_block(obs_text: str) -> VLMSentinelInfo:
    """
    解析 [VLM Sentinel] 块，提取布尔值。

    格式示例：
    [VLM Sentinel]
    on_platform: True
    enemy_below_ahead: True
    enemy_ahead: False
    """
    result = VLMSentinelInfo()

    vlm_block = _slice_block(obs_text, "[VLM Sentinel]")
    pit_block = _slice_block(obs_text, "[Mario Pit Detector]")

    if (not vlm_block) and (not pit_block):
        return result

    result.present = True

    def parse_bool(text: str, pattern: str) -> bool:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip().lower()
            return val in ("true", "yes", "1")
        return False

    def parse_bucket(text: str, pattern: str) -> str:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            val = match.group(1).strip().lower()
            if val in {"very_close", "close", "mid", "far", "unknown"}:
                return val
        return "unknown"

    def parse_int(text: str, pattern: str) -> int | None:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    # 1) VLM Sentinel block
    if vlm_block:
        result.on_platform = parse_bool(vlm_block, r"on_platform:\s*(true|false|yes|no|1|0)")
        result.enemy_below_ahead = parse_bool(vlm_block, r"enemy_below_ahead:\s*(true|false|yes|no|1|0)")
        result.enemy_ahead = parse_bool(vlm_block, r"enemy_ahead:\s*(true|false|yes|no|1|0)")
        match = re.search(r"enemy_count_ahead:\s*(none|one|two_plus|unknown)", vlm_block, re.IGNORECASE)
        if match:
            result.enemy_count_ahead = match.group(1).strip().lower()
        result.dense_enemies_ahead = parse_bool(vlm_block, r"dense_enemies_ahead:\s*(true|false|yes|no|1|0)")
        result.pipe_ahead = parse_bool(vlm_block, r"pipe_ahead:\s*(true|false|yes|no|1|0)")
        result.pit_ahead = parse_bool(vlm_block, r"pit_ahead:\s*(true|false|yes|no|1|0)")
        result.stairs_ahead = parse_bool(vlm_block, r"stairs_ahead:\s*(true|false|yes|no|1|0)")
        result.stairs_pit_ahead = parse_bool(vlm_block, r"stairs_pit_ahead:\s*(true|false|yes|no|1|0)")
        result.low_ceiling = parse_bool(vlm_block, r"low_ceiling:\s*(true|false|yes|no|1|0)")
        result.pipe_distance = parse_bucket(vlm_block, r"pipe_distance:\s*(very_close|close|mid|far|unknown)")
        result.pit_distance = parse_bucket(vlm_block, r"pit_distance:\s*(very_close|close|mid|far|unknown)")

    # 2) Deterministic pit detector block (merge into sentinel evidence)
    if pit_block:
        det_pit_ahead = parse_bool(pit_block, r"pit_ahead:\s*(true|false|yes|no|1|0)")
        det_pit_distance = parse_bucket(pit_block, r"pit_distance:\s*(very_close|close|mid|far|unknown)")
        det_start = parse_int(pit_block, r"pit_start_dx:\s*(-?\d+)")
        det_end = parse_int(pit_block, r"pit_end_dx:\s*(-?\d+)")

        if det_pit_ahead:
            result.pit_ahead = True
            if det_pit_distance != "unknown":
                result.pit_distance = det_pit_distance
            if det_start is not None:
                result.pit_start_dx = det_start
            if det_end is not None:
                result.pit_end_dx = det_end

    return result
