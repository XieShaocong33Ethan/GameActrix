from __future__ import annotations

from agents.crowsphere.mario_jump_rules import JUMP_PARAMS
from agents.crowsphere.mario_preprocess.types import Position


def level_for_height(height: int) -> int:
    # 跳跃高度: 1->35, 2->46, 3->53, 4->60, 5->65, 6->68
    if height <= 30:
        return 1
    if height <= 40:
        return 2
    if height <= 50:
        return 3
    if height <= 58:
        return 4
    if height <= 63:
        return 5
    return 6


def level_for_distance(dist: int) -> int:
    # JUMP_PARAMS 水平距离: 1->42, 2->56, 3->63, 4->70, 5->77, 6->84
    # 该函数主要用于实体障碍（Pipe/Stairs）的“需要跨越的水平距离”估计。
    #
    # 实测上，5px 的保守余量会在 Pipe+Enemy 等窗口里把 level3 过度抬到 level4，
    # 导致“越过 Pipe 后落点太远”→落在敌人附近/进入必死窗。
    #
    # 因此这里把余量收紧到 2px，让 SAFE 更倾向“刚好能清障”的短跳。
    margin = 2
    if dist <= 42 - margin:
        return 1
    if dist <= 56 - margin:
        return 2
    if dist <= 63 - margin:
        return 3
    if dist <= 70 - margin:
        return 4
    if dist <= 77 - margin:
        return 5
    return 6


def estimate_ceiling_collision(
    *,
    jump_level: int,
    mario_pos: Position | None,
    ceiling_blocks: list[Position] | None,
    lookbehind_px: int = 40,
    ceiling_tile_h: int = 16,
    clip_threshold_px: int = 6,
    ceiling_x_margin: int = 4,
) -> tuple[bool, int | None, int | None]:
    """
    估计某个 jump_level 是否会“顶到低天花板”（HIT_CEILING）。

    返回：(would_hit, block_dx, block_y)
    - block_dx: ceiling block 相对 Mario 的 dx（像素）
    - block_y: ceiling block 的 y（像素）
    """
    jl = max(0, min(6, int(jump_level)))
    if jl <= 0 or mario_pos is None or not ceiling_blocks:
        return False, None, None

    # 模板匹配在低天花板段会有一定的 x 抖动，偶尔会把“头顶砖块”检测到 Mario 身后。
    # 但如果只检测到 1 块且它明显在身后（<= -17px，约 1 块砖宽），
    # 更可能是“刚走过的单砖/噪声”，继续把它当作顶头风险会导致：
    # - SAFE 被过度 clamp 到 lvl1
    # - 进而逼迫主 VLM 选择不稳定的 STOMP/慢走，最终贴脸撞死
    #
    # 因此：仅在“单块且明显身后”的情况下直接忽略该块的顶头判定。
    # 真实低天花板 corridor 通常会检测到多块砖（或至少有一块在前方），不受该规则影响。
    dxs = [int(b.x) - int(mario_pos.x) for b in ceiling_blocks]
    has_ahead = any(int(dx) >= 0 for dx in dxs)
    max_dx = max(int(dx) for dx in dxs) if dxs else 0
    # 当所有低砖都在身后且“明显远离”（<=-12px）时，大概率是模板抖动/刚走过的单砖，
    # 不应继续把它当作顶头风险（否则会在 warmup 偏移下错误 ban 掉可行跳法，诱导慢走撞死）。
    if (not has_ahead) and int(max_dx) <= -12:
        return False, None, None
    if len(dxs) == 1 and int(dxs[0]) <= -17:
        return False, None, None

    landing_dx = int(JUMP_PARAMS[jl][0])
    apex_y = int(mario_pos.y) + int(JUMP_PARAMS[jl][1])
    lookbehind_px_int = max(0, int(lookbehind_px))
    back_margin_px = 24  # 经验值：约 1.5 块砖宽度；覆盖低天花板 recall，避免 -30px 的远身后误报
    min_dx = -min(int(back_margin_px), int(lookbehind_px_int))

    for blk in sorted(ceiling_blocks, key=lambda p: int(p.x)):
        dx = int(blk.x) - int(mario_pos.x)
        if dx < int(min_dx):
            continue
        if dx > int(landing_dx) + int(ceiling_x_margin):
            break
        underside_y = int(blk.y) - int(ceiling_tile_h)
        clip_px = int(apex_y) - int(underside_y)
        if clip_px >= int(clip_threshold_px):
            return True, int(dx), int(blk.y)

    return False, None, None
