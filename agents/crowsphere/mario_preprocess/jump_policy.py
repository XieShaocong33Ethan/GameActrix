from __future__ import annotations

from agents.crowsphere.mario_jump_rules import estimate_has_low_ceiling, is_enemy_threat_name
from agents.crowsphere.mario_preprocess.jump_policy_enemy_cluster import choose_jump_for_enemy_cluster
from agents.crowsphere.mario_preprocess.jump_policy_pit import choose_jump_for_pit
from agents.crowsphere.mario_preprocess.jump_policy_threats import choose_jump_for_nearest_threat
from agents.crowsphere.mario_preprocess.sentinel import VLMSentinelInfo
from agents.crowsphere.mario_preprocess.types import Position, ThreatInfo


def calculate_jump_level(
    *,
    nearest_threat: ThreatInfo | None,
    pit_info: tuple[int, int] | None,
    mario_pos: Position | None,
    bricks: list[Position] | None,
    question_blocks: list[Position] | None,
    stair_block_count: int,
    all_threats: list[ThreatInfo] | None,
    vlm_sentinel: VLMSentinelInfo | None,
    on_stairs: bool,
    has_low_ceiling_override: bool | None = None,
) -> tuple[int, str, str]:
    # 前瞻性低天花板检测：也检测身后 40px 内的砖块
    has_low_ceiling = estimate_has_low_ceiling(mario_pos, bricks, question_blocks, lookbehind_px=40)
    if has_low_ceiling_override is not None:
        has_low_ceiling = bool(has_low_ceiling_override)

    # ============= Height-diff warning (VLM + deterministic fallback) =============
    sentinel_height_diff_warning = False
    ground_enemies = [
        t
        for t in (all_threats or [])
        if is_enemy_threat_name(str(t.name)) and t.distance > 0
    ]

    if vlm_sentinel is not None and vlm_sentinel.present:
        if vlm_sentinel.on_platform and vlm_sentinel.enemy_below_ahead and ground_enemies:
            sentinel_height_diff_warning = True

    if (not sentinel_height_diff_warning) and ground_enemies:
        near_solid = any(
            (t.name in {"Pipe", "Stairs"}) and (0 < t.distance <= 10) for t in (all_threats or [])
        )
        enemy_ahead_close = any(t.distance <= 140 for t in ground_enemies)
        if near_solid and enemy_ahead_close:
            sentinel_height_diff_warning = True
    # ============= End height-diff warning =============

    pit_choice = choose_jump_for_pit(
        pit_info=pit_info,
        mario_pos=mario_pos,
        nearest_threat=nearest_threat,
        all_threats=all_threats,
        on_stairs=bool(on_stairs),
        stair_block_count=int(stair_block_count),
    )
    if pit_choice is not None:
        return pit_choice

    cluster_choice = choose_jump_for_enemy_cluster(
        nearest_threat=nearest_threat,
        all_threats=all_threats,
        has_low_ceiling=bool(has_low_ceiling),
    )
    if cluster_choice is not None:
        return cluster_choice

    threat_choice = choose_jump_for_nearest_threat(
        nearest_threat=nearest_threat,
        pit_info=pit_info,
        mario_pos=mario_pos,
        bricks=bricks,
        question_blocks=question_blocks,
        all_threats=all_threats,
        sentinel_height_diff_warning=bool(sentinel_height_diff_warning),
        has_low_ceiling_override=bool(has_low_ceiling),
    )
    if threat_choice is not None:
        return threat_choice

    # 无威胁时：保守策略，避免开环跳跃进入未知区域
    has_overhead = False
    if mario_pos and (bricks or question_blocks):
        all_blocks = (bricks or []) + (question_blocks or [])
        for block in all_blocks:
            if abs(block.x - mario_pos.x) < 60 and 80 < block.y < 130:
                has_overhead = True
                break

    if has_overhead:
        return (1, "overhead obstacle, small jump to advance", "NONE")
    return (3, "no threats, jump to advance faster", "NONE")
