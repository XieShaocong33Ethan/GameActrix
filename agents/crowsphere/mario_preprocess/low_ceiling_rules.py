from __future__ import annotations

from dataclasses import dataclass

from agents.crowsphere.mario_jump_rules import is_enemy_threat_name
from agents.crowsphere.mario_preprocess.low_ceiling_memory import LowCeilingMemory
from agents.crowsphere.mario_preprocess.sentinel import VLMSentinelInfo
from agents.crowsphere.mario_preprocess.types import Position, ThreatInfo


@dataclass(frozen=True)
class LowCeilingDetectionResult:
    has_low_ceiling: bool
    max_jump_level: int


def detect_low_ceiling_and_max_jump_level(
    *,
    mario_pos: Position | None,
    bricks: list[Position] | None,
    question_blocks: list[Position] | None,
    threats: list[ThreatInfo] | None,
    vlm_sentinel: VLMSentinelInfo,
    ceiling_memory: LowCeilingMemory | None,
    env_x_pos: int | None,
    stair_blocks: list[Position] | None,
) -> LowCeilingDetectionResult:
    """
    检测 low-ceiling corridor，并给出 max_jump_level（4/6）。

    设计目标：
    - 在真实 low-ceiling 段尽快 clamp（避免 lvl5/6 顶头或起跳侧碰撞导致稳定死亡）
    - 但避免被“仅身后残留的一块低砖”长期误导（离开 corridor 后仍保持 clamp，会丢失后续清怪/跨坑窗口）

    注意：官方模板匹配坐标与 Mario 的屏幕坐标（x=122 clamp）在部分窗口会出现 ~14px 相对偏移。
    为了避免错判，我们把“贴身身后”阈值从 -16 扩到 -32（约 2 块砖宽）。
    """

    strong_seen = False
    weak_seen = False
    low_ceiling_behind_only = False
    closest_low_ceiling_dx: int | None = None
    low_dxs: list[int] = []
    has_ahead = False
    has_near_behind = False

    if mario_pos is not None:
        blocks = list(bricks or []) + list(question_blocks or [])
        for block in blocks:
            dx = int(block.x) - int(mario_pos.x)
            if -40 <= int(dx) <= 80 and 80 < int(block.y) < 130:
                low_dxs.append(int(dx))

        has_ahead = any(int(dx) >= 0 for dx in low_dxs)

        # “贴身身后”的阈值（用于离开 corridor 后的残留过滤）
        has_near_behind = any(-16 <= int(dx) <= -1 for dx in low_dxs)
        far_behind_count = sum(1 for dx in low_dxs if -24 <= int(dx) <= -17)
        weak_far_behind = any(-40 <= int(dx) <= -25 for dx in low_dxs) or (
            (far_behind_count == 1) and (not has_ahead) and (not has_near_behind)
        )

        strong_seen = bool(has_ahead or has_near_behind or (far_behind_count >= 2))
        weak_seen = bool(weak_far_behind)
        if low_dxs and (not has_ahead):
            closest_low_ceiling_dx = int(max(low_dxs))
            # <= -12px：明显在身后（大约 3/4 块砖宽），更像“残留/抖动”而非仍在 corridor。
            low_ceiling_behind_only = bool(int(closest_low_ceiling_dx) <= -12)

    # sentinel 明确标注时直接视为 strong（最高优先级证据）
    strong_seen = bool(getattr(vlm_sentinel, "low_ceiling", False)) or bool(strong_seen)

    if ceiling_memory is not None:
        ceiling_memory.update(strong_seen=strong_seen, weak_seen=weak_seen)
        has_low_ceiling = bool(strong_seen or ceiling_memory.active)
    else:
        has_low_ceiling = bool(strong_seen)
    max_jump_level = 4 if has_low_ceiling else 6

    dense_near_count = sum(
        1 for t in (threats or []) if is_enemy_threat_name(str(t.name)) and 0 < float(t.distance) <= 120
    )

    # 经验修正：low-ceiling 记忆可能在离开 corridor 后短暂残留（ttl_steps），
    # 若本步完全没有低砖证据且 VLM sentinel 也不认为是 low-ceiling，但前方出现密集敌人，
    # 继续 clamp 到 4 会丢失关键的清怪/逃生大跳窗口。
    if (
        has_low_ceiling
        and (not strong_seen)
        and (not weak_seen)
        and (not low_dxs)
        and dense_near_count >= 2
        and (ceiling_memory is not None)
        and bool(getattr(ceiling_memory, "active", False))
        and bool(getattr(vlm_sentinel, "present", False))
        and (not getattr(vlm_sentinel, "low_ceiling", False))
    ):
        has_low_ceiling = False
        max_jump_level = 6

    # weak_budget 耗尽后：如果仍只有 behind-only 弱证据，并且前方密集敌人（>=2），则解除 clamp。
    if (
        has_low_ceiling
        and (not strong_seen)
        and low_ceiling_behind_only
        and (closest_low_ceiling_dx is not None)
        and int(closest_low_ceiling_dx) <= -17
        and (ceiling_memory is not None)
        and bool(getattr(ceiling_memory, "active", False))
        and int(getattr(ceiling_memory, "weak_refresh_budget", 0)) <= 0
        and dense_near_count >= 2
    ):
        has_low_ceiling = False
        max_jump_level = 6

    # VLM 明确否认 low-ceiling 时：只在“轻微身后残留”（dx≈-12~-16）时解除 clamp。
    # 若低砖已明显在身后（例如 dx<=-17~-32），更可能是坐标漂移/仍在 corridor；
    # 此时放开 lvl5/6 会造成稳定顶头/贴脸死亡。
    if (
        has_low_ceiling
        and low_ceiling_behind_only
        and (not has_ahead)
        and (len(low_dxs) == 1)
        and dense_near_count >= 2
        and bool(getattr(vlm_sentinel, "present", False))
        and (not getattr(vlm_sentinel, "low_ceiling", False))
        and (
            bool(getattr(vlm_sentinel, "dense_enemies_ahead", False))
            or (getattr(vlm_sentinel, "enemy_count_ahead", "unknown") == "two_plus")
        )
    ):
        has_low_ceiling = False
        max_jump_level = 6
        if ceiling_memory is not None:
            ceiling_memory.reset()

    # 经验修正：在关卡后段台阶区域（x≈2100+），若当前帧完全没有 low-ceiling 证据，
    # 直接清空该记忆，避免 max_jump_level=4 导致后续跨坑/跨台阶窗口丢失。
    if (
        has_low_ceiling
        and (env_x_pos is not None)
        and int(env_x_pos) >= 2100
        and (not strong_seen)
        and (not weak_seen)
        and (not low_dxs)
        and bool(stair_blocks)
    ):
        has_low_ceiling = False
        max_jump_level = 6
        if ceiling_memory is not None:
            ceiling_memory.reset()

    return LowCeilingDetectionResult(
        has_low_ceiling=bool(has_low_ceiling),
        max_jump_level=int(max_jump_level),
    )
