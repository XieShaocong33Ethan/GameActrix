from __future__ import annotations

from agents.crowsphere.mario_jump_rules import (
    ENEMY_CLUSTER_CLEARANCE_PX,
    ENEMY_CLUSTER_LOOKAHEAD_PX,
    JUMP_PARAMS,
    MARIO_SPEED_PER_STEP,
    POST_LANDING_RUN_PX,
    TAKEOFF_COLLISION_THRESHOLD_PX,
    detect_ground_enemy_cluster,
    enemy_width_px,
    estimate_enemy_approach_during_macro,
    estimate_landing_danger,
    min_jump_level_for_total_dx,
    select_safest_jump_level,
    is_enemy_threat_name,
)
from agents.crowsphere.mario_preprocess.types import ThreatInfo


def choose_jump_for_enemy_cluster(
    *,
    nearest_threat: ThreatInfo | None,
    all_threats: list[ThreatInfo] | None,
    has_low_ceiling: bool,
) -> tuple[int, str, str] | None:
    if nearest_threat is None:
        return None

    cluster = detect_ground_enemy_cluster(all_threats or [])
    if not (is_enemy_threat_name(str(nearest_threat.name)) and cluster and cluster[0].distance <= 80):
        return None

    max_level = 4 if has_low_ceiling else 6
    max_total_dx = int(JUMP_PARAMS[max_level][0]) + int(POST_LANDING_RUN_PX)

    cluster_end = cluster[-1].distance
    last_width = enemy_width_px(cluster[-1].name)
    raw_total_dx_needed = int(cluster_end) + int(last_width) + int(ENEMY_CLUSTER_CLEARANCE_PX)
    # Non-low-ceiling + dense clusters:
    # When the visible cluster appears longer than our max jump, committing to a long jump too early
    # can land Mario into unseen/occluded enemies (x≈1900~2000). Prefer a short "approach" phase
    # until the first enemy becomes close enough to force a commit (<=~30px).
    if (not has_low_ceiling) and (int(raw_total_dx_needed) > int(max_total_dx)) and (float(cluster[0].distance) >= 30.0):
        return (
            0,
            f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), too long for max jump, approach first",
            "MEDIUM",
        )
    total_dx_needed = int(raw_total_dx_needed)
    if not has_low_ceiling:
        # 敌人会在宏动作期间向左逼近；当“静态估算”显示需要的总位移明显超出 max_total_dx 时，
        # 原逻辑会输出 jl=0 逼近，容易在 x≈1900~2000 段走进贴脸必死态。
        # 仅在这种“明显超限”的情况下，引入一个上限为 15px 的逼近允许量，提前触发清簇跳。
        over_by = int(raw_total_dx_needed) - int(max_total_dx)
        if int(over_by) >= 6:
            approach_allowance_px = min(int(estimate_enemy_approach_during_macro(int(max_level))), 15)
            total_dx_needed = max(0, int(raw_total_dx_needed) - int(approach_allowance_px))

    if has_low_ceiling and len(cluster) >= 2 and cluster[0].distance <= 55:
        if has_low_ceiling:
            # 低天花板 + 连续敌人：反应窗口更紧，过度“逼近”很容易直接进入贴脸必死态。
            # 因此把 CRITICAL 的阈值放宽到 ~60px，用于触发 adapter 的确定性“非 0 候选优先”仲裁。
            if float(cluster[0].distance) <= 60.0:
                urgency = "CRITICAL"
            elif float(cluster[0].distance) <= 75.0:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"
        else:
            urgency = "MEDIUM"
            if cluster[0].distance < 30:
                urgency = "CRITICAL"
            elif cluster[0].distance < 50:
                urgency = "HIGH"
        # 回归：warmup=21 在 x≈2005 的低天花板段，最近 Goomba 距离≈14~16px 时，
        # lvl2 的起跳阶段更容易发生侧向碰撞；优先用更短的 lvl1 及时离地。
        # 回归：warmup=21 在 x≈2005 的低天花板段，双 Goomba 会出现“贴脸（~16px）”窗口。
        # 在该窗口里，lvl1/lvl2 往往会在起跳早期发生侧向擦边碰撞；
        # 更稳的做法是用更高的短程跳（优先 lvl3），尽快获得足够的垂直裕量跨过贴脸敌人。
        if float(cluster[0].distance) <= 20.0:
            split_level = min(3, int(max_level))
        else:
            split_level = min(2, int(max_level))
        return (
            int(split_level),
            f"enemy cluster x{len(cluster)} under low ceiling, split with lvl{int(split_level)}",
            urgency,
        )

    if (not has_low_ceiling) and (max_level >= 6) and (float(cluster[0].distance) < 30.0):
        # 非低天花板 + CRITICAL 密集敌人：优先提供“一步清簇”的大跳（lvl6）。
        # 经验：若仍选择小跳/慢走，常会被推入后续贴脸必死态（官方 eval x≈1900~2000 段）。
        # 但当最近敌人过近时，起跳阶段可能发生侧向碰撞；因此要求一定的起跳跑道。
        min_takeoff_runway_px = int(TAKEOFF_COLLISION_THRESHOLD_PX) + 4
        if float(cluster[0].distance) >= float(min_takeoff_runway_px):
            return (
                int(max_level),
                f"enemy cluster x{len(cluster)} critical, clear with lvl{int(max_level)}",
                "CRITICAL",
            )

    chosen: int | None
    if total_dx_needed <= max_total_dx:
        chosen = min_jump_level_for_total_dx(total_dx_needed, max_level=max_level)
    else:
        chosen = None

    # Non-low-ceiling dense clusters: in the 30~60px window, committing to a (seemingly) viable
    # clear jump can still be unstable and die (occluded enemies / tight edge clearance).
    # Prefer one-step approach until it becomes CRITICAL (<~35px), then commit.
    if (
        chosen is not None
        and (not has_low_ceiling)
        and len(cluster) >= 2
        and 35.0 <= float(cluster[0].distance) < 60.0
    ):
        chosen_total_dx = int(JUMP_PARAMS[int(chosen)][0]) + int(POST_LANDING_RUN_PX)
        slack = int(chosen_total_dx) - int(raw_total_dx_needed)
        is_long = int(chosen) >= 5
        is_tight = int(slack) <= 3
        if is_long or is_tight:
            return (
                0,
                f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), approach to enter stable clear window",
                "MEDIUM",
            )

    if chosen is None:
        # 非低天花板场景：敌人簇较远时先逼近（更稳）。
        # 低天花板场景：逼近往往会把自己送入“贴脸+无跳窗”态，因此即便稍远也倾向先做一次分割跳。
        if (not has_low_ceiling) and float(cluster[0].distance) >= 55.0:
            return (
                0,
                f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), too long, approach",
                "LOW",
            )

        if has_low_ceiling:
            # 低天花板 + 连续敌人：反应窗口更紧，过度“逼近”很容易直接进入贴脸必死态。
            # 因此把 CRITICAL 的阈值放宽到 ~60px，用于触发 adapter 的确定性“非 0 候选优先”仲裁。
            if float(cluster[0].distance) <= 60.0:
                urgency = "CRITICAL"
            elif float(cluster[0].distance) <= 75.0:
                urgency = "HIGH"
            else:
                urgency = "MEDIUM"
        else:
            urgency = "MEDIUM"
            if cluster[0].distance < 30:
                urgency = "CRITICAL"
            elif cluster[0].distance < 50:
                urgency = "HIGH"

        split_level: int | None = None
        for cand in range(1, int(max_level) + 1):
            is_dangerous, _, _ = estimate_landing_danger(cand, cluster)
            if not is_dangerous:
                split_level = int(cand)
                break
        if split_level is None:
            return (
                0,
                f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), too long, approach",
                urgency,
            )
        return (
            int(split_level),
            f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), too long, split with lvl{int(split_level)}",
            urgency,
        )

    if chosen >= 5 and cluster[0].distance > 50:
        shorter = int(chosen) - 1
        shorter_total_dx = int(JUMP_PARAMS[shorter][0]) + int(POST_LANDING_RUN_PX)
        if shorter_total_dx >= int(total_dx_needed) - int(MARIO_SPEED_PER_STEP):
            urgency = "MEDIUM"
            # 非低天花板场景下，如果“敌人簇”已经很近且几乎吃满可用位移，
            # 则需要把 urgency 拉到 CRITICAL，避免 adapter 继续偏向 jl=0 逼近而送命。
            if (not has_low_ceiling) and float(cluster[0].distance) <= 55.0 and int(total_dx_needed) >= int(max_total_dx) - 4:
                urgency = "CRITICAL"
            return (
                int(shorter),
                f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), use lvl{int(shorter)}",
                urgency,
            )

    urgency = "MEDIUM"
    if cluster[0].distance < 30:
        urgency = "CRITICAL"
    elif cluster[0].distance < 50:
        urgency = "HIGH"

    if has_low_ceiling:
        safe_level, rsn = select_safest_jump_level(
            int(chosen),
            cluster,
            max_level=int(max_level),
            allow_wait=True,
        )
        suffix = f", {rsn}" if rsn else ""
        return (
            int(safe_level),
            f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), need_dx≈{int(total_dx_needed)}px{suffix}",
            urgency,
        )

    return (
        int(chosen),
        f"enemy cluster x{len(cluster)} (end={int(cluster_end)}px), need_dx≈{int(total_dx_needed)}px",
        urgency,
    )
