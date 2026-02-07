from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.mario_jump_rules import (
    ENEMY_CLUSTER_CLEARANCE_PX,
    ENEMY_CLUSTER_GAP_PX,
    JUMP_PARAMS,
    LANDING_DANGER_MARGIN,
    MARIO_SPEED_PER_STEP,
    ENEMY_SPEED_PER_FRAME,
    POST_LANDING_RUN_PX,
    TAKEOFF_COLLISION_THRESHOLD_PX,
    detect_ground_enemy_cluster,
    enemy_width_px,
    estimate_enemy_approach_during_macro,
    estimate_landing_danger,
    estimate_landing_displacement,
    is_enemy_threat_name,
    is_stompable_enemy_name,
)
from agents.crowsphere.mario_preprocess.candidates import MarioCandidates
from agents.crowsphere.mario_preprocess.jump_level_helpers import estimate_ceiling_collision, level_for_height
from agents.crowsphere.mario_preprocess.state import MarioPreprocessState
from agents.crowsphere.mario_preprocess.types import ThreatInfo


PIT_MIN_RUNWAY_PX = 16
PIT_MIN_RUNWAY_NEAR_STAIRS_PX = 4
# 台阶坑：要求落点越过坑尾的最小裕量（过严会“等到太晚才跳”掉坑；过松会落短）。
STAIRS_PIT_END_CLEARANCE_MIN_PX = 9


def build_risk_entries(state: MarioPreprocessState, candidates: MarioCandidates) -> list[dict[str, Any]]:
    mario_pos = state.mario_pos
    threats = state.threats
    pit_info = state.pit_info
    max_jump_level = int(state.max_jump_level)
    ceiling_blocks = list(state.ceiling_blocks or [])
    on_stairs = bool(state.on_stairs)
    ground_y_est = int(state.ground_y_est)
    urgency = str(state.urgency or "").upper()
    has_low_ceiling = bool(state.has_low_ceiling)
    vlm_sentinel = state.vlm_sentinel
    enemy_cluster = detect_ground_enemy_cluster(threats)
    cluster_required_total_dx: int | None = None
    if enemy_cluster and pit_info is None:
        last = enemy_cluster[-1]
        cluster_required_total_dx = (
            int(last.distance) + int(enemy_width_px(str(last.name))) + int(ENEMY_CLUSTER_CLEARANCE_PX)
        )

    def _solid_clearance_level(threat: ThreatInfo) -> int:
        raw_height = int(threat.height)
        if mario_pos is None:
            return level_for_height(raw_height)
        elevation = max(0, int(mario_pos.y) - int(ground_y_est))
        effective_height = max(0, raw_height - elevation)
        if int(effective_height) <= 0:
            return 0
        return level_for_height(int(effective_height))

    def _risk_report_for_jump_level(jl: int) -> dict[str, Any]:
        jl = max(0, min(int(jl), max_jump_level))
        total_dx = int(estimate_landing_displacement(jl)) if jl > 0 else int(MARIO_SPEED_PER_STEP)
        landing_dx = int(JUMP_PARAMS[jl][0]) if jl > 0 else 0

        enemy_approach_margin = int(estimate_enemy_approach_during_macro(jl))

        # Low ceiling collision (HIT_CEILING)
        would_hit_ceiling, ceiling_block_dist, ceiling_block_y = estimate_ceiling_collision(
            jump_level=int(jl),
            mario_pos=mario_pos,
            ceiling_blocks=ceiling_blocks,
        )

        # TAKEOFF_COLLISION
        takeoff_collision = False
        takeoff_enemy_dist: int | None = None
        if jl > 0:
            # In low-ceiling segments, higher jump levels tend to have a longer "takeoff" phase
            # before Mario gains vertical clearance. Treat slightly farther enemies as a
            # takeoff side-collision risk to avoid unstable deaths (e.g. x≈2000, enemy@14px).
            takeoff_threshold_px = int(TAKEOFF_COLLISION_THRESHOLD_PX)
            if bool(has_low_ceiling) and int(jl) >= 2 and str(urgency) in {"HIGH", "CRITICAL"}:
                takeoff_threshold_px = max(int(takeoff_threshold_px), int(TAKEOFF_COLLISION_THRESHOLD_PX) + 2 * (int(jl) - 1))
            for t in threats:
                if not is_enemy_threat_name(str(t.name)):
                    continue
                if 0 < t.distance <= float(takeoff_threshold_px):
                    takeoff_collision = True
                    takeoff_enemy_dist = int(t.distance)
                    break

        # Solid collision
        would_hit_solid = False
        solid_name = None
        solid_dist = None
        for t in threats:
            if t.name not in {"Pipe", "Stairs"}:
                continue
            if t.name == "Stairs" and jl == 0:
                continue
            if 0 < t.distance <= float(total_dx) + 10.0:
                need = _solid_clearance_level(t)
                if jl < int(need):
                    would_hit_solid = True
                    solid_name = t.name
                    solid_dist = int(t.distance)
                    break

        pit_landing = False
        pit_too_close_to_jump = False
        pit_edge_too_close_after = False
        pit_edge_dist_after: int | None = None
        pit_edge_near_after = False
        pit_edge_near_dist_after: int | None = None
        if pit_info and mario_pos:
            p0, p1 = pit_info[0] - mario_pos.x, pit_info[1] - mario_pos.x
            near_stairs = bool(on_stairs) or any(
                (t.name == "Stairs") and (0 < float(t.distance) <= 16.0) for t in threats
            )
            runway_min_px = PIT_MIN_RUNWAY_NEAR_STAIRS_PX if (near_stairs and int(mario_pos.y) >= 70) else PIT_MIN_RUNWAY_PX
            if jl > 0 and 0 < p0 < runway_min_px:
                pit_too_close_to_jump = True

            if (p0 < landing_dx < p1) and (p1 > 0):
                pit_landing = True

            # near_stairs 下，1-1 末段在台阶顶附近（y>=70）常会出现“落点到坑尾裕量”抖动；
            # 因此对这种情况禁用 run-in 检查，改由 PIT_EDGE_TOO_CLOSE 系列规则兜底。
            # 但在 RandomStages / castle lava 中，Mario 往往并不在台阶顶（y<70），
            # 此时落地后的 POST_LANDING_RUN_PX 仍会把 Mario 推进坑里，必须视为 LAND_IN_PIT。
            apply_run_in_check = (jl == 0) or (not near_stairs) or (int(mario_pos.y) < 70)
            if (not pit_landing) and apply_run_in_check:
                seg_start = landing_dx
                seg_end = total_dx
                if (p1 > seg_start) and (p0 < seg_end) and (p1 > 0):
                    pit_landing = True

            if jl > 0 and near_stairs and int(mario_pos.y) >= 70 and (p1 > 0) and (landing_dx > p1):
                end_clearance = landing_dx - p1
                can_delay = int(p0) > int(MARIO_SPEED_PER_STEP)
                if can_delay and 0 <= end_clearance < STAIRS_PIT_END_CLEARANCE_MIN_PX:
                    if (not pit_edge_too_close_after) or (pit_edge_dist_after is None) or (end_clearance < pit_edge_dist_after):
                        pit_edge_too_close_after = True
                        pit_edge_dist_after = int(end_clearance)

            dx_after_for_edge = landing_dx if near_stairs and jl > 0 else total_dx
            new_p0 = p0 - dx_after_for_edge
            if 0 <= new_p0 < PIT_MIN_RUNWAY_PX:
                if near_stairs and int(mario_pos.y) >= 70:
                    pit_edge_near_after = True
                    pit_edge_near_dist_after = int(new_p0)
                else:
                    pit_edge_too_close_after = True
                    pit_edge_dist_after = int(new_p0)

        is_dangerous, min_dist_after, danger_reason = estimate_landing_danger(jl, threats)
        # NOTE: estimate_landing_danger() returns the enemy with *minimum absolute* distance after the macro,
        # which can be a behind-enemy (negative) in dense scenes.
        # For the "with_approach" adjustment we must look at the closest enemy still *ahead* after the macro,
        # otherwise we may miss a trailing enemy and incorrectly label a move as OK.
        min_ahead_dist_after: float | None = None
        closest_ahead_enemy_name: str | None = None
        for t in threats:
            if not is_enemy_threat_name(str(t.name)):
                continue
            if float(t.distance) <= 0:
                continue
            dist_after = float(t.distance) - float(total_dx)
            if dist_after < 0:
                continue
            if (min_ahead_dist_after is None) or (dist_after < float(min_ahead_dist_after)):
                min_ahead_dist_after = float(dist_after)
                closest_ahead_enemy_name = str(t.name)
        # Dense scenes: run-in detector assumes stationary enemies; apply a limited correction to avoid false negatives.
        if (
            bool(is_dangerous)
            and isinstance(danger_reason, str)
            and danger_reason.startswith("landing_danger_run_in:")
            # 该修正只对长跳更可靠；短跳过松会把不稳定窗口标成 OK 并稳定死亡（x≈1900~2000）。
            and int(jl) >= 5
        ):
            ground_enemies = [t for t in threats if is_enemy_threat_name(str(t.name)) and float(t.distance) > 0]
            if len(ground_enemies) >= 2 and (urgency in {"HIGH", "CRITICAL"} or bool(enemy_cluster)):
                m = re.search(r"@(-?\d+)\s*px", danger_reason, flags=re.IGNORECASE)
                run_in_dist: float | None = None
                if m:
                    try:
                        run_in_dist = float(m.group(1))
                    except Exception:
                        run_in_dist = None
                if run_in_dist is not None:
                    adjusted = float(run_in_dist) - float(enemy_approach_margin)
                    if adjusted <= 0:
                        is_dangerous = False
                        danger_reason = f"safe_run_in_with_approach:{adjusted:.0f}px"
        if (not is_dangerous) and (min_ahead_dist_after is not None):
            adjusted_dist = float(min_ahead_dist_after) - float(enemy_approach_margin)
            if 0 < adjusted_dist < float(LANDING_DANGER_MARGIN):
                is_dangerous = True
                if closest_ahead_enemy_name:
                    danger_reason = f"landing_danger_with_approach:{closest_ahead_enemy_name}@{adjusted_dist:.0f}px"
                else:
                    danger_reason = f"landing_danger_with_approach:{adjusted_dist:.0f}px"

        # Dense enemy cluster: a "just barely clears" macro is unstable and often dies at the cluster edge.
        if (
            (not bool(is_dangerous))
            and (cluster_required_total_dx is not None)
            and (jl > 0)
            and (not has_low_ceiling)
        ):
            # Dense clusters: enemies are moving toward Mario during the macro action.
            # Treat a small amount of "shortfall" as OK by subtracting a conservative
            # cap of the enemy approach margin from the clearance requirement.
            #
            # IMPORTANT: only apply approach-credit to long jumps (>=5). For short jumps,
            # crediting enemy movement makes jl=2/3/4 look like it "clears" the cluster
            # even when the clearance is actually razor-thin and unstable (x≈1900~2000).
            approach_credit = 0
            if int(jl) >= 5:
                approach_credit = min(int(enemy_approach_margin), 18)
            effective_required = max(0, int(cluster_required_total_dx) - int(approach_credit))
            slack = int(total_dx) - int(effective_required)
            # Dense scenes: require clearing the cluster end with some slack.
            if int(slack) < 0:
                is_dangerous = True
                danger_reason = f"dense_cluster_not_cleared:{slack}px"
            # Require a small positive buffer; tuned so jl=6 becomes OK in the x≈1900~2000 window,
            # while jl=5 remains too tight.
            elif int(slack) < 3:
                is_dangerous = True
                danger_reason = f"dense_cluster_tight_clearance:{slack}px"
        # Platform height-diff: long jumps (>=5) are unstable when an enemy is below and a pipe is ahead.
        if (
            (not bool(is_dangerous))
            and pit_info is None
            and int(jl) >= 5
            and (mario_pos is not None)
            and bool(getattr(vlm_sentinel, "present", False))
            and bool(getattr(vlm_sentinel, "on_platform", False))
            and bool(getattr(vlm_sentinel, "enemy_below_ahead", False))
        ):
            has_pipe_ahead = any(t.name == "Pipe" and 0 < float(t.distance) <= 120.0 for t in threats)
            stompable = next(
                (t for t in threats if is_stompable_enemy_name(str(t.name)) and float(t.distance) > 0),
                None,
            )
            if has_pipe_ahead and stompable is not None and float(stompable.distance) <= 90.0:
                is_dangerous = True
                danger_reason = f"platform_long_jump_unstable:{stompable.name}@{int(stompable.distance)}px"

        panic_ahead = False
        panic_dist_after: int | None = None
        PANIC_ZONE_PX = 25
        for t in threats:
            if not is_enemy_threat_name(str(t.name)):
                continue
            if t.distance <= 0:
                continue
            dist_after = float(t.distance) - float(total_dx)
            adjusted_dist_after = dist_after - float(enemy_approach_margin)
            if 0 < adjusted_dist_after < PANIC_ZONE_PX:
                panic_ahead = True
                panic_dist_after = int(adjusted_dist_after)
                break
            if 0 < dist_after < PANIC_ZONE_PX:
                panic_ahead = True
                panic_dist_after = int(dist_after)
                break

        # Skill-first: walk(0) 是“慢速向右走”，不是原地等待。
        # 在“多敌人”的 HIGH/CRITICAL 窗口里，如果 walk(0) 仍显示为 OK，
        # VLM 很容易连续选择 0 进入贴脸必死态（即使不在低天花板段也会发生）。
        # 因此对 jump_level=0 增加短视野 lookahead，把这种“即将贴脸”标为 PANIC_AHEAD，
        # 让主 VLM/adapter 更倾向选择非 0 候选（例如 STOMP / DENSE_SAFE / SAFE）。
        if (
            (jl == 0)
            and (not panic_ahead)
            and (mario_pos is not None)
            and (int(mario_pos.y) <= 60)
            and (
                (urgency in {"HIGH", "CRITICAL"})
                or bool(enemy_cluster)
                or (
                    bool(getattr(vlm_sentinel, "present", False))
                    and (
                        bool(getattr(vlm_sentinel, "dense_enemies_ahead", False))
                        or (getattr(vlm_sentinel, "enemy_count_ahead", "unknown") == "two_plus")
                    )
                )
            )
        ):
            # If SAFE is intentionally set to jump_level=0 for a long dense cluster (approach phase),
            # do not mark walk(0) as PANIC_AHEAD. Otherwise the adapter cannot treat it as an OK
            # alternative and may allow risky jumps.
            prefer_dense_approach_walk = bool(
                int(candidates.safe_level) == 0
                and pit_info is None
                and (not has_low_ceiling)
                and (
                    bool(enemy_cluster)
                    or (
                        bool(getattr(vlm_sentinel, "present", False))
                        and (
                            bool(getattr(vlm_sentinel, "dense_enemies_ahead", False))
                            or (getattr(vlm_sentinel, "enemy_count_ahead", "unknown") == "two_plus")
                        )
                    )
                )
            )
            if not prefer_dense_approach_walk:
                ground_enemies = [
                    t for t in threats if is_enemy_threat_name(str(t.name)) and float(t.distance) > 0
                ]
                dense_signal = bool(
                    bool(getattr(vlm_sentinel, "present", False))
                    and (
                        bool(getattr(vlm_sentinel, "dense_enemies_ahead", False))
                        or (getattr(vlm_sentinel, "enemy_count_ahead", "unknown") == "two_plus")
                    )
                )
                min_count = 1 if dense_signal else 2
                if len(ground_enemies) >= int(min_count):
                    if enemy_cluster or dense_signal:
                        panic_trigger_px = float(PANIC_ZONE_PX)
                        if has_low_ceiling:
                            # 在 CRITICAL 窗口里，walk(0) 的“贴脸风险”上升更快：
                            # 多看一步，避免把 walk(0) 错误标为 OK 而稳定撞死。
                            WALK_PANIC_LOOKAHEAD_STEPS = 4 if urgency == "CRITICAL" else 3
                        else:
                            # 非低天花板段：允许短暂逼近以“展开视野”，避免把密集敌人误判为可盲跳的大位移宏动作。
                            # 只在“即将必撞”（<~8px）时才标记 PANIC_AHEAD 并触发 adapter 拒绝。
                            WALK_PANIC_LOOKAHEAD_STEPS = 4
                            panic_trigger_px = 8.0
                    else:
                        WALK_PANIC_LOOKAHEAD_STEPS = 5 if urgency == "HIGH" else 4
                        panic_trigger_px = float(PANIC_ZONE_PX)
                    close_speed = float(MARIO_SPEED_PER_STEP) + float(ENEMY_SPEED_PER_FRAME)
                    nearest = min(ground_enemies, key=lambda e: float(e.distance))
                    predicted_after = float(nearest.distance) - close_speed * float(WALK_PANIC_LOOKAHEAD_STEPS)
                    if float(predicted_after) < float(panic_trigger_px):
                        panic_ahead = True
                        panic_dist_after = max(0, int(predicted_after))

        return {
            "jump_level": int(jl),
            "dx_est": int(total_dx),
            "enemy_approach_margin": int(enemy_approach_margin),
            "would_hit_ceiling": bool(would_hit_ceiling),
            "ceiling_block_dist": ceiling_block_dist,
            "ceiling_block_y": ceiling_block_y,
            "takeoff_collision": bool(takeoff_collision),
            "takeoff_enemy_dist": takeoff_enemy_dist,
            "would_hit_solid": bool(would_hit_solid),
            "solid": {"name": solid_name, "dist_px": solid_dist} if would_hit_solid else None,
            "pit_landing": bool(pit_landing),
            "pit_too_close_to_jump": bool(pit_too_close_to_jump),
            "pit_edge_too_close_after": bool(pit_edge_too_close_after),
            "pit_edge_dist_after": pit_edge_dist_after,
            "pit_edge_near_after": bool(pit_edge_near_after),
            "pit_edge_near_dist_after": pit_edge_near_dist_after,
            "landing_danger": bool(is_dangerous),
            "landing_danger_reason": danger_reason,
            "min_enemy_dist_after": (int(min_dist_after) if isinstance(min_dist_after, (int, float)) else None),
            "panic_ahead_after_landing": bool(panic_ahead),
            "panic_dist_after": panic_dist_after,
        }

    def _risk_report_for_stomp(jl: int) -> dict[str, Any]:
        base = _risk_report_for_jump_level(jl)

        ground_enemies = [
            t
            for t in threats
            if is_stompable_enemy_name(str(t.name)) and 0 < float(t.distance) <= 200
        ]
        ground_enemies.sort(key=lambda t: float(t.distance))
        if len(ground_enemies) < 1:
            return base

        dense_cluster_gap_px: float | None = None
        if len(ground_enemies) >= 2:
            try:
                dense_cluster_gap_px = float(ground_enemies[1].distance) - float(ground_enemies[0].distance)
            except Exception:
                dense_cluster_gap_px = None
            if dense_cluster_gap_px is not None and float(dense_cluster_gap_px) > float(ENEMY_CLUSTER_GAP_PX):
                dense_cluster_gap_px = None

        other_enemies = ground_enemies[1:]
        STOMP_BOUNCE_BONUS_PX = 28
        STOMP_BOUNCE_EXTRA_ENEMY_APPROACH_PX = 12
        effective_total_dx = int(base.get("dx_est") or 0) + int(STOMP_BOUNCE_BONUS_PX)

        would_hit_solid = False
        solid_name = None
        solid_dist = None
        for t in threats:
            if t.name not in {"Pipe", "Stairs"}:
                continue
            if 0 < float(t.distance) <= float(effective_total_dx) + 10.0:
                need = _solid_clearance_level(t)
                if int(jl) < int(need):
                    would_hit_solid = True
                    solid_name = t.name
                    solid_dist = int(t.distance)
                    break

        pit_landing = False
        pit_too_close_to_jump = bool(base.get("pit_too_close_to_jump"))
        pit_edge_too_close_after = False
        pit_edge_dist_after: int | None = None
        if pit_info and mario_pos:
            p0, p1 = pit_info[0] - mario_pos.x, pit_info[1] - mario_pos.x
            if (p1 > 0) and (p0 < effective_total_dx) and (p1 > 0):
                if 0 < p0 < effective_total_dx or 0 < p1 < effective_total_dx:
                    pit_landing = True
            new_p0 = p0 - effective_total_dx
            if 0 < new_p0 < PIT_MIN_RUNWAY_PX:
                pit_edge_too_close_after = True
                pit_edge_dist_after = int(new_p0)

        enemy_approach_margin = int(base.get("enemy_approach_margin") or 0) + int(STOMP_BOUNCE_EXTRA_ENEMY_APPROACH_PX)
        landing_danger = False
        landing_danger_reason = "ok"
        min_enemy_dist_after: int | None = None
        for t in other_enemies:
            dist_after = float(t.distance) - float(effective_total_dx)
            adjusted_after = dist_after - float(enemy_approach_margin)
            if min_enemy_dist_after is None or abs(dist_after) < abs(float(min_enemy_dist_after)):
                min_enemy_dist_after = int(dist_after)
            if 0 < adjusted_after < float(LANDING_DANGER_MARGIN):
                landing_danger = True
                landing_danger_reason = f"landing_danger_with_approach:{adjusted_after:.0f}px"
                break

        panic_ahead = False
        panic_dist_after: int | None = None
        PANIC_ZONE_PX = 25
        if not landing_danger:
            for t in other_enemies:
                dist_after = float(t.distance) - float(effective_total_dx)
                adjusted_after = dist_after - float(enemy_approach_margin)
                if 0 < adjusted_after < PANIC_ZONE_PX:
                    panic_ahead = True
                    panic_dist_after = int(adjusted_after)
                    break

        base["dx_est"] = int(effective_total_dx)
        base["would_hit_solid"] = bool(would_hit_solid)
        base["solid"] = {"name": solid_name, "dist_px": solid_dist} if would_hit_solid else None
        base["pit_landing"] = bool(pit_landing)
        base["pit_edge_too_close_after"] = bool(pit_edge_too_close_after)
        base["pit_edge_dist_after"] = pit_edge_dist_after
        base["landing_danger"] = bool(landing_danger)
        base["landing_danger_reason"] = str(landing_danger_reason)
        base["min_enemy_dist_after"] = min_enemy_dist_after
        base["panic_ahead_after_landing"] = bool(panic_ahead)
        base["panic_dist_after"] = panic_dist_after
        # Dense cluster STOMP can be unstable; only allow it when the post-bounce plan
        # clears the next enemy with enough margin (otherwise it tends to die).
        if (not base.get("landing_danger")) and (dense_cluster_gap_px is not None) and other_enemies:
            # Late in the level (x≈2700+), there is a low-ceiling corridor with two Goombas where
            # the stable solution is often a *tight* chain-stomp (lvl1). Requiring "clear next enemy"
            # margin here is overly pessimistic and causes the policy to keep walking into a death.
            late_in_level = bool(state.env_x_pos is not None and int(state.env_x_pos) >= 2600)
            if late_in_level and has_low_ceiling:
                return base
            min_clearance_after_bounce_px: float | None = None
            for t in other_enemies:
                width = float(enemy_width_px(str(t.name)))
                # Enemies move during the macro; in low-ceiling corridors, allow a capped
                # approach credit so we don't reject otherwise-stable "double stomp" windows.
                approach_credit = float(min(int(enemy_approach_margin), 18)) if has_low_ceiling else 0.0
                clearance = float(effective_total_dx) - (float(t.distance) + width) + approach_credit
                if (min_clearance_after_bounce_px is None) or (clearance < float(min_clearance_after_bounce_px)):
                    min_clearance_after_bounce_px = float(clearance)
            # Tuned for the common x≈1900~2000 dense-goomba window.
            STOMP_CLUSTER_CLEARANCE_MIN_PX = 6.0
            if (min_clearance_after_bounce_px is None) or (float(min_clearance_after_bounce_px) < float(STOMP_CLUSTER_CLEARANCE_MIN_PX)):
                base["landing_danger"] = True
                base["landing_danger_reason"] = f"dense_cluster_gap:{dense_cluster_gap_px:.0f}px"
        return base

    risk_entries: list[dict[str, Any]] = []
    risk_entries.append({"name": "SAFE", **_risk_report_for_jump_level(int(candidates.safe_level))})
    if candidates.fast_level is not None:
        risk_entries.append({"name": "FAST", **_risk_report_for_jump_level(int(candidates.fast_level))})
    if candidates.stomp_level is not None:
        risk_entries.append({"name": "STOMP", **_risk_report_for_stomp(int(candidates.stomp_level))})
    if candidates.dense_safe_level is not None:
        risk_entries.append({"name": "DENSE_SAFE", **_risk_report_for_jump_level(int(candidates.dense_safe_level))})
    if candidates.stairs_jump_level is not None:
        risk_entries.append({"name": "STAIRS_JUMP", **_risk_report_for_jump_level(int(candidates.stairs_jump_level))})
    risk_entries.append({"name": "SLOW_WALK", **_risk_report_for_jump_level(0)})
    return risk_entries
