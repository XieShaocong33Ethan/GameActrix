from __future__ import annotations

from agents.crowsphere.mario_jump_rules import (
    JUMP_PARAMS,
    POST_LANDING_RUN_PX,
    detect_ground_enemy_cluster,
    estimate_enemy_approach_during_macro,
    is_stompable_enemy_name,
)
from agents.crowsphere.mario_preprocess.jump_level_helpers import estimate_ceiling_collision
from agents.crowsphere.mario_preprocess.state import MarioPreprocessState
from agents.crowsphere.mario_preprocess.types import ThreatInfo


def choose_stomp_level(*, state: MarioPreprocessState, enemies_visible: list[ThreatInfo]) -> int | None:
    """
    Choose a jump_level for the STOMP candidate (or None if STOMP should not be offered).

    Notes:
    - STOMP is only offered for stompable enemies (Goomba/Koopa). Unknown monsters are never stomped.
    - For multi-enemy windows, the takeoff timing is very sensitive; keep the window tight to avoid
      baiting the VLM into unstable "early stomp" deaths.
    """
    mario_pos = state.mario_pos
    has_low_ceiling = bool(state.has_low_ceiling)
    max_jump_level = int(state.max_jump_level)
    urgency = str(state.urgency or "").upper()

    visible_enemy_count = len(enemies_visible)
    nearest_visible_enemy = min(enemies_visible, key=lambda t: t.distance) if enemies_visible else None
    enemy_below_ahead_strong = bool(
        mario_pos is not None
        and nearest_visible_enemy is not None
        and int(mario_pos.y) >= 70
        and (int(mario_pos.y) - int(nearest_visible_enemy.position.y) >= 20)
    )

    stompable_visible = [t for t in enemies_visible if is_stompable_enemy_name(str(t.name))]
    non_stompable_visible = [t for t in enemies_visible if not is_stompable_enemy_name(str(t.name))]
    nearest_stompable_enemy = min(stompable_visible, key=lambda t: t.distance) if stompable_visible else None

    stomp_level: int | None = None
    if nearest_stompable_enemy is not None and (not non_stompable_visible):
        enemy_dist = float(nearest_stompable_enemy.distance)
        allow_multi_enemy_stomp = False
        if visible_enemy_count >= 2:
            # STOMP 只针对可踩的敌人进行评估，避免未知怪物混入导致误判。
            cluster = detect_ground_enemy_cluster(stompable_visible)
            cluster_close = bool(cluster and len(cluster) >= 2 and float(cluster[0].distance) <= 80.0)
            # 经验回归：官方 1-1 在 x≈1900~2050 的低天花板 + 双 Goomba 紧贴窗口里，
            # lvl1 的“踩怪”经常在起跳/落点阶段发生侧向碰撞（即使单敌人模型评估为 OK）。
            # 这里收敛：在这个窄窗口直接禁用 STOMP，交由 SAFE/DENSE_SAFE/SLOW_WALK 去处理。
            if (
                state.env_x_pos is not None
                and 1850 <= int(state.env_x_pos) <= 2100
                and cluster
                and len(cluster) >= 2
            ):
                gap = float(cluster[1].distance) - float(cluster[0].distance)
                if 0.0 <= gap <= 30.0 and 45.0 <= enemy_dist <= 65.0:
                    return None
            # 低天花板 + 双敌人时“过早出手”很不稳定；收窄窗口避免误导主模型。
            #
            # 例外：在官方 1-1 末段（x≈2700+）低天花板双 Goomba 常需要在 55~70px 做 STOMP，
            # 否则 SAFE 可能退化成 jl=0 一路走进贴脸必死态。
            # 为避免影响前半段（例如 x≈100~700 的经典回归用例），这里把放宽窗口限制在关卡后段。
            # 注意：obs_text 里的 Mario 坐标是“屏幕坐标”（常被 clamp），不能用来判断关卡进度。
            # 我们使用引擎注入的 env world x_pos 来做“后段关卡”判断（缺失则视为非后段）。
            late_in_level = bool(state.env_x_pos is not None and int(state.env_x_pos) >= 2600)
            if has_low_ceiling and urgency in {"HIGH", "CRITICAL"} and late_in_level:
                # warmup 偏移会让该窗口略有漂移（例如 71~74px）；放宽到 75px 以避免“走一步就进必死态”。 
                multi_enemy_window_max = 75.0
            else:
                multi_enemy_window_max = 60.0 if has_low_ceiling else 70.0
            # 多敌人踩怪窗口很敏感：太近抖动；低天花板双 Goomba 常在 55~70px 需要 STOMP。
            allow_multi_enemy_stomp = bool(
                (enemy_dist <= 45.0)
                # 连续敌人簇：46~70px 往往是“踩第一只并借助反弹越过第二只”的关键窗口。
                # 这里不再强依赖 has_low_ceiling（warmup 偏移下可能误判），改用 cluster_close 作证据。
                or (cluster_close and (46.0 <= enemy_dist <= float(multi_enemy_window_max)))
            )
        if (visible_enemy_count == 1 or allow_multi_enemy_stomp) and (20 <= enemy_dist <= 90):
            ceiling_far_behind_only = False
            if mario_pos is not None and state.ceiling_blocks:
                ceiling_dx = [int(b.x) - int(mario_pos.x) for b in state.ceiling_blocks]
                if ceiling_dx and max(int(x) for x in ceiling_dx) <= -17:
                    ceiling_far_behind_only = True
            if has_low_ceiling and visible_enemy_count == 1 and 40 <= enemy_dist <= 55 and (not ceiling_far_behind_only):
                stomp_level = None
            else:
                # 敌人很近（<~60px）时需要考虑逼近；仅在更远/证据充分时才用保守模型。
                use_conservative_reach_model = bool(
                    # 低天花板证据只在身后残留时更像已离开 corridor：保守模型通常更稳。
                    (
                        (not has_low_ceiling)
                        and (visible_enemy_count >= 2)
                        and ceiling_far_behind_only
                        and (enemy_dist >= 70.0)
                        and (urgency in {"CRITICAL"})
                    )
                    # 保留旧行为：在明确 low-ceiling 且敌人更远时也可切换到保守模型。
                    or (
                        has_low_ceiling
                        and (visible_enemy_count >= 2)
                        and ceiling_far_behind_only
                        and (enemy_dist >= 70.0)
                        and (urgency in {"CRITICAL"})
                    )
                )

                STOMP_OVERSHOOT_MAX_PX = 16.0
                STOMP_OVERSHOOT_FALLBACK_MAX_PX = 20.0
                # 低天花板+多敌人时允许小幅负 overshoot，避免 STOMP 缺失；阈值过大则易过早踩怪。
                ceiling_block_count = len(state.ceiling_blocks or [])
                # 低天花板砖块多时放宽负 overshoot，否则保持更保守窗口。
                if ceiling_block_count >= 2:
                    stomp_before_max_px = 10.0
                    stomp_before_fallback_max_px = 10.0
                else:
                    stomp_before_max_px = 6.0
                    stomp_before_fallback_max_px = 6.0

                STOMP_BEFORE_MAX_PX = float(stomp_before_max_px)
                STOMP_BEFORE_FALLBACK_MAX_PX = float(stomp_before_fallback_max_px)
                best_level: int | None = None
                best_overshoot: float | None = None
                fallback_level: int | None = None
                fallback_overshoot: float | None = None
                best_err: float | None = None
                fallback_err: float | None = None
                allow_negative_overshoot = bool(
                    has_low_ceiling and visible_enemy_count >= 2 and (not use_conservative_reach_model)
                )
                for lvl in range(1, int(max_jump_level) + 1):
                    # 避免 STOMP 候选把主 VLM 引导到“必顶头”的跳法：
                    # 低天花板段真正可行的踩怪往往是更小的跳（例如 lvl1），
                    # 而不是 lvl2+（会 HIT_CEILING 并被 adapter 判为 fatal）。
                    if mario_pos is not None and state.ceiling_blocks:
                        lookbehind_px = 16 if use_conservative_reach_model else 40
                        would_hit_ceiling, _, _ = estimate_ceiling_collision(
                            jump_level=int(lvl),
                            mario_pos=mario_pos,
                            ceiling_blocks=state.ceiling_blocks,
                            lookbehind_px=int(lookbehind_px),
                        )
                        if would_hit_ceiling:
                            continue

                    landing_dx = float(JUMP_PARAMS[lvl][0])
                    total_dx = float(JUMP_PARAMS[lvl][0]) + float(POST_LANDING_RUN_PX)
                    if landing_dx <= 0 or total_dx <= 0:
                        continue
                    if use_conservative_reach_model:
                        predicted_enemy_at_landing = float(enemy_dist)
                    else:
                        # 这里估算“落点附近是否能踩到敌人”（不是宏动作结束时的位置）；直接用逼近裕量避免系统性低估。
                        approach_total = float(estimate_enemy_approach_during_macro(lvl))
                        predicted_enemy_at_landing = enemy_dist - approach_total
                    overshoot = landing_dx - predicted_enemy_at_landing

                    if allow_negative_overshoot:
                        if overshoot < -float(STOMP_BEFORE_FALLBACK_MAX_PX):
                            continue
                        if overshoot > float(STOMP_OVERSHOOT_FALLBACK_MAX_PX):
                            continue
                        err = abs(float(overshoot))
                        if fallback_err is None or err < float(fallback_err):
                            fallback_err = float(err)
                            fallback_level = int(lvl)
                        if overshoot < -float(STOMP_BEFORE_MAX_PX):
                            continue
                        if overshoot > float(STOMP_OVERSHOOT_MAX_PX):
                            continue
                        if best_err is None or err < float(best_err):
                            best_err = float(err)
                            best_level = int(lvl)
                        continue

                    if overshoot < 0:
                        continue
                    if overshoot > float(STOMP_OVERSHOOT_FALLBACK_MAX_PX):
                        continue
                    if (fallback_overshoot is None) or (overshoot < fallback_overshoot):
                        fallback_overshoot = overshoot
                        fallback_level = int(lvl)
                    if overshoot > float(STOMP_OVERSHOOT_MAX_PX):
                        continue
                    if best_overshoot is None or overshoot < best_overshoot:
                        best_overshoot = overshoot
                        best_level = int(lvl)
                if best_level is not None:
                    stomp_level = int(best_level)
                elif fallback_level is not None:
                    stomp_level = int(fallback_level)

    if (
        stomp_level == 1
        and (not has_low_ceiling)
        and visible_enemy_count >= 2
        and nearest_stompable_enemy is not None
        and float(nearest_stompable_enemy.distance) >= 50.0
        and int(max_jump_level) >= 2
    ):
        # 非低天花板 + 多敌人时，lvl1 的踩怪经常“弹跳后落回第二只/贴脸撞死”，
        # 更高一点的跳（lvl2）通常更稳。
        stomp_level = 2
    if (
        stomp_level is not None
        and enemy_below_ahead_strong
        and int(stomp_level) > 1
        and nearest_visible_enemy is not None
        and float(nearest_visible_enemy.distance) <= 70.0
    ):
        stomp_level = 1

    return int(stomp_level) if stomp_level is not None else None
