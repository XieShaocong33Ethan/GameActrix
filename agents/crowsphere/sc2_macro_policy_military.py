from __future__ import annotations

from typing import Any

from agents.crowsphere.sc2_macro_policy_utils import (
    compute_enemy_info_fallback_attack_policy as _compute_enemy_info_fallback_attack_policy,
    ensure_action as _ensure_action,
    remove_actions as _remove_actions,
)


def apply_military_actions(
    *,
    state: dict[str, Any],
    result: list[str],
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any],
    # Map / time
    rush_map: bool,
    game_time_seconds: int,
    # Our army & production
    army_supply: int,
    stalker_producing: int,
    # Enemy
    enemy_total_visible: int,
    enemy_ground_threat: int,
    enemy_ranged_memory: int,
    enemy_air_memory: int,
    early_ranged_rush: bool,
    # Strategy choices
    tech_plan: str,
    military: str,
    enable_retreat: bool,
    # Context (cross-step memory)
    peak_army: int,
    last_attack_step: int,
    last_retreat_step: int,
    step_index: int,
    last_scout_step: int,
    scout_count: int,
    enemy_info_supported: bool,
) -> None:
    """Apply attack/retreat actions (MULTI-ATTACK / MULTI-RETREAT) to `result` in-place.

    This function intentionally only decides high-level "go out" / "pull back".
    Micro is handled by the environment-side bot (python-sc2).
    """
    attack_min_time_seconds = int(policy_config.get("attack_min_time_seconds", 720))
    attack_army_threshold = int(policy_config.get("attack_army_threshold", 32))
    attack_cooldown_steps = int(policy_config.get("attack_cooldown_steps", 6))
    # Cleanup mode: when only a tiny amount of enemy presence remains, send MULTI-ATTACK
    # more frequently so the bot keeps sweeping instead of idling until the time limit.
    cleanup_attack_min_time_seconds = int(policy_config.get("cleanup_attack_min_time_seconds", 720))
    cleanup_attack_enemy_visible_max = int(policy_config.get("cleanup_attack_enemy_visible_max", 2))
    cleanup_attack_cooldown_steps = int(policy_config.get("cleanup_attack_cooldown_steps", 2))
    if (game_time_seconds >= max(0, cleanup_attack_min_time_seconds)) and (
        enemy_total_visible <= max(0, cleanup_attack_enemy_visible_max)
    ):
        attack_cooldown_steps = max(1, min(attack_cooldown_steps, cleanup_attack_cooldown_steps))

    effective_attack_min_time = attack_min_time_seconds
    effective_attack_threshold = attack_army_threshold
    if rush_map:
        effective_attack_min_time = min(
            effective_attack_min_time,
            int(policy_config.get("rush_map_attack_min_time_seconds") or 240),
        )
        effective_attack_threshold = int(
            policy_config.get("rush_map_attack_army_threshold") or min(effective_attack_threshold, 8)
        )

    stalker_count = int(state.get("stalker_count", 0))
    stalker_total = stalker_count + int(stalker_producing or 0)
    immortal_count = int(state.get("immortal_count", 0))

    extra_attack_requirements_met = True
    # Map-specific tuning hook: some maps/seed distributions punish small Immortal counts.
    # Default keeps behavior close to the original policy.
    min_immortals_before_attack = int(policy_config.get("min_immortals_before_attack", 1))
    min_immortals_before_ranged_attack = int(
        policy_config.get("min_immortals_before_ranged_attack", 2)
    )
    if (not rush_map) and tech_plan == "immortal" and game_time_seconds < 720:
        extra_attack_requirements_met = extra_attack_requirements_met and (
            immortal_count >= max(0, min_immortals_before_attack)
        )

    # Standard maps: early ranged ground pressure (e.g. Roach/Hydra timing) punishes
    # small attacks, so delay our first move-out.
    #
    # Rush maps (Flat64): if we stall too long, we get contained and die. Do not apply
    # the "big army only" gate there; rush-map thresholds are handled separately below.
    if early_ranged_rush and (not rush_map):
        effective_attack_min_time = max(effective_attack_min_time, 420)
        effective_attack_threshold = max(effective_attack_threshold, 44)

    if tech_plan == "immortal" and enemy_ranged_memory > 0:
        # Standard maps: ranged ground armies (Roach/Hydra) punish small attacks.
        # Rush maps (Flat64): if we stall too long, we get contained and die, so keep the
        # threshold lower to ensure we still move out.
        if rush_map:
            effective_attack_min_time = max(effective_attack_min_time, 240)
            effective_attack_threshold = max(
                effective_attack_threshold,
                int(policy_config.get("rush_map_ranged_attack_army_threshold", 30)),
            )
            # Rush maps: do not add strict unit-composition gates, otherwise we can stall forever
            # and get contained to death. Composition is handled by macro rules separately.
        else:
            effective_attack_min_time = max(effective_attack_min_time, 420)
            effective_attack_threshold = max(effective_attack_threshold, 40)
            extra_attack_requirements_met = extra_attack_requirements_met and (
                immortal_count >= max(0, min_immortals_before_ranged_attack)
            )

    if enemy_air_memory > 0:
        effective_attack_min_time = max(effective_attack_min_time, 420)
        if rush_map:
            # Rush maps: don't let a small air force (e.g. early Mutalisk) stall attacks forever.
            effective_attack_threshold = max(
                effective_attack_threshold,
                int(policy_config.get("rush_map_enemy_air_attack_army_threshold", 30)),
            )
            # Same as above: avoid hard gates on rush maps.
        else:
            effective_attack_threshold = max(effective_attack_threshold, 40)
            extra_attack_requirements_met = extra_attack_requirements_met and (
                stalker_total >= max(6, enemy_air_memory * 2)
            )

    (
        effective_attack_min_time,
        effective_attack_threshold,
        enemy_info_uncertain,
        scouted_recently,
        force_attack_by_time,
    ) = _compute_enemy_info_fallback_attack_policy(
        policy_config=policy_config,
        game_time_seconds=game_time_seconds,
        step_index=step_index,
        last_scout_step=last_scout_step,
        scout_count=scout_count,
        enemy_info_supported=enemy_info_supported,
        effective_attack_min_time=effective_attack_min_time,
        effective_attack_threshold=effective_attack_threshold,
    )

    # 当我方军队已经成型时，不应因为“当前视野里有少量敌军”而迟迟不出兵。
    force_attack_army_supply = int(policy_config.get("force_attack_army_supply", 120))
    force_attack_big_army = army_supply >= max(0, force_attack_army_supply)
    # 当军队规模已经很大时，不要再被“阵容门槛”卡住（例如 Immortal 数量不足）。
    # 否则会出现“我方 100+ army_supply 仍不出门 -> 被对手拖到后期翻盘/超时”的情况。
    if force_attack_big_army:
        extra_attack_requirements_met = True

    if rush_map:
        # Flat64 上“视野里还有敌军”往往意味着对手就在门口；此时贸然出门容易被一波打穿。
        # 允许两种情况出门：
        # - 视野里没有敌方战斗单位（enemy_total_visible==0）
        # - 或者我方军队已经足够大（例如 40+ army_supply）能正面硬推
        rush_map_enemy_visible_min_army = int(
            policy_config.get("rush_map_attack_enemy_visible_min_army_supply", 40)
        )
        enemy_safe_to_attack = (
            force_attack_big_army
            or (enemy_total_visible == 0)
            or (army_supply >= max(0, rush_map_enemy_visible_min_army))
        )
    else:
        # 标准图：允许“少量敌军可见”也能出门（例如 1-2 个侦察/骚扰单位），否则会出现
        # “敌军一直有小单位在视野里 -> 我方永远等不到 enemy_total_visible==0 -> 攻击节奏被拖慢”。
        max_visible = int(policy_config.get("attack_enemy_visible_max", 2))
        # 证据：并行评测中出现过“全程不出兵（attack_count=0）”的失败局。
        # 原因往往是：对手持续用一堆小单位（例如 Zergling）挂在视野里，导致 enemy_total_visible
        # 长期 > 2，从而永远无法触发出门条件。此时应该允许在“我方军队已成型”时强制出门。
        min_army_for_visible_enemy = int(policy_config.get("attack_enemy_visible_min_army_supply", 70))
        max_visible_for_min_army = int(policy_config.get("attack_enemy_visible_max_for_min_army", 20))
        enemy_safe_to_attack = (
            force_attack_big_army
            or (enemy_total_visible <= max(0, max_visible))
            or (
                (army_supply >= max(0, min_army_for_visible_enemy))
                and (enemy_total_visible <= max(0, max_visible_for_min_army))
            )
        )

    attack_ready = (
        (game_time_seconds >= effective_attack_min_time)
        and (army_supply >= effective_attack_threshold)
        and enemy_safe_to_attack
        and extra_attack_requirements_met
        and ((not enemy_info_uncertain) or scouted_recently or force_attack_by_time)
    )

    manual_attack = military == "attack"
    if enemy_info_uncertain:
        manual_attack = manual_attack and (enemy_total_visible == 0) and (
            scouted_recently or force_attack_by_time
        )
        manual_attack = manual_attack and (game_time_seconds >= effective_attack_min_time) and (
            army_supply >= effective_attack_threshold
        )

    should_attack = manual_attack or (military != "defend" and attack_ready)

    # 注意：military=defend 表示“不要主动出兵”，不应等价为“全军持续撤退”。
    # 环境侧 bot 已有 defend() 微操；频繁 MULTI-RETREAT 会覆盖战斗指令，导致基地被白打。
    #
    # 经验：若不加约束，某些 seed 下会出现“整局几百次撤退指令”，最终把自己拖死（尤其是标准图后期）。
    # 因此：
    # - 只在“近期确实发起过进攻”时允许撤退（用于拉回出门部队）
    # - 加一个撤退冷却，避免每个 step 都撤退
    retreat_recent_attack_window = int(policy_config.get("retreat_recent_attack_window_steps", 60))
    retreat_cooldown_steps = int(policy_config.get("retreat_cooldown_steps", 30))
    recently_attacked = (step_index - last_attack_step) <= retreat_recent_attack_window
    retreat_off_cooldown = (step_index - last_retreat_step) >= retreat_cooldown_steps

    should_retreat = (
        enable_retreat
        and recently_attacked
        and retreat_off_cooldown
        and peak_army >= 12
        and army_supply < int(peak_army * 0.6)
        and enemy_ground_threat > 0
    )

    if should_retreat:
        _remove_actions(result, action_names={"MULTI-ATTACK"})
        _ensure_action(result, "MULTI-RETREAT", allowed_actions=allowed_actions, max_count=1)
        return

    if should_attack and (step_index - last_attack_step) >= attack_cooldown_steps:
        _remove_actions(result, action_names={"MULTI-RETREAT"})
        _ensure_action(result, "MULTI-ATTACK", allowed_actions=allowed_actions, max_count=1)
        return

    _remove_actions(result, action_names={"MULTI-ATTACK", "MULTI-RETREAT"})
