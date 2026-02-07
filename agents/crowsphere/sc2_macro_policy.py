from __future__ import annotations
from typing import Any
from agents.crowsphere.sc2_macro_policy_utils import (
    apply_air_cleanup_actions as _apply_air_cleanup_actions,
    can_afford as _can_afford,
    clamp_action_counts as _clamp_action_counts,
    ensure_action as _ensure_action,
    effective_robotics_facility_count as _effective_robotics_facility_count,
    remove_actions as _remove_actions,
    recently_issued as _recently_issued,
    supply_cost as _supply_cost,
)
from agents.crowsphere.sc2_macro_policy_military import apply_military_actions
from agents.crowsphere.sc2_macro_policy_expansion import apply_expansion_actions
from agents.crowsphere.sc2_macro_policy_scouting import apply_scouting_actions
from agents.crowsphere.sc2_enemy_classifier import aggregate_enemy_units, earliest_seen_time_seconds
def postprocess_actions(
    *,
    state: dict[str, Any],
    actions: list[str],
    train: bool,
    build_defense: bool,
    military: str,
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any] | None,
    context: dict[str, Any] | None,
) -> list[str]:
    policy_config = policy_config or {}
    context = context or {}
    result = list(actions)
    map_name = str(context.get("_map_name") or "")
    rush_map = map_name.strip().lower() == "flat64"
    disable_static_defense = bool(policy_config.get("disable_static_defense", True))
    if rush_map:
        # Flat64 距离极近，早期压力大；允许造电池做兜底（环境侧已避免 near=None 崩溃）。
        disable_static_defense = False
    if allowed_actions is not None:
        result = [a if a in allowed_actions else "EMPTY ACTION" for a in result]
    if disable_static_defense:
        # Shield Battery/Photon Cannon 曾触发官方 bot 建造崩溃（near=None）；为保证评测稳定性，默认禁用。
        _remove_actions(result, action_names={"BUILD PHOTONCANNON", "BUILD SHIELDBATTERY"})
    game_time_seconds = int(state.get("game_time_seconds", 0))
    supply_left = int(state.get("supply_left", 0))
    supply_cap = int(state.get("supply_cap", 0))
    max_supply_cap = int(policy_config.get("max_supply_cap", 200))
    supply_blocked = bool(state.get("supply_blocked", False))
    nexus_count = int(state.get("nexus_count", 0))
    probe_count = int(state.get("probe_count", 0)) or int(state.get("worker_supply", 0))
    probe_producing = int(state.get("probe_producing", 0))
    gateway_count = int(state.get("gateway_count", 0)) + int(state.get("warp_gate_count", 0))
    gateway_constructing = int(state.get("gateway_constructing", 0))
    gateway_total = gateway_count + gateway_constructing
    pylon_count = int(state.get("pylon_count", 0))
    pylon_constructing = int(state.get("pylon_constructing", 0))
    zealot_producing = int(state.get("zealot_producing", 0))
    stalker_producing = int(state.get("stalker_producing", 0))
    army_supply = int(state.get("army_supply", 0))
    # Enemy threat aggregation is generalized so bot_race != Zerg still works.
    enemy_ranged_threat = int(state.get("enemy_ground_ranged_threat", state.get("enemy_ranged_threat", 0)))
    enemy_melee_threat = int(state.get("enemy_ground_melee_threat", state.get("enemy_zergling", 0)))
    enemy_air_threat = int(state.get("enemy_air_threat", 0))
    enemy_total_visible = int(
        state.get("enemy_total_visible", enemy_ranged_threat + enemy_melee_threat + enemy_air_threat)
    )
    enemy_ground_threat = int(state.get("enemy_ground_threat", enemy_ranged_threat + enemy_melee_threat))
    peak_army = int(context.get("peak_army", 0))
    last_attack_step = int(context.get("last_attack_step", -10))
    last_retreat_step = int(context.get("last_retreat_step", -9999))
    last_scout_step = int(context.get("last_scout_step", -9999))
    step_index = int(context.get("step", 0))
    scout_count = int(context.get("scout_count", 0) or 0)
    # `enemy_info_supported` 用于判断环境是否提供 enemy unit lines；一旦证实就应持久为 True。
    # 否则会触发 fallback 逻辑，导致标准图迟迟不出兵，给对手足够时间上空军/高科技反打。
    enemy_info_supported = bool(context.get("enemy_info_supported", False))
    last_enemy_combat_seen_step = int(context.get("last_enemy_combat_seen_step", -9999) or -9999)
    enemy_combat_recent_steps = int(policy_config.get("enemy_info_recent_enemy_seen_steps", 120))
    enemy_combat_seen_recently = (step_index - last_enemy_combat_seen_step) <= enemy_combat_recent_steps
    enemy_seen = context.get("enemy_seen") if isinstance(context.get("enemy_seen"), dict) else {}
    enemy_first_seen_time = (
        context.get("enemy_first_seen_time_seconds")
        if isinstance(context.get("enemy_first_seen_time_seconds"), dict)
        else {}
    )
    enable_enemy_memory = bool(policy_config.get("enable_enemy_memory", True))
    if not enable_enemy_memory:
        enemy_seen = {}
        enemy_first_seen_time = {}
    seen_agg = aggregate_enemy_units(enemy_seen)
    enemy_ranged_seen = int(seen_agg.get("ground_ranged", 0))
    enemy_air_seen = int(seen_agg.get("air", 0))
    enemy_ranged_memory = max(enemy_ranged_threat, enemy_ranged_seen)
    enemy_air_memory = max(enemy_air_threat, enemy_air_seen)
    # Proactive air-tech hinting: seeing a Spire means Mutalisk/Corruptor can appear soon.
    # Treat that as a small "air memory" so we start anti-air production before flying combat appears.
    try:
        spire_seen = int(enemy_seen.get("spire", 0) or 0)
    except Exception:
        spire_seen = 0
    try:
        greaterspire_seen = int(enemy_seen.get("greaterspire", 0) or 0)
    except Exception:
        greaterspire_seen = 0
    if (spire_seen > 0) and enemy_air_memory <= 0:
        enemy_air_memory = 2
    if greaterspire_seen > 0:
        enemy_air_memory = max(enemy_air_memory, 6)
    ranged_first_seen_time = earliest_seen_time_seconds(enemy_first_seen_time, category="ground_ranged")
    early_ranged_rush = (ranged_first_seen_time is not None) and (ranged_first_seen_time <= 240)
    tech_plan = str(policy_config.get("tech_plan") or "immortal").strip().lower()
    if tech_plan not in {"zealot_only", "immortal"}:
        tech_plan = "zealot_only"
    # Flat64（rush_map）上若对手很早就上 Roach/Hydra，纯 zealot 往往打不穿且容易被滚雪球。
    # 证据：并行评测中出现“Flat64 全局不出兵（army_peak≈30 < 阈值）-> 被 Roach/Infestor 拖死”的局。
    # 因此：在 rush_map 上一旦确认存在远程地面威胁，转入 Immortal 科技线提高稳定性。
    if rush_map and tech_plan == "zealot_only" and enemy_ranged_memory > 0:
        tech_plan = "immortal"
    # 默认使用 two_base 策略（基于消融实验证据：胜率从 0% 提升到 100%）
    # 详见 docs/SC2_AGENT_MAINTENANCE.md 的消融实验结果
    macro_policy = str(policy_config.get("macro_policy") or "two_base").strip().lower()
    if macro_policy not in {"legacy", "two_base"}:
        macro_policy = "two_base"
    enable_warpgate_research = bool(policy_config.get("enable_warpgate_research", False))
    enable_retreat = bool(policy_config.get("enable_retreat", False))
    _clamp_action_counts(result, action_name="BUILD PYLON", max_count=1)
    _clamp_action_counts(result, action_name="BUILD GATEWAY", max_count=1)
    _clamp_action_counts(result, action_name="BUILD NEXUS", max_count=1)
    _clamp_action_counts(result, action_name="BUILD FORGE", max_count=1)
    _clamp_action_counts(result, action_name="BUILD PHOTONCANNON", max_count=0 if disable_static_defense else 1)
    _clamp_action_counts(result, action_name="BUILD SHIELDBATTERY", max_count=0 if disable_static_defense else 1)
    _clamp_action_counts(result, action_name="TRAIN PROBE", max_count=1)
    _clamp_action_counts(result, action_name="TRAIN STALKER", max_count=1)
    _clamp_action_counts(result, action_name="RESEARCH WARPGATERESEARCH", max_count=1)
    _clamp_action_counts(result, action_name="MULTI-ATTACK", max_count=1)
    _clamp_action_counts(result, action_name="MULTI-RETREAT", max_count=1)
    if game_time_seconds >= 480:
        pylon_supply_threshold = 7
        pylon_pending_limit = 4
    elif game_time_seconds >= 360:
        pylon_supply_threshold = 5
        pylon_pending_limit = 3
    else:
        pylon_supply_threshold = 5 if rush_map else 3
        pylon_pending_limit = 4 if rush_map else 2
    pylon_cooldown_steps = 1 if (rush_map and supply_blocked) else 3
    should_build_pylon = (
        supply_left <= pylon_supply_threshold
        and supply_cap < max(0, max_supply_cap)
        and pylon_constructing <= pylon_pending_limit
        and not _recently_issued("BUILD PYLON", context=context, step=step_index, cooldown_steps=pylon_cooldown_steps)
        and _can_afford("BUILD PYLON", state)
    )
    if should_build_pylon:
        _ensure_action(result, "BUILD PYLON", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"BUILD PYLON"})
    if game_time_seconds < 180:
        probe_target = 20
    elif game_time_seconds < 360:
        probe_target = 30
    else:
        probe_target = 44
    nexus_constructing = int(state.get("nexus_constructing", 0))
    probe_target = min(80, probe_target + max(0, nexus_count - 1) * 16)
    # 两矿过度补 Probe 会显著拖慢军队成型时间，容易吃 Timing Attack。
    try:
        probe_target_cap = int(policy_config.get("probe_target_cap", 48))
    except Exception:
        probe_target_cap = 48
    if rush_map and (nexus_count + nexus_constructing) < 2:
        # Flat64 单矿阶段 Probe 过多会拖慢军队成型；未出二矿前将 Probe cap 压到 30 左右更稳。
        probe_target_cap = min(probe_target_cap, 30 if game_time_seconds >= 360 else 32)
    if probe_target_cap > 0:
        probe_target = min(probe_target, probe_target_cap)
    if (
        nexus_count > 0
        and train
        and probe_count < probe_target
        and probe_producing < max(1, nexus_count)
        and supply_left >= _supply_cost("TRAIN PROBE")
        and _can_afford("TRAIN PROBE", state)
    ):
        _ensure_action(result, "TRAIN PROBE", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"TRAIN PROBE"})
    if game_time_seconds < 180:
        gateway_target = 1
    elif game_time_seconds < 360:
        gateway_target = 3
    else:
        gateway_target = 5
    if gateway_count <= 0:
        gateway_target = 1
    if enemy_ranged_memory > 0 and game_time_seconds < 360 and tech_plan != "immortal":
        gateway_target = max(gateway_target, 4)
    if (
        pylon_count > 0
        and gateway_total < gateway_target
        and gateway_constructing <= 2
        and not _recently_issued("BUILD GATEWAY", context=context, step=step_index, cooldown_steps=4)
        and _can_afford("BUILD GATEWAY", state)
    ):
        _ensure_action(result, "BUILD GATEWAY", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"BUILD GATEWAY"})
    apply_expansion_actions(
        state=state,
        result=result,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
        context=context,
        rush_map=rush_map,
        macro_policy=macro_policy,
        game_time_seconds=game_time_seconds,
        step_index=step_index,
        army_supply=army_supply,
        probe_count=probe_count,
        gateway_total=gateway_total,
        supply_blocked=(supply_blocked if (not rush_map or supply_left <= 0) else False),
        enemy_total_visible=enemy_total_visible,
    )
    if "BUILD NEXUS" in result:
        return result
    cybernetics_core_total = int(state.get("cybernetics_core_count", 0)) + int(
        state.get("cybernetics_core_constructing", 0)
    )
    robotics_facility_count = _effective_robotics_facility_count(state)
    robotics_facility_total = robotics_facility_count + int(state.get("robotics_facility_constructing", 0))
    gas_buildings_total = int(state.get("gas_buildings_count", 0)) + int(
        state.get("gas_buildings_constructing", 0)
    )
    forge_total = int(state.get("forge_count", 0)) + int(state.get("forge_constructing", 0))
    photon_cannon_total = int(state.get("photon_cannon_count", 0)) + int(
        state.get("photon_cannon_constructing", 0)
    )
    shield_battery_total = int(state.get("shield_battery_count", 0)) + int(
        state.get("shield_battery_constructing", 0)
    )
    defense_tech_triggered = (
        build_defense
        or game_time_seconds >= 120
        or enemy_ranged_memory > 0
        or enemy_air_memory > 0
    )
    if defense_tech_triggered:
        # Gas: prefer 2 Assimilators when we need to respond to air tech or sustain Immortal tech.
        desired_gas_buildings = (
            2 if (not rush_map) and (enemy_air_memory > 0 or tech_plan == "immortal" or game_time_seconds >= 420) else 1
        )
        # Flat64：如果已转入 Immortal 科技线，单气往往不够支撑持续产 Immortal/Stalker。
        # 仅在确认早期存在远程地面威胁（Roach/Hydra）时才提前到 04:00 补第二气；
        # 否则仍维持更晚的节奏，避免早期矿不足导致崩盘。
        if rush_map and tech_plan == "immortal":
            if (enemy_ranged_memory > 0 and game_time_seconds >= 240) or game_time_seconds >= 360:
                desired_gas_buildings = max(desired_gas_buildings, 2)
        if (
            gas_buildings_total < desired_gas_buildings
            and pylon_count > 0
            and not _recently_issued("BUILD ASSIMILATOR", context=context, step=step_index, cooldown_steps=6)
            and _can_afford("BUILD ASSIMILATOR", state)
        ):
            _ensure_action(result, "BUILD ASSIMILATOR", allowed_actions=allowed_actions, max_count=1)
        if (
            cybernetics_core_total < 1
            and gateway_count > 0
            and not _recently_issued(
                "BUILD CYBERNETICSCORE",
                context=context,
                step=step_index,
                cooldown_steps=8,
            )
            and _can_afford("BUILD CYBERNETICSCORE", state)
        ):
            _ensure_action(result, "BUILD CYBERNETICSCORE", allowed_actions=allowed_actions, max_count=1)
        # Flat64（rush map）避免过早补 Forge（150 矿）；仅在敌军压到视野或进入中期后再补（用于炮台/升级）。
        want_forge = ((not disable_static_defense) and (enemy_air_memory > 0 or game_time_seconds >= 420)) or (rush_map and (enemy_total_visible > 0 or game_time_seconds >= 360))
        if (
            want_forge
            and forge_total < 1
            and pylon_count > 0
            and not _recently_issued("BUILD FORGE", context=context, step=step_index, cooldown_steps=10)
            and _can_afford("BUILD FORGE", state)
        ):
            _ensure_action(result, "BUILD FORGE", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"BUILD ASSIMILATOR", "BUILD CYBERNETICSCORE", "BUILD FORGE"})
    cybernetics_core_count = int(state.get("cybernetics_core_count", 0))
    forge_count = int(state.get("forge_count", 0))
    defense_buildings_triggered = (
        build_defense
        or game_time_seconds >= 360
        or enemy_ranged_memory > 0
        or enemy_air_memory > 0
        or (rush_map and (enemy_total_visible > 0 or game_time_seconds >= 240))
    )
    if defense_buildings_triggered and (not disable_static_defense):
        # 防止出门进攻被偷家/空军翻盘：随时间逐步补炮台/电池（Flat64 更偏防守兜底）。
        desired_cannons = (0 if game_time_seconds < 360 else (1 if game_time_seconds < 600 else 2)) if rush_map else (4 if game_time_seconds >= 900 else (3 if game_time_seconds >= 600 else (2 if game_time_seconds >= 480 else 1)))
        desired_batteries = (1 if game_time_seconds < 360 else (2 if game_time_seconds < 600 else 3)) if rush_map else (4 if game_time_seconds >= 900 else (3 if game_time_seconds >= 600 else (2 if game_time_seconds >= 480 else 1)))
        if enemy_air_memory > 0:
            desired_cannons = max(desired_cannons, 2 if rush_map else 3)
            desired_batteries = max(desired_batteries, 2 if rush_map else 3)
        if (
            forge_count > 0
            and photon_cannon_total < desired_cannons
            and not _recently_issued(
                "BUILD PHOTONCANNON",
                context=context,
                step=step_index,
                cooldown_steps=6,
            )
            and _can_afford("BUILD PHOTONCANNON", state)
        ):
            _ensure_action(result, "BUILD PHOTONCANNON", allowed_actions=allowed_actions, max_count=1)
        if (
            cybernetics_core_count > 0
            and shield_battery_total < desired_batteries
            and not _recently_issued(
                "BUILD SHIELDBATTERY",
                context=context,
                step=step_index,
                cooldown_steps=6,
            )
            and _can_afford("BUILD SHIELDBATTERY", state)
        ):
            _ensure_action(result, "BUILD SHIELDBATTERY", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"BUILD PHOTONCANNON", "BUILD SHIELDBATTERY"})

    robo_min_time_seconds = 180 if rush_map else 240
    immortal_triggered = (tech_plan == "immortal") and (
        game_time_seconds >= 300 or (enemy_ranged_memory > 0 and game_time_seconds >= robo_min_time_seconds)
    )
    if not immortal_triggered:
        _remove_actions(result, action_names={"BUILD ROBOTICSFACILITY", "TRAIN IMMORTAL"})
    else:
        if (
            robotics_facility_total < 1
            and cybernetics_core_count > 0
            and not _recently_issued(
                "BUILD ROBOTICSFACILITY",
                context=context,
                step=step_index,
                cooldown_steps=8,
            )
            and _can_afford("BUILD ROBOTICSFACILITY", state)
        ):
            _ensure_action(result, "BUILD ROBOTICSFACILITY", allowed_actions=allowed_actions, max_count=1)

        if (
            robotics_facility_count > 0
            and int(state.get("immortal_producing", 0)) < robotics_facility_count
            and supply_left >= _supply_cost("TRAIN IMMORTAL")
            and _can_afford("TRAIN IMMORTAL", state)
        ):
            _ensure_action(result, "TRAIN IMMORTAL", allowed_actions=allowed_actions, max_count=1)

    if _apply_air_cleanup_actions(
        state=state,
        result=result,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
        context=context,
        map_name=map_name,
        rush_map=rush_map,
        step_index=step_index,
        enemy_air_memory=enemy_air_memory,
    ):
        return result

    if enable_warpgate_research:
        warpgate_status = int(state.get("warpgate_research_status", 0))
        if (
            warpgate_status == 0
            and cybernetics_core_count > 0
            and _can_afford("RESEARCH WARPGATERESEARCH", state)
        ):
            _ensure_action(result, "RESEARCH WARPGATERESEARCH", allowed_actions=allowed_actions, max_count=1)
    else:
        _remove_actions(result, action_names={"RESEARCH WARPGATERESEARCH"})

    max_gateway_queue_depth = int(policy_config.get("zealot_queue_depth", 1))
    max_gateway_queue_depth = max(0, min(4, max_gateway_queue_depth))
    # Keep queues shallow: deep queues can over-spend minerals and hide supply blocks.

    if not train or gateway_count <= 0 or max_gateway_queue_depth <= 0:
        _remove_actions(result, action_names={"TRAIN ZEALOT", "TRAIN STALKER"})
    else:
        max_gateway_queue = gateway_count * max_gateway_queue_depth
        gateway_queue_used = zealot_producing + stalker_producing
        available_slots = max(0, max_gateway_queue - gateway_queue_used)

        stalker_count = int(state.get("stalker_count", 0))
        stalker_total = stalker_count + stalker_producing
        if game_time_seconds < 240:
            stalker_target = 0
        elif game_time_seconds < 360:
            stalker_target = 4
        elif game_time_seconds < 480:
            stalker_target = 8
        else:
            stalker_target = 16
        if enemy_air_memory > 0:
            stalker_target = max(stalker_target, 6 + enemy_air_memory * 2)

        want_stalker = (
            cybernetics_core_count > 0
            and stalker_total < stalker_target
            and supply_left >= _supply_cost("TRAIN STALKER")
            and _can_afford("TRAIN STALKER", state)
        )
        stalker_actions = min(2 if (enemy_air_memory > 0 or ((not rush_map) and game_time_seconds >= 480)) else 1, available_slots) if want_stalker else 0
        _clamp_action_counts(result, action_name="TRAIN STALKER", max_count=stalker_actions)
        for _ in range(stalker_actions - result.count("TRAIN STALKER")):
            _ensure_action(
                result,
                "TRAIN STALKER",
                allowed_actions=allowed_actions,
                max_count=result.count("TRAIN STALKER") + 1,
                allow_replacement_when_full=False,
            )
        available_slots = max(0, available_slots - stalker_actions)

        can_train_zealot = (
            supply_left >= _supply_cost("TRAIN ZEALOT") and _can_afford("TRAIN ZEALOT", state)
        )
        if rush_map and early_ranged_rush:
            zealot_actions = min(1, available_slots) if can_train_zealot else 0
        else:
            zealot_actions = (
                (min(1, available_slots) if (enemy_air_memory > 0 or ((not rush_map) and stalker_total < stalker_target)) else min(3, available_slots))
                if can_train_zealot
                else 0
            )
        _clamp_action_counts(result, action_name="TRAIN ZEALOT", max_count=zealot_actions)
        for _ in range(zealot_actions - result.count("TRAIN ZEALOT")):
            _ensure_action(
                result,
                "TRAIN ZEALOT",
                allowed_actions=allowed_actions,
                max_count=result.count("TRAIN ZEALOT") + 1,
                allow_replacement_when_full=False,
            )

    apply_scouting_actions(
        state=state,
        result=result,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
        game_time_seconds=game_time_seconds,
        enemy_total_visible=enemy_total_visible,
        probe_count=probe_count,
        gateway_count=gateway_count,
        step_index=step_index,
        last_scout_step=last_scout_step,
    )
    apply_military_actions(
        state=state,
        result=result,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
        rush_map=rush_map,
        game_time_seconds=game_time_seconds,
        army_supply=army_supply,
        stalker_producing=stalker_producing,
        enemy_total_visible=enemy_total_visible,
        enemy_ground_threat=enemy_ground_threat,
        enemy_ranged_memory=enemy_ranged_memory,
        enemy_air_memory=enemy_air_memory,
        early_ranged_rush=early_ranged_rush,
        tech_plan=tech_plan,
        military=military,
        enable_retreat=enable_retreat,
        peak_army=peak_army,
        last_attack_step=last_attack_step,
        last_retreat_step=last_retreat_step,
        step_index=step_index,
        last_scout_step=last_scout_step,
        scout_count=scout_count,
        enemy_info_supported=enemy_info_supported,
    )
    if tech_plan == "zealot_only":
        tech_buildings = {
            "BUILD ROBOTICSFACILITY",
            "BUILD STARGATE",
            "BUILD TWILIGHTCOUNCIL",
            "BUILD TEMPLARARCHIVE",
            "BUILD DARKSHRINE",
            "BUILD ROBOTICSBAY",
            "BUILD FLEETBEACON",
        }
        # Flat64（rush_map）在少数 seed 下会拖到后期，地面部队可能找不到/到不了最后的建筑或农民。
        # 允许在很晚的时间点解锁 Stargate 作为“空军收尾”兜底，避免 29:43 超时。
        if rush_map and game_time_seconds >= int(policy_config.get("rush_map_air_cleanup_min_time_seconds", 900)):
            tech_buildings.discard("BUILD STARGATE")
        _remove_actions(result, action_names=tech_buildings)
    if (not disable_static_defense) and build_defense and "BUILD PHOTONCANNON" in (allowed_actions or {"BUILD PHOTONCANNON"}):
        _ensure_action(result, "BUILD PHOTONCANNON", allowed_actions=allowed_actions, max_count=1)
    if not train:
        _remove_actions(result, action_names={"TRAIN ZEALOT", "TRAIN STALKER", "TRAIN IMMORTAL"})
    return result
