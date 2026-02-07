"""SC2 策略工具箱 - VLM选择策略，程序执行"""
from __future__ import annotations

import re
from typing import Any

from agents.crowsphere.sc2_macro_policy import postprocess_actions
from agents.crowsphere.sc2_macro_policy_utils import (
    can_afford as _can_afford,
    ensure_action as _ensure_action,
    remove_actions as _remove_actions,
    supply_cost as _supply_cost,
)
from agents.crowsphere.sc2_enemy_classifier import aggregate_enemy_units


def _parse_int(obs_str: str, pattern: str) -> int:
    match = re.search(pattern, obs_str, re.IGNORECASE | re.MULTILINE)
    return int(match.group(1)) if match else 0


def _parse_game_time_to_seconds(game_time: str) -> int:
    match = re.search(r"(\d+):(\d+)", str(game_time))
    if not match:
        return 0
    minutes = int(match.group(1))
    seconds = int(match.group(2))
    return minutes * 60 + seconds


def parse_game_state(obs_str: str) -> dict[str, Any]:
    """从 obs_str 解析游戏状态"""
    state: dict[str, Any] = {}

    patterns = {
        "nexus_count": r"(?:^|\n)\s*-\s*Nexus count:\s*(\d+)",
        "worker_supply": r"(?:^|\n)\s*-\s*Worker supply:\s*(\d+)",
        "gateway_count": r"(?:^|\n)\s*-\s*Gateway count:\s*(\d+)",
        "probe_count": r"(?:^|\n)\s*-\s*Probe count:\s*(\d+)",
        "army_supply": r"(?:^|\n)\s*-\s*Army supply:\s*(\d+)",
        "pylon_count": r"(?:^|\n)\s*-\s*Pylon count:\s*(\d+)",
        "zealot_count": r"(?:^|\n)\s*-\s*Zealot count:\s*(\d+)",
        "stalker_count": r"(?:^|\n)\s*-\s*Stalker count:\s*(\d+)",
        "observer_count": r"(?:^|\n)\s*-\s*Observer count:\s*(\d+)",
        "immortal_count": r"(?:^|\n)\s*-\s*Immortal count:\s*(\d+)",
        # `StarCraftObs.to_text()` currently prints the building key as "Statgate count"
        # (typo from env side). Accept both "Stargate" and "Statgate" to be robust.
        "stargate_count": r"(?:^|\n)\s*-\s*Sta(?:r|t)gate count:\s*(\d+)",
        "gas_buildings_count": r"(?:^|\n)\s*-\s*Gas buildings count:\s*(\d+)",
        "cybernetics_core_count": r"(?:^|\n)\s*-\s*Cybernetics\s*core count:\s*(\d+)",
        "robotics_facility_count": r"(?:^|\n)\s*-\s*Robotics\s*facility count:\s*(\d+)",
        "warp_gate_count": r"(?:^|\n)\s*-\s*Warp\s*gate count:\s*(\d+)",
        "photon_cannon_count": r"(?:^|\n)\s*-\s*Photon\s*cannon count:\s*(\d+)",
        "shield_battery_count": r"(?:^|\n)\s*-\s*Shield\s*battery count:\s*(\d+)",
        "mineral": r"(?:^|\n)\s*-\s*Mineral:\s*(\d+)",
        "gas": r"(?:^|\n)\s*-\s*Gas:\s*(\d+)",
        "supply_left": r"(?:^|\n)\s*-\s*Supply left:\s*(-?\d+)",
        "supply_cap": r"(?:^|\n)\s*-\s*Supply cap:\s*(\d+)",
        "supply_used": r"(?:^|\n)\s*-\s*Supply used:\s*(\d+)",
        "phoenix_count": r"(?:^|\n)\s*-\s*Phoenix count:\s*(\d+)",
        "voidray_count": r"(?:^|\n)\s*-\s*Voidray count:\s*(\d+)",
    }

    for key, pattern in patterns.items():
        state[key] = _parse_int(obs_str, pattern)

    # 解析游戏时间
    time_match = re.search(r"(?:At|Game time:?)\s*(\d+:\d+)", obs_str, re.IGNORECASE)
    state["game_time"] = time_match.group(1) if time_match else "00:00"
    state["game_time_seconds"] = _parse_game_time_to_seconds(state["game_time"])

    # 解析建造中的建筑
    constructing_patterns = {
        "nexus_constructing": r"(?:^|\n)\s*-\s*Constructing nexus count:\s*(\d+)",
        "gateway_constructing": r"(?:^|\n)\s*-\s*Constructing gateway count:\s*(\d+)",
        "pylon_constructing": r"(?:^|\n)\s*-\s*Constructing pylon count:\s*(\d+)",
        "gas_buildings_constructing": r"(?:^|\n)\s*-\s*Constructing gas buildings count:\s*(\d+)",
        "cybernetics_core_constructing": r"(?:^|\n)\s*-\s*Constructing cybernetics\s*core count:\s*(\d+)",
        "robotics_facility_constructing": r"(?:^|\n)\s*-\s*Constructing robotics\s*facility count:\s*(\d+)",
        "stargate_constructing": r"(?:^|\n)\s*-\s*Constructing stargate count:\s*(\d+)",
        "forge_constructing": r"(?:^|\n)\s*-\s*Constructing forge count:\s*(\d+)",
        "photon_cannon_constructing": r"(?:^|\n)\s*-\s*Constructing photon\s*cannon count:\s*(\d+)",
        "shield_battery_constructing": r"(?:^|\n)\s*-\s*Constructing shield\s*battery count:\s*(\d+)",
    }
    for key, pattern in constructing_patterns.items():
        state[key] = _parse_int(obs_str, pattern)

    # 解析生产队列（用于“队列满还在点训练”的过滤）
    producing_patterns = {
        "probe_producing": r"(?:^|\n)\s*-\s*Producing probe count:\s*(\d+)",
        "zealot_producing": r"(?:^|\n)\s*-\s*Producing zealot count:\s*(\d+)",
        "stalker_producing": r"(?:^|\n)\s*-\s*Producing stalker count:\s*(\d+)",
        "observer_producing": r"(?:^|\n)\s*-\s*Producing observer count:\s*(\d+)",
        "immortal_producing": r"(?:^|\n)\s*-\s*Producing immortal count:\s*(\d+)",
        "phoenix_producing": r"(?:^|\n)\s*-\s*Producing phoenix count:\s*(\d+)",
        "voidray_producing": r"(?:^|\n)\s*-\s*Producing voidray count:\s*(\d+)",
    }
    for key, pattern in producing_patterns.items():
        state[key] = _parse_int(obs_str, pattern)

    # 解析关键研究
    state["warpgate_research_status"] = _parse_int(
        obs_str, r"(?:^|\n)\s*-\s*Warpgate research status:\s*(\d+)"
    )

    # 解析敌方单位
    # Use a bullet-anchored regex to avoid confusing our own `* count:` with `Enemy unittypeid.*`.
    enemy_units: dict[str, int] = {}
    for match in re.finditer(
        r"(?:^|\n)\s*-\s*Enemy unittypeid\.([a-z0-9_]+)\s*:\s*(\d+)",
        obs_str,
        re.IGNORECASE,
    ):
        unit_name = match.group(1).lower()
        count = int(match.group(2))
        if count <= 0:
            continue
        enemy_units[unit_name] = count
    state["enemy_units"] = enemy_units

    # Backwards-compatible Zerg-focused fields (useful for debugging/log analysis).
    state["enemy_roach"] = int(enemy_units.get("roach", 0))
    state["enemy_hydralisk"] = int(enemy_units.get("hydralisk", 0))
    state["enemy_zergling"] = int(enemy_units.get("zergling", 0))
    state["enemy_mutalisk"] = int(enemy_units.get("mutalisk", 0))
    state["enemy_corruptor"] = int(enemy_units.get("corruptor", 0))

    # Generalized threat aggregation to support bot_race != Zerg.
    agg = aggregate_enemy_units(enemy_units)
    state["enemy_ground_ranged_threat"] = agg["ground_ranged"]
    state["enemy_ground_melee_threat"] = agg["ground_melee"]
    state["enemy_ground_threat"] = agg["ground_total"]
    state["enemy_air_threat"] = agg["air"]
    state["enemy_total_visible"] = agg["total_combat"]

    # Keep old key name used by legacy logic.
    state["enemy_ranged_threat"] = state["enemy_ground_ranged_threat"]

    # 解析Forge数量
    state["forge_count"] = _parse_int(obs_str, r"(?:^|\n)\s*-\s*Forge count:\s*(\d+)")

    # 计算人口状态
    supply_left = state.get("supply_left", 0)
    pylon_constructing = state.get("pylon_constructing", 0)
    gateway_count = state.get("gateway_count", 0) + state.get("warp_gate_count", 0)
    production_rate = gateway_count * 2
    pylon_threshold = max(4, production_rate * 2)

    state["supply_blocked"] = (
        supply_left <= pylon_threshold and pylon_constructing == 0
    )
    state["supply_critical"] = supply_left <= 0

    return state


def apply_strategy(
    state: dict[str, Any],
    actions: list[str],
    *,
    attack: bool = False,
) -> list[str]:
    """根据VLM决策执行策略（兼容旧接口）"""
    military = "attack" if attack else "none"
    return apply_strategy_combo(state, actions, train=True, build_defense=False, military=military)


def apply_strategy_combo(
    state: dict[str, Any],
    actions: list[str],
    *,
    train: bool = True,
    build_defense: bool = False,
    military: str = "none",
    allowed_actions: set[str] | None = None,
    policy_config: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> list[str]:
    """根据VLM策略组合执行调整

    决策维度：
    - train: 是否继续训练单位
    - build_defense: 是否建造防御塔
    - military: attack=进攻 / defend=用单位防守 / none=不行动

    程序做"安全网"：
    - 防止卡人口（并尽量让 BUILD PYLON 命中官方条件）
    - 经济与产能：补 Probe / 补 Gateway
    - 生产：避免“队列满还在点训练”
    - 可选：科技（Immortal）与进攻节奏
    """
    result = postprocess_actions(
        state=state,
        actions=actions,
        train=train,
        build_defense=build_defense,
        military=military,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
        context=context,
    )
    _maybe_add_detection_actions(
        state=state,
        actions=result,
        allowed_actions=allowed_actions,
        policy_config=policy_config,
    )
    return result


def _maybe_add_detection_actions(
    *,
    state: dict[str, Any],
    actions: list[str],
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any] | None,
) -> None:
    """Add minimal detection (Observer) when the opponent can field burrowed threats.

    Evidence: some losses happen late-game on standard maps when the enemy reaches
    Lurker/Swarm Host tech; without detection, attacks stall and we bleed out.
    """
    if "BUILD NEXUS" in actions:
        return
    policy_config = policy_config or {}
    if not bool(policy_config.get("detection_enabled", True)):
        return

    robo = int(state.get("robotics_facility_count", 0))
    if robo <= 0:
        _remove_actions(actions, action_names={"TRAIN OBSERVER"})
        return

    observer_total = int(state.get("observer_count", 0)) + int(state.get("observer_producing", 0))
    enemy_units = state.get("enemy_units") if isinstance(state.get("enemy_units"), dict) else {}
    wants_extra = any(k in enemy_units for k in ("lurkermp", "lurkermpburrowed", "swarmhostmp", "locustmp"))
    desired = 2 if wants_extra else 1
    if observer_total >= desired:
        _remove_actions(actions, action_names={"TRAIN OBSERVER"})
        return

    supply_left = int(state.get("supply_left", 0))
    if supply_left < _supply_cost("TRAIN OBSERVER") or (not _can_afford("TRAIN OBSERVER", state)):
        _remove_actions(actions, action_names={"TRAIN OBSERVER"})
        return

    _ensure_action(actions, "TRAIN OBSERVER", allowed_actions=allowed_actions, max_count=1)


def build_tools_prompt_section() -> str:
    """构建策略提示（兼容旧接口）"""
    return (
        "战术决策：\n"
        "attack=true: 进攻（actions中包含MULTI-ATTACK）\n"
        "attack=false: 发展/撤退（专注生产）\n"
    )
