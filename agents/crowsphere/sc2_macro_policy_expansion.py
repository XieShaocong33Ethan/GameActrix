from __future__ import annotations

from typing import Any

from agents.crowsphere.sc2_macro_policy_utils import (
    can_afford as _can_afford,
    ensure_action as _ensure_action,
    remove_actions as _remove_actions,
    recently_issued as _recently_issued,
)


def apply_expansion_actions(
    *,
    state: dict[str, Any],
    result: list[str],
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any],
    context: dict[str, Any],
    # Map / strategy
    rush_map: bool,
    macro_policy: str,
    # Current situation
    game_time_seconds: int,
    step_index: int,
    army_supply: int,
    probe_count: int,
    gateway_total: int,
    supply_blocked: bool,
    enemy_total_visible: int,
) -> None:
    """Decide BUILD NEXUS (2nd base timing + late-game 3rd base fallback).

    Evidence: On standard maps, if we don't end the game by ~15-20 minutes, two bases
    can mine out and we get stuck with high gas but near-zero minerals. That leads to
    unstable late-game losses/ties. A controlled third Nexus helps sustain production.
    """
    nexus_count = int(state.get("nexus_count", 0))
    nexus_constructing = int(state.get("nexus_constructing", 0))
    nexus_total = nexus_count + nexus_constructing

    build_nexus_cooldown_steps = 12
    minerals = int(state.get("mineral", 0))
    can_try_build_nexus = (
        nexus_constructing <= 0
        and not _recently_issued(
            "BUILD NEXUS",
            context=context,
            step=step_index,
            cooldown_steps=build_nexus_cooldown_steps,
        )
        and _can_afford("BUILD NEXUS", state)
    )

    def bank_for_nexus() -> None:
        """Temporarily stop spending minerals so we can reach 400 for BUILD NEXUS.

        Evidence: on Flat64 and occasionally on standard maps, we satisfy all expansion
        conditions except minerals>=400. If we keep spamming unit production, minerals
        can stay <400 for minutes -> never expand -> unstable losses/timeouts.
        """
        # Flat64（rush_map）矿更紧，提早一点存钱更容易触发二矿窗口。
        if minerals < (250 if rush_map else 300):
            return
        for i, a in enumerate(result):
            # Keep non-economic commands like MULTI-ATTACK/SCOUTING; pause spending actions.
            token = str(a or "").upper()
            if token.startswith(("TRAIN ", "BUILD ", "RESEARCH ", "CHRONOBOOST ")):
                result[i] = "EMPTY ACTION"

    try:
        expand_enemy_visible_limit = int(policy_config.get("expand_enemy_visible_limit", 6))
    except Exception:
        expand_enemy_visible_limit = 6
    try:
        third_expand_enemy_visible_limit = int(policy_config.get("third_expand_enemy_visible_limit", 20))
    except Exception:
        third_expand_enemy_visible_limit = 20

    should_expand = False
    if macro_policy == "two_base":
        # Two-base opening: get the 2nd Nexus early enough to avoid missing the mineral window.
        if nexus_total < 2:
            # Flat64 is a rush map: expanding too early can be fatal, but never expanding
            # makes long games extremely unstable (mineral starvation + cleanup failures).
            if rush_map:
                expand_min_time_seconds = int(policy_config.get("rush_map_expand_min_time_seconds", 240))
                min_probes = int(policy_config.get("rush_map_expand_min_probe_count", 16))
                min_army = int(policy_config.get("rush_map_expand_min_army_supply", 10))
                # Flat64 距离近，enemy_total_visible 往往略高；对二矿的可见敌军门槛适当放宽更稳。
                expand_enemy_visible_limit = int(policy_config.get("rush_map_expand_enemy_visible_limit", 12))
            else:
                expand_min_time_seconds = 150
                min_probes = 18
                min_army = 0
            should_expand = (
                game_time_seconds >= expand_min_time_seconds
                and gateway_total >= 1
                and probe_count >= min_probes
                and (not supply_blocked)
                and (army_supply >= min_army)
                and enemy_total_visible <= expand_enemy_visible_limit
                and can_try_build_nexus
            )
            if (
                (not should_expand)
                and game_time_seconds >= expand_min_time_seconds
                and gateway_total >= 1
                and probe_count >= min_probes
                and (not supply_blocked)
                and (army_supply >= min_army)
                and enemy_total_visible <= expand_enemy_visible_limit
                and nexus_constructing <= 0
                and (not _recently_issued("BUILD NEXUS", context=context, step=step_index, cooldown_steps=build_nexus_cooldown_steps))
                and (not _can_afford("BUILD NEXUS", state))
            ):
                bank_for_nexus()
            if (not should_expand) and rush_map:
                # Late-game safety valve: if the rush map drags on, staying on one base
                # often leads to mineral starvation and cleanup timeouts.
                force_time_seconds = int(policy_config.get("rush_map_force_expand_min_time_seconds", 900))
                force_min_army = int(policy_config.get("rush_map_force_expand_min_army_supply", 30))
                force_enemy_visible_limit = int(
                    policy_config.get("rush_map_force_expand_enemy_visible_limit", 20)
                )
                should_expand = (
                    game_time_seconds >= force_time_seconds
                    and gateway_total >= 1
                    and probe_count >= min_probes
                    and army_supply >= force_min_army
                    and enemy_total_visible <= force_enemy_visible_limit
                    and can_try_build_nexus
                )
                if (
                    (not should_expand)
                    and game_time_seconds >= force_time_seconds
                    and gateway_total >= 1
                    and probe_count >= min_probes
                    and army_supply >= force_min_army
                    and enemy_total_visible <= force_enemy_visible_limit
                    and nexus_constructing <= 0
                    and (not _recently_issued("BUILD NEXUS", context=context, step=step_index, cooldown_steps=build_nexus_cooldown_steps))
                    and (not _can_afford("BUILD NEXUS", state))
                ):
                    bank_for_nexus()
        # Standard maps: if we are still not finished deep into the game, add a 3rd base
        # to avoid mineral starvation and stabilize late-game.
        elif (not rush_map) and nexus_total < 3:
            third_min_time_seconds = int(policy_config.get("third_nexus_min_time_seconds", 780))
            third_min_army = int(policy_config.get("third_nexus_min_army_supply", 60))
            third_min_probes = int(policy_config.get("third_nexus_min_probe_count", 40))
            should_expand = (
                game_time_seconds >= third_min_time_seconds
                and gateway_total >= 3
                and probe_count >= third_min_probes
                and army_supply >= third_min_army
                and enemy_total_visible <= third_expand_enemy_visible_limit
                and can_try_build_nexus
            )
            if (
                (not should_expand)
                and game_time_seconds >= third_min_time_seconds
                and gateway_total >= 3
                and probe_count >= third_min_probes
                and army_supply >= third_min_army
                and enemy_total_visible <= third_expand_enemy_visible_limit
                and nexus_constructing <= 0
                and (not _recently_issued("BUILD NEXUS", context=context, step=step_index, cooldown_steps=build_nexus_cooldown_steps))
                and (not _can_afford("BUILD NEXUS", state))
            ):
                bank_for_nexus()
            if not should_expand:
                # Late-game safety valve: if we are still on two bases very late,
                # force a third Nexus to avoid "gas rich / mineral poor" losses.
                force_time_seconds = int(policy_config.get("third_nexus_force_time_seconds", 960))
                force_enemy_visible_limit = int(
                    policy_config.get("third_nexus_force_enemy_visible_limit", 50)
                )
                should_expand = (
                    game_time_seconds >= force_time_seconds
                    and gateway_total >= 3
                    and probe_count >= third_min_probes
                    and army_supply >= third_min_army
                    and enemy_total_visible <= force_enemy_visible_limit
                    and can_try_build_nexus
                )
                if (
                    (not should_expand)
                    and game_time_seconds >= force_time_seconds
                    and gateway_total >= 3
                    and probe_count >= third_min_probes
                    and army_supply >= third_min_army
                    and enemy_total_visible <= force_enemy_visible_limit
                    and nexus_constructing <= 0
                    and (not _recently_issued("BUILD NEXUS", context=context, step=step_index, cooldown_steps=build_nexus_cooldown_steps))
                    and (not _can_afford("BUILD NEXUS", state))
                ):
                    bank_for_nexus()
    else:
        # Legacy behavior: delay the 2nd base more.
        should_expand = (
            nexus_total < 2
            and game_time_seconds >= 360
            and gateway_total >= 3
            and probe_count >= 20
            and enemy_total_visible <= expand_enemy_visible_limit
            and can_try_build_nexus
        )

    if should_expand:
        # Build Nexus must be "first and only" this step; otherwise earlier mineral spending
        # actions (e.g. BUILD PYLON / TRAIN ZEALOT) can make BUILD NEXUS fail silently.
        for i in range(len(result)):
            result[i] = "EMPTY ACTION"
        _ensure_action(
            result,
            "BUILD NEXUS",
            allowed_actions=allowed_actions,
            max_count=1,
        )
        return

    _remove_actions(result, action_names={"BUILD NEXUS"})
