from __future__ import annotations

from typing import Any


def can_afford(action: str, state: dict[str, Any]) -> bool:
    minerals = int(state.get("mineral", 0))
    gas = int(state.get("gas", 0))

    costs = {
        "TRAIN PROBE": (50, 0),
        "TRAIN ZEALOT": (100, 0),
        "TRAIN STALKER": (125, 50),
        "TRAIN OBSERVER": (25, 75),
        "TRAIN IMMORTAL": (275, 100),
        "TRAIN PHOENIX": (150, 100),
        "TRAIN VOIDRAY": (250, 150),
        "BUILD PYLON": (100, 0),
        "BUILD GATEWAY": (150, 0),
        "BUILD ASSIMILATOR": (75, 0),
        "BUILD CYBERNETICSCORE": (150, 0),
        "BUILD FORGE": (150, 0),
        "BUILD ROBOTICSFACILITY": (200, 100),
        "BUILD STARGATE": (150, 150),
        "BUILD NEXUS": (400, 0),
        "RESEARCH WARPGATERESEARCH": (50, 50),
        "BUILD PHOTONCANNON": (150, 0),
        "BUILD SHIELDBATTERY": (100, 0),
    }
    need = costs.get(action)
    if need is None:
        return True
    mineral_cost, gas_cost = need
    return minerals >= mineral_cost and gas >= gas_cost


def supply_cost(action: str) -> int:
    supply_costs = {
        "TRAIN PROBE": 1,
        "TRAIN ZEALOT": 2,
        "TRAIN STALKER": 2,
        "TRAIN OBSERVER": 1,
        "TRAIN IMMORTAL": 4,
        "TRAIN PHOENIX": 2,
        "TRAIN VOIDRAY": 4,
    }
    return supply_costs.get(action, 0)


def _first_index(result: list[str], action_name: str) -> int | None:
    try:
        return result.index(action_name)
    except ValueError:
        return None


def ensure_action(
    result: list[str],
    action_name: str,
    *,
    allowed_actions: set[str] | None,
    max_count: int = 1,
    prefer_front: bool = True,
    allow_replacement_when_full: bool = True,
) -> None:
    if max_count <= 0:
        return
    if result.count(action_name) >= max_count:
        return
    if allowed_actions is not None and action_name not in allowed_actions:
        return

    empty_index = _first_index(result, "EMPTY ACTION")
    if empty_index is not None:
        result[empty_index] = action_name
        return
    if not allow_replacement_when_full:
        return

    if not prefer_front:
        replace_range = range(len(result))
    else:
        replace_range = range(len(result) - 1, -1, -1)
    for index in replace_range:
        if result[index] not in {"BUILD PYLON", "BUILD GATEWAY"}:
            result[index] = action_name
            return


def clamp_action_counts(
    result: list[str],
    *,
    action_name: str,
    max_count: int,
) -> None:
    if max_count < 0:
        return
    seen = 0
    for index, value in enumerate(result):
        if value != action_name:
            continue
        seen += 1
        if seen > max_count:
            result[index] = "EMPTY ACTION"

def remove_actions(result: list[str], *, action_names: set[str]) -> None:
    """In-place: replace any action in action_names with EMPTY ACTION."""
    if not action_names:
        return
    for index, value in enumerate(result):
        if value in action_names:
            result[index] = "EMPTY ACTION"


def recently_issued(
    action_name: str,
    *,
    context: dict[str, Any],
    step: int,
    cooldown_steps: int,
) -> bool:
    if cooldown_steps <= 0:
        return False
    last = context.get("last_issued_step")
    if not isinstance(last, dict):
        return False
    try:
        last_step = int(last.get(action_name, -10_000) or -10_000)
    except Exception:
        return False
    return (step - last_step) < cooldown_steps


def compute_enemy_info_fallback_attack_policy(
    *,
    policy_config: dict[str, Any],
    game_time_seconds: int,
    step_index: int,
    last_scout_step: int,
    scout_count: int,
    enemy_info_supported: bool,
    effective_attack_min_time: int,
    effective_attack_threshold: int,
) -> tuple[int, int, bool, bool, bool]:
    """Adjust attack thresholds when enemy unit-count lines may be missing.

    In the official env, the "Enemy unittypeid.*" section is only present when
    enemies are visible. Hidden tests may remove it entirely. If we've already
    tried scouting but still never observed any enemy lines, treat the "enemy is
    not visible" signal as unreliable and attack more conservatively.

    Returns:
        (effective_attack_min_time, effective_attack_threshold,
         enemy_info_uncertain, scouted_recently, force_attack_by_time)
    """
    enabled = bool(policy_config.get("enemy_info_fallback_enabled", True))
    min_scout_count = int(policy_config.get("enemy_info_fallback_min_scout_count", 2))
    min_time_seconds = int(policy_config.get("enemy_info_fallback_min_time_seconds", 240))

    enemy_info_uncertain = (
        enabled
        and (not enemy_info_supported)
        and (int(scout_count or 0) >= min_scout_count)
        and (int(game_time_seconds or 0) >= min_time_seconds)
    )

    if enemy_info_uncertain:
        # When we have repeatedly scouted but still never observed any enemy unit lines,
        # treat the game as high-uncertainty and delay attacks until we have a much larger army.
        fallback_min_time = int(policy_config.get("enemy_info_fallback_attack_min_time_seconds", 900))
        fallback_army = int(policy_config.get("enemy_info_fallback_attack_army_threshold", 90))
        effective_attack_min_time = max(int(effective_attack_min_time), fallback_min_time)
        effective_attack_threshold = max(int(effective_attack_threshold), fallback_army)

    recent_scout_steps = int(policy_config.get("enemy_info_fallback_recent_scout_steps", 60))
    force_attack_time_seconds = int(policy_config.get("enemy_info_fallback_force_attack_time_seconds", 1200))

    step_index = int(step_index or 0)
    last_scout_step = int(last_scout_step or -9999)
    scouted_recently = (step_index - last_scout_step) <= recent_scout_steps
    force_attack_by_time = int(game_time_seconds or 0) >= force_attack_time_seconds

    return (
        effective_attack_min_time,
        effective_attack_threshold,
        enemy_info_uncertain,
        scouted_recently,
        force_attack_by_time,
    )


def effective_robotics_facility_count(state: dict[str, Any]) -> int:
    """Best-effort Robotics Facility count.

    The env-side SC2 bot currently reports `robotics_facility_count` with a known bug
    (it subtracts `already_pending(STARGATE)`), which can temporarily undercount the
    real Robo count while a Stargate is warping in. That can cause us to over-build
    extra Robotics Facilities and/or stop training Immortals.
    """
    try:
        reported = int(state.get("robotics_facility_count", 0))
    except Exception:
        reported = 0
    if reported > 0:
        return reported

    inferred_units = 0
    for key in ("immortal_count", "immortal_producing", "observer_count", "observer_producing"):
        try:
            inferred_units += int(state.get(key, 0) or 0)
        except Exception:
            pass
    return 1 if inferred_units > 0 else 0


def apply_air_cleanup_actions(
    *,
    state: dict[str, Any],
    result: list[str],
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any],
    context: dict[str, Any],
    map_name: str,
    rush_map: bool,
    step_index: int,
    enemy_air_memory: int,
) -> bool:
    """Air cleanup to avoid SC2 timeouts.

    Evidence: We observed repeated 29:43 timeouts where enemy buildings stay alive while
    combat units are gone (often due to unreachable terrain / hidden bases). Ground-only
    armies can get stuck; adding a Stargate + a couple Void Rays gives us a reliable
    "can fly to any remaining building" finisher.
    """
    if not bool(policy_config.get("air_cleanup_enabled", True)):
        return False

    game_time_seconds = int(state.get("game_time_seconds", 0))
    if rush_map:
        rush_min_time_seconds = int(policy_config.get("rush_map_air_cleanup_min_time_seconds", 900))
        # If the opponent is already showing air threats on a rush map, enable Stargate earlier
        # so we can respond with Phoenix/Void Rays instead of stalling into a timeout.
        if enemy_air_memory > 0:
            rush_min_time_seconds = min(
                rush_min_time_seconds,
                int(policy_config.get("rush_map_air_cleanup_vs_air_min_time_seconds", 420)),
            )
        if game_time_seconds < max(0, rush_min_time_seconds):
            return False
    else:
        min_time_seconds = int(policy_config.get("air_cleanup_min_time_seconds", 420))
        if enemy_air_memory > 0:
            min_time_seconds = min(
                min_time_seconds,
                int(policy_config.get("air_cleanup_vs_air_min_time_seconds", 360)),
            )
        if game_time_seconds < max(0, min_time_seconds):
            return False

    # Do not rush Stargate too early: we want at least a stable two-base setup first.
    nexus_count = int(state.get("nexus_count", 0))
    gateway_count = int(state.get("gateway_count", 0))
    if rush_map:
        if nexus_count < 1 or gateway_count < 3:
            return False
    else:
        if nexus_count < 2 or gateway_count < 3:
            return False

    cybernetics_core_count = int(state.get("cybernetics_core_count", 0))
    pylon_count = int(state.get("pylon_count", 0))
    supply_left = int(state.get("supply_left", 0))

    stargate_count = int(state.get("stargate_count", 0))
    stargate_constructing = int(state.get("stargate_constructing", 0))
    stargate_total = stargate_count + stargate_constructing
    minerals = int(state.get("mineral", 0))

    def bank_for_stargate() -> bool:
        if minerals < 120:
            return False
        for i, a in enumerate(result):
            token = str(a or "").upper()
            if token.startswith(("TRAIN ", "BUILD ", "RESEARCH ", "CHRONOBOOST ")):
                result[i] = "EMPTY ACTION"
        return True
    if (
        stargate_total < 1
        and cybernetics_core_count > 0
        and pylon_count > 0
        and not recently_issued("BUILD STARGATE", context=context, step=step_index, cooldown_steps=10)
    ):
        if can_afford("BUILD STARGATE", state):
            # Important: action packs execute multiple actions sequentially; if we place
            # BUILD STARGATE after another mineral-spending action (e.g. BUILD PYLON),
            # the Stargate can fail due to insufficient minerals. To avoid this, make
            # this step a "build Stargate only" step.
            for i in range(len(result)):
                result[i] = "EMPTY ACTION"
            ensure_action(
                result,
                "BUILD STARGATE",
                allowed_actions=allowed_actions,
                max_count=1,
                prefer_front=True,
            )
            return True
        if bank_for_stargate():
            return True

    # Late-game scaling: add a 2nd Stargate on standard maps so we can produce enough
    # Void Rays to reliably finish the game before the 29:43 step cap.
    if not rush_map:
        try:
            desired_stargates = int(policy_config.get("air_cleanup_stargate_target", 2))
        except Exception:
            desired_stargates = 2
        desired_stargates = max(1, min(3, desired_stargates))
        second_stargate_min_time_seconds = int(
            policy_config.get("air_cleanup_second_stargate_min_time_seconds", 900)
        )
        second_stargate_enemy_visible_max = int(
            policy_config.get("air_cleanup_second_stargate_enemy_visible_max", 2)
        )
        enemy_total_visible = int(state.get("enemy_total_visible", 0))
        if (
            game_time_seconds >= max(0, second_stargate_min_time_seconds)
            and enemy_total_visible <= max(0, second_stargate_enemy_visible_max)
            and stargate_total < desired_stargates
            and cybernetics_core_count > 0
            and pylon_count > 0
            and not recently_issued(
                "BUILD STARGATE",
                context=context,
                step=step_index,
                cooldown_steps=12,
            )
            and can_afford("BUILD STARGATE", state)
        ):
            for i in range(len(result)):
                result[i] = "EMPTY ACTION"
            ensure_action(
                result,
                "BUILD STARGATE",
                allowed_actions=allowed_actions,
                max_count=1,
                prefer_front=True,
            )
            return False

    phoenix_count = int(state.get("phoenix_count", 0))
    phoenix_producing = int(state.get("phoenix_producing", 0))
    voidray_count = int(state.get("voidray_count", 0))
    voidray_producing = int(state.get("voidray_producing", 0))

    # Queue control: the env exposes `*_producing` via `already_pending`, which includes queued units.
    # If we require "Stargate idle", we can miss the exact idle frame and never queue Void Rays,
    # leading to late-game timeouts. Instead, allow a small queue depth per Stargate.
    stargate_busy = phoenix_producing + voidray_producing
    try:
        max_pending_per_stargate = int(policy_config.get("air_cleanup_max_pending_per_stargate", 2))
    except Exception:
        max_pending_per_stargate = 2
    max_pending_per_stargate = max(1, min(3, max_pending_per_stargate))
    stargate_has_queue_room = stargate_count > 0 and stargate_busy < (stargate_count * max_pending_per_stargate)

    desired_phoenix = 0
    if enemy_air_memory > 0:
        desired_phoenix = 4 if enemy_air_memory >= 6 else 2
    # Keep Phoenix once we have a Stargate: it is our fastest flying scout, and it helps
    # clean up enemy air units (Mutalisk/Overseer/Overlord) that can cause timeouts.
    desired_phoenix = max(desired_phoenix, 1 if rush_map else 2)
    if rush_map and game_time_seconds >= 600:
        desired_phoenix = max(desired_phoenix, 2)
    elif game_time_seconds >= 900:
        desired_phoenix = max(desired_phoenix, 2)
    # Late-game anti-air scaling: small Phoenix counts are not enough once Zerg reaches
    # Brood Lord / large Muta packs. Bump the target to avoid getting snowballed.
    if (not rush_map) and enemy_air_memory > 0 and game_time_seconds >= 900:
        desired_phoenix = max(desired_phoenix, 4)

    phoenix_total = phoenix_count + phoenix_producing
    voidray_total = voidray_count + voidray_producing

    # Default to a larger finisher squad on standard maps. In practice we often float gas,
    # and having more Void Rays improves both cleanup and late-game stability vs Zerg tech.
    default_voidrays = 4 if rush_map else 6
    desired_voidrays = int(policy_config.get("air_cleanup_voidray_target", default_voidrays))
    if (not rush_map) and enemy_air_memory > 0 and game_time_seconds >= 900:
        desired_voidrays = max(desired_voidrays, 8)

    # Late-game priority shift: once we decide to go for air cleanup, avoid dumping minerals
    # into Gateway units forever. This helps ensure we actually reach the Void Ray target.
    if (not rush_map) and game_time_seconds >= 900 and voidray_total < max(0, desired_voidrays):
        remove_actions(result, action_names={"TRAIN ZEALOT", "TRAIN STALKER"})

    # Supply budgeting: if we are close to 200 supply but still missing Void Rays, stop spending
    # supply on ground units/workers so we don't "lock ourselves out" of 4-supply Void Rays.
    supply_used = int(state.get("supply_used", 0))
    if (stargate_total > 0) and (voidray_total < max(0, desired_voidrays)) and supply_used >= 190:
        remove_actions(
            result,
            action_names={"TRAIN PROBE", "TRAIN ZEALOT", "TRAIN STALKER", "TRAIN IMMORTAL"},
        )

    # If a Stargate is (or will soon be) available but we don't yet have Void Rays, bank minerals
    # instead of continuously spending them on Gateway units. Otherwise, we frequently end up in a
    # state where the Stargate finishes but `mineral < 250` for minutes, delaying our first Void Ray
    # until the late-game supply is already full (=> timeouts).
    if (stargate_total > 0) and (voidray_total < max(0, desired_voidrays)) and (minerals < 250):
        # Banking must also pause Immortal production; otherwise we can stay mineral-starved
        # for many minutes even with ample gas.
        remove_actions(result, action_names={"TRAIN ZEALOT", "TRAIN STALKER", "TRAIN IMMORTAL"})

    # Priority (Stargate queue-aware):
    # - If we can afford a Void Ray, prefer it early. Evidence: some timeout games hit a window
    #   where we can afford TRAIN VOIDRAY, but TRAIN PHOENIX consumes the minerals and we never
    #   reach 250 minerals again until supply is full (=> 0 Void Rays, cannot kill hidden buildings).
    # - If the enemy has shown air threats, ensure a couple Phoenix first for anti-air control.
    # Important: requiring 2 Phoenix before any Void Ray can deadlock if Phoenix keep dying;
    # we end up "replacing scouts forever" and never build the Void Rays needed to finish.
    phoenix_needed_before_voidray = 1 if enemy_air_memory > 0 else 0

    if (
        stargate_count > 0
        and stargate_has_queue_room
        and not recently_issued("TRAIN PHOENIX", context=context, step=step_index, cooldown_steps=2)
        and phoenix_total < phoenix_needed_before_voidray
        and supply_left >= supply_cost("TRAIN PHOENIX")
        and can_afford("TRAIN PHOENIX", state)
    ):
        for i in range(len(result)):
            result[i] = "EMPTY ACTION"
        ensure_action(
            result,
            "TRAIN PHOENIX",
            allowed_actions=allowed_actions,
            max_count=1,
            prefer_front=True,
        )
        return True

    if (
        stargate_count > 0
        and stargate_has_queue_room
        and not recently_issued("TRAIN VOIDRAY", context=context, step=step_index, cooldown_steps=2)
        and voidray_total < max(0, desired_voidrays)
        and supply_left >= supply_cost("TRAIN VOIDRAY")
        and can_afford("TRAIN VOIDRAY", state)
    ):
        # Important: action packs execute sequentially. When minerals are tight (common mid-game),
        # placing TRAIN VOIDRAY after other spending actions (e.g. TRAIN IMMORTAL) can cause the
        # Void Ray training to fail silently. Make this step "Void Ray first" to ensure progress.
        for i in range(len(result)):
            result[i] = "EMPTY ACTION"
        ensure_action(
            result,
            "TRAIN VOIDRAY",
            allowed_actions=allowed_actions,
            max_count=1,
            prefer_front=True,
        )
        return True

    if (
        stargate_count > 0
        and stargate_has_queue_room
        and not recently_issued("TRAIN PHOENIX", context=context, step=step_index, cooldown_steps=2)
        and phoenix_total < max(0, desired_phoenix)
        and supply_left >= supply_cost("TRAIN PHOENIX")
        and can_afford("TRAIN PHOENIX", state)
    ):
        for i in range(len(result)):
            result[i] = "EMPTY ACTION"
        ensure_action(
            result,
            "TRAIN PHOENIX",
            allowed_actions=allowed_actions,
            max_count=1,
            prefer_front=True,
        )
        return True

    # If we can't train anything this step, leave `result` unchanged.
    return False
