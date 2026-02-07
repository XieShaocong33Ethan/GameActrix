from __future__ import annotations

from typing import Any

from agents.crowsphere.sc2_macro_policy_utils import (
    can_afford as _can_afford,
    ensure_action as _ensure_action,
    supply_cost as _supply_cost,
)


def apply_scouting_actions(
    *,
    state: dict[str, Any],
    result: list[str],
    allowed_actions: set[str] | None,
    policy_config: dict[str, Any],
    # Current situation
    game_time_seconds: int,
    enemy_total_visible: int,
    probe_count: int,
    gateway_count: int,
    # Context (cross-step memory)
    step_index: int,
    last_scout_step: int,
) -> None:
    """Add scouting actions (prefer air scouts late-game to avoid timeouts/ties).

    Evidence: We observed repeated ties on standard maps where our army is large but
    the game does not end before the map time limit. Late-game, a single Overlord
    (enemy_total_visible=1) can block our previous `enemy_total_visible==0` scouting
    gate, so we never search for the last hidden building/base.
    """
    if not bool(policy_config.get("scouting_enabled", True)):
        return

    scouting_interval_steps = int(policy_config.get("scouting_interval_steps", 40))
    scouting_min_time_seconds = int(policy_config.get("scouting_min_time_seconds", 180))

    cleanup_enabled = bool(policy_config.get("cleanup_scouting_enabled", True))
    cleanup_min_time_seconds = int(policy_config.get("cleanup_scouting_min_time_seconds", 600))
    cleanup_interval_steps = int(policy_config.get("cleanup_scouting_interval_steps", 15))
    cleanup_enemy_visible_max = int(policy_config.get("cleanup_scouting_enemy_visible_max", 2))

    if game_time_seconds < max(0, scouting_min_time_seconds):
        return

    steps_since_scout = step_index - int(last_scout_step or -9999)

    # Late-game cleanup scouting: allow a small amount of enemy visibility to avoid
    # getting stuck forever (e.g. a lone Overlord).
    if (
        cleanup_enabled
        and game_time_seconds >= max(0, cleanup_min_time_seconds)
        and enemy_total_visible <= max(0, cleanup_enemy_visible_max)
        and steps_since_scout >= max(1, cleanup_interval_steps)
    ):
        enemy_units = state.get("enemy_units") if isinstance(state.get("enemy_units"), dict) else {}
        extractor_only = (
            bool(enemy_units)
            and set(enemy_units.keys()) == {"extractor"}
            and int(enemy_units.get("extractor", 0) or 0) > 0
        )

        if extractor_only:
            # Phoenix is air-to-air only and cannot kill ground buildings like Extractors.
            # Send a ground unit to the enemy start location to finish the last structure.
            if int(state.get("zealot_count", 0) or 0) > 0:
                _ensure_action(
                    result,
                    "SCOUTING ZEALOT",
                    allowed_actions=allowed_actions,
                    max_count=1,
                )
                return
            if probe_count >= 14:
                _ensure_action(
                    result,
                    "SCOUTING PROBE",
                    allowed_actions=allowed_actions,
                    max_count=1,
                )
                return

        phoenix_count = int(state.get("phoenix_count", 0))
        observer_count = int(state.get("observer_count", 0))
        stargate_count = int(state.get("stargate_count", 0))
        supply_left = int(state.get("supply_left", 0))

        # Prefer a flying scout if we have one.
        if phoenix_count > 0:
            _ensure_action(
                result,
                "SCOUTING PHOENIX",
                allowed_actions=allowed_actions,
                max_count=1,
            )
            return
        if observer_count > 0:
            _ensure_action(
                result,
                "SCOUTING OBSERVER",
                allowed_actions=allowed_actions,
                max_count=1,
            )
            return

        # If we have a Stargate but no Phoenix yet, try to make a scout quickly.
        if (
            stargate_count > 0
            and supply_left >= _supply_cost("TRAIN PHOENIX")
            and _can_afford("TRAIN PHOENIX", state)
        ):
            _ensure_action(result, "TRAIN PHOENIX", allowed_actions=allowed_actions, max_count=1)
            return

        if probe_count >= 14:
            _ensure_action(
                result,
                "SCOUTING PROBE",
                allowed_actions=allowed_actions,
                max_count=1,
            )
            return
        if gateway_count > 0:
            _ensure_action(
                result,
                "SCOUTING ZEALOT",
                allowed_actions=allowed_actions,
                max_count=1,
            )
            return
        return

    # Early/mid-game scouting (more conservative): only scout when nothing is visible.
    if steps_since_scout < max(1, scouting_interval_steps) or enemy_total_visible != 0:
        return

    if probe_count >= 14:
        _ensure_action(
            result,
            "SCOUTING PROBE",
            allowed_actions=allowed_actions,
            max_count=1,
            allow_replacement_when_full=False,
        )
        return
    if gateway_count > 0:
        _ensure_action(
            result,
            "SCOUTING ZEALOT",
            allowed_actions=allowed_actions,
            max_count=1,
            allow_replacement_when_full=False,
        )
        return
