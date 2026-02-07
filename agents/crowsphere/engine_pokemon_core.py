"""Pokemon engine core utilities (queue policy, config).

This file exists to keep each source file under 500 lines while preserving the
public API exposed by agents.crowsphere.engine_pokemon.
"""

from __future__ import annotations

from typing import Any


def pokemon_tool_use_enabled(config: dict[str, Any]) -> bool:
    """Check if Pokemon tool use is enabled."""
    pokemon_cfg = config.get("pokemon") if isinstance(config.get("pokemon"), dict) else {}
    tool_use = pokemon_cfg.get("tool_use") if isinstance(pokemon_cfg.get("tool_use"), dict) else {}
    enabled = tool_use.get("enabled", None)
    # Default to disabled unless explicitly enabled.
    return False if enabled is None else bool(enabled)


def pokemon_should_break_queue(
    *,
    obs_text: str,
    config: dict[str, Any],
    memory: dict[str, dict[str, Any]],
) -> bool:
    """Determine if action queue should break.

    Auto break queued action pack when:
    - we are not in Field state anymore (dialog/selection/title/battle needs immediate reaction)
    - or the last outcome indicates stuck/oscillation
    """
    try:
        from agents.crowsphere.pokemon_tools import pokemon_action_pack_policy

        policy = pokemon_action_pack_policy(obs_text=obs_text, config=config)
        state = str(policy.get("state") or "UNKNOWN").strip()
        if state.lower() != "field":
            return True
    except Exception:
        # If parsing fails, be conservative and break.
        return True

    mem = memory.get("pokemon_red") or {}
    note = mem.get("note")
    if isinstance(note, str) and note:
        if "oscillating" in note or "unchanged" in note:
            return True
    return False


def pokemon_tool_use_max_tool_rounds(config: dict[str, Any]) -> int:
    """Get max tool rounds for Pokemon."""
    pokemon_cfg = config.get("pokemon") if isinstance(config.get("pokemon"), dict) else {}
    tool_use = pokemon_cfg.get("tool_use") if isinstance(pokemon_cfg.get("tool_use"), dict) else {}
    return max(0, min(2, int(tool_use.get("max_tool_rounds", 1))))


def max_actions_per_pack_pokemon(
    *,
    obs_text: str,
    config: dict[str, Any],
) -> int:
    """Get max actions per pack for Pokemon.

    State-dependent from config.pokemon.action_pack (deterministic).
    """
    try:
        from agents.crowsphere.pokemon_tools import pokemon_action_pack_policy

        policy = pokemon_action_pack_policy(obs_text=obs_text, config=config)
        return max(1, min(8, int(policy.get("max_actions", 1))))
    except Exception:
        return 1
