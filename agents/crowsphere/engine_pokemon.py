"""Pokemon game-specific entrypoints for CrowSphereEngine.

Public API is kept stable here, while the implementation is split into:
- engine_pokemon_act.py: VLM-driven tool-calling loop
- engine_pokemon_core.py: queue policy + config helpers
"""

from __future__ import annotations

from agents.crowsphere.engine_pokemon_act import act_pokemon_with_tool
from agents.crowsphere.engine_pokemon_core import (
    max_actions_per_pack_pokemon,
    pokemon_should_break_queue,
    pokemon_tool_use_enabled,
    pokemon_tool_use_max_tool_rounds,
)

__all__ = [
    "act_pokemon_with_tool",
    "max_actions_per_pack_pokemon",
    "pokemon_should_break_queue",
    "pokemon_tool_use_enabled",
    "pokemon_tool_use_max_tool_rounds",
]
