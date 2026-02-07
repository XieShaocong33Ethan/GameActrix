from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .pokemon_tools_parsing import _non_na_lines, _split_sections


@dataclass(frozen=True)
class DialogStrategy:
    """Dialog strategy output for pokemon_dialog_strategy tool."""

    dialog_type: str  # "tv_snes", "npc_dialog", "event_dialog", "normal"
    recommended_keys: list[str]
    reason: str
    dialog_context: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dialog_type": self.dialog_type,
            "recommended_keys": self.recommended_keys,
            "reason": self.reason,
            "dialog_context": self.dialog_context,
        }


def pokemon_dialog_strategy(
    *,
    obs_text: str,
    memory: dict[str, Any] | None,
    config: dict[str, Any],
) -> DialogStrategy:
    """
    Analyze dialog content and recommend best key presses.

    This tool is specialized for Dialog state. It can identify special dialogs
    (TV/SNES, special NPCs, events) and return correct key presses.

    - For TV/SNES dialogs: press 'b' to exit
    - For normal dialogs: press 'a' repeatedly to advance
    - Extensible for special event handling
    """
    # Get config parameters
    pokemon_cfg = config.get("pokemon", {}) if isinstance(config.get("pokemon"), dict) else {}
    tool_cfg = pokemon_cfg.get("tools", {}) if isinstance(pokemon_cfg.get("tools"), dict) else {}
    dialog_advance_repeats = int(tool_cfg.get("dialog_advance_repeats", 4))
    dialog_advance_repeats = max(1, min(8, dialog_advance_repeats))

    # Parse sections from obs_text
    sections = _split_sections(obs_text)
    filtered_screen = "\n".join(_non_na_lines(sections.get("Filtered Screen Text", "")))
    selection_box = sections.get("Selection Box Text", "")

    # Detect TV/SNES scenario
    # Pattern: "playing the SNES", "SNES", etc. in dialog text
    if re.search(r"\bSNES\b|playing the", filtered_screen, re.IGNORECASE):
        return DialogStrategy(
            dialog_type="tv_snes",
            recommended_keys=["b"],
            reason="Dialog contains 'SNES'/'playing the' → watching TV, press B to exit",
            dialog_context={"text": filtered_screen[:100]},
        )

    # Detect name selection dialog - choose default name instead of "NEW NAME"
    # Pattern: "what is your name" or "what is his name" + "NEW NAME" in selection
    is_name_dialog = re.search(r"what is\s+(your|his)\s+name", filtered_screen, re.IGNORECASE | re.DOTALL)
    has_new_name = "NEW NAME" in selection_box
    if is_name_dialog and has_new_name:
        return DialogStrategy(
            dialog_type="name_selection",
            recommended_keys=["down", "a"],
            reason="Name selection dialog → press DOWN then A to choose default name (saves steps vs NEW NAME)",
            dialog_context={"text": filtered_screen[:100], "selection": selection_box[:100]},
        )

    # Default: advance dialog normally
    return DialogStrategy(
        dialog_type="normal",
        recommended_keys=["a"] * dialog_advance_repeats,
        reason="Normal dialog → press A repeatedly to advance",
        dialog_context={"text": filtered_screen[:100]},
    )


def dialog_strategy_to_json(ds: DialogStrategy) -> str:
    """Convert DialogStrategy to JSON string for tool result."""
    return json.dumps(ds.to_dict(), ensure_ascii=False, indent=2, sort_keys=False)

