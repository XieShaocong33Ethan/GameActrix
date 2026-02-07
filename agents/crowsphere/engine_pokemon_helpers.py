"""Internal helpers for Pokemon engine logic.

This module is intentionally small and dependency-light, so the larger engine
implementation can be split into multiple files while keeping a stable public
API via agents.crowsphere.engine_pokemon.
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.crowsphere import action_plan
from agents.crowsphere.tool_definitions import ToolResult


def _pokemon_parse_state(obs_text: str) -> str:
    m = re.search(r"^State:\s*([A-Za-z_]+)", obs_text or "", flags=re.MULTILINE)
    return (m.group(1) if m else "").strip()


def _pokemon_extract_section(obs_text: str, section_name: str) -> str:
    marker = f"[{section_name}]"
    idx = (obs_text or "").find(marker)
    if idx < 0:
        return ""
    start = (obs_text or "").find("\n", idx)
    if start < 0:
        return ""
    end = (obs_text or "").find("\n[", start + 1)
    if end < 0:
        end = len(obs_text or "")
    return (obs_text or "")[start + 1 : end].strip()


def _auto_action_from_decision_support_tool_result(
    *,
    tool_result: ToolResult,
    config: dict[str, Any],
    game_info: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]] | None:
    if not tool_result.success or not tool_result.result:
        return None
    try:
        ds = json.loads(str(tool_result.result))
    except Exception:
        return None
    candidates = ds.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        return None
    candidates_sorted = sorted(candidates, key=lambda c: int(c.get("priority", 0)), reverse=True)
    top = candidates_sorted[0]
    if not isinstance(top, dict):
        return None

    action: dict[str, Any]
    if "env_tool" in top and isinstance(top.get("env_tool"), dict):
        action = {"env_tool": top["env_tool"]}
    elif "keys" in top and isinstance(top.get("keys"), list):
        action = {"keys": top["keys"]}
    else:
        return None

    plan = {"type": "action_chunk", "chunk_size": 1, "actions": [action]}
    normalized, action_strs = action_plan.plan_to_action_strs(
        game="pokemon_red",
        plan=plan,
        config=config,
        game_info=game_info,
    )
    if not action_strs:
        return None
    return action_strs[0], normalized, top

