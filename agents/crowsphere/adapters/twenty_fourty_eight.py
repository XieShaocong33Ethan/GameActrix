from __future__ import annotations

from typing import Any

from agents.crowsphere.errors import FatalActionPlanError


_ALLOWED_DIRS = {"up", "down", "left", "right"}


def tfe_plan_to_action_strs(plan: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    actions = plan["actions"]
    normalized_actions: list[dict[str, Any]] = []
    action_strs: list[str] = []

    for a in actions:
        if not isinstance(a, dict) or "dir" not in a:
            raise FatalActionPlanError("invalid_dir_for_2048", "2048 动作缺少 dir")
        d = str(a["dir"]).lower()
        if d not in _ALLOWED_DIRS:
            raise FatalActionPlanError("invalid_dir_for_2048", f"2048 dir 非法: {d}")
        normalized_actions.append({"dir": d})
        action_strs.append(d)

    normalized_plan = dict(plan)
    normalized_plan["actions"] = normalized_actions
    normalized_plan["chunk_size"] = len(normalized_actions)
    return normalized_plan, action_strs
