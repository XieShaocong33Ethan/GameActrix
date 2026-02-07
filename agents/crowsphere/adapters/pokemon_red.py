from __future__ import annotations

from typing import Any

from agents.crowsphere.errors import FatalActionPlanError


_ALLOWED_KEYS = {
    "up",
    "down",
    "left",
    "right",
    "a",
    "b",
    "start",
    "select",
    "none",
    "quit",
}

_ALLOWED_ENV_TOOLS = {
    # Movement / navigation
    "move_to",
    "warp_with_warp_point",
    "overworld_map_transition",
    # Interaction / dialog
    "interact_with_object",
    "continue_dialog",
    # Battle helpers
    "select_move_in_battle",
    "switch_pkmn_in_battle",
    "run_away",
    "use_item_in_battle",
}


def pokemon_plan_to_action_strs(
    plan: dict[str, Any], config: dict[str, Any]
) -> tuple[dict[str, Any], list[str]]:
    max_keys = int(config.get("pokemon", {}).get("max_keys_per_step", 4))

    actions = plan["actions"]
    normalized_actions: list[dict[str, Any]] = []
    action_strs: list[str] = []

    def _env_tool_to_action_str(env_tool: dict[str, Any]) -> str:
        if not isinstance(env_tool, dict):
            raise FatalActionPlanError("pokemon_env_tool_invalid", "env_tool 必须是对象")
        name = env_tool.get("name")
        args = env_tool.get("args", {})
        if not isinstance(name, str) or not name.strip():
            raise FatalActionPlanError("pokemon_env_tool_invalid", "env_tool.name 不能为空")
        name = name.strip()
        if name not in _ALLOWED_ENV_TOOLS:
            raise FatalActionPlanError("pokemon_env_tool_invalid", f"不支持的 env_tool: {name}")
        if args is None:
            args = {}
        if not isinstance(args, dict):
            raise FatalActionPlanError("pokemon_env_tool_invalid", "env_tool.args 必须是对象")

        # The official env parses strings like:
        #   use_tool(move_to, (x_dest=1, y_dest=2))
        # It eval()s dict(<kwargs_str>), so we must output python-like kwargs.
        if not args:
            kwargs_str = ""
        else:
            parts: list[str] = []
            for k, v in args.items():
                if not isinstance(k, str) or not k:
                    continue
                # Use repr for values so strings are quoted safely.
                parts.append(f"{k}={repr(v)}")
            kwargs_str = ", ".join(parts)

        return f"use_tool({name}, ({kwargs_str}))"

    for a in actions:
        if not isinstance(a, dict):
            raise FatalActionPlanError("pokemon_action_invalid", "Pokémon 动作必须是对象")

        # Option A: env_tool (single macro action executed inside the environment)
        if "env_tool" in a:
            env_tool = a.get("env_tool")
            action_str = _env_tool_to_action_str(env_tool)
            normalized_actions.append({"env_tool": env_tool})
            action_strs.append(action_str)
            continue

        # Option B: keys (low-level key pack)
        if "keys" not in a or not isinstance(a["keys"], list):
            raise FatalActionPlanError("pokemon_keys_empty_after_filter", "Pokémon 动作缺少 keys 数组")
        raw_list = a["keys"]
        if len(raw_list) == 0:
            normalized_actions.append({"keys": []})
            action_strs.append("pass")
            continue

        raw_keys = [str(k).lower() for k in raw_list]
        filtered = [k for k in raw_keys if k in _ALLOWED_KEYS]
        if not filtered:
            raise FatalActionPlanError(
                "pokemon_keys_empty_after_filter", "过滤非法 key 后为空"
            )

        meaningful = [k for k in filtered if k != "none"]
        if not meaningful:
            normalized_actions.append({"keys": []})
            action_strs.append("pass")
            continue

        truncated = meaningful[:max_keys]
        normalized_actions.append({"keys": truncated})
        action_strs.append(" ".join(truncated))

    normalized_plan = dict(plan)
    normalized_plan["actions"] = normalized_actions
    normalized_plan["chunk_size"] = len(normalized_actions)
    return normalized_plan, action_strs
