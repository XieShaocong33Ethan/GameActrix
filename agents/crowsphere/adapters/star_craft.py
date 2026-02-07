from __future__ import annotations

import ast
import re
from typing import Any

from agents.crowsphere.errors import FatalActionPlanError, RetryableActionPlanError


def _parse_action_dict(value: Any) -> dict[str, Any] | None:
    """
    官方评测可能用 protobuf 的 map<string,string> 传递 game_info，导致 action_dict 变成字符串。
    这里允许从 str/dict 两种形态解析。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _normalize_action_name(name: Any, *, allowed: set[str]) -> str | None:
    """
    允许对动作名做轻量“格式归一化”，但不做任何策略性替换/兜底：
    - 大小写
    - 空格/下划线/连字符的等价写法
    """
    token = str(name).upper().strip()
    if token in allowed:
        return token

    candidates = [token]
    if "_" in token:
        candidates.append(token.replace("_", " "))
        # Some env actions use hyphens (e.g. MULTI-ATTACK). Treat "_" and "-" as equivalent.
        candidates.append(token.replace("_", "-"))
    if " " in token:
        candidates.append(token.replace(" ", "_"))
        candidates.append(token.replace(" ", "-"))
    if "-" in token:
        candidates.append(token.replace("-", " "))
        candidates.append(token.replace("-", "_"))

    expanded: list[str] = []
    for c in candidates:
        expanded.append(re.sub(r"\s+", " ", c).strip())
        expanded.append(re.sub(r"_+", "_", c).strip())

    seen: set[str] = set()
    for c in expanded:
        if c in seen:
            continue
        seen.add(c)
        if c in allowed:
            return c
    return None


def _extract_actions_list(plan: dict[str, Any]) -> list[Any] | None:
    """
    兼容多种历史输出格式：
    - 新格式：plan["actions"] == ["BUILD PYLON", "TRAIN ZEALOT", ...]
    - 旧格式：plan["actions"] == [{"actions": ["BUILD PYLON", ...]}]
    - 更旧格式：plan["actions"] == [["BUILD PYLON", ...]]
    """
    def _split_actions_text(text: str) -> list[str]:
        # Some models may emit a single string like:
        # "BUILD PYLON;BUILD GATEWAY;TRAIN ZEALOT;..."
        # This is a format issue (not a policy decision). We split deterministically.
        parts = re.split(r"[;\n,]+", str(text))
        return [p.strip() for p in parts if p.strip()]

    actions = plan.get("actions")
    if not isinstance(actions, list):
        if isinstance(actions, str):
            return _split_actions_text(actions)
        return None
    if actions and isinstance(actions[0], dict) and "actions" in actions[0]:
        nested = actions[0].get("actions")
        if isinstance(nested, list):
            return nested
        if isinstance(nested, str):
            return _split_actions_text(nested)
        return None
    if actions and isinstance(actions[0], list):
        return actions[0]
    return actions


def sc2_plan_to_action_strs(
    plan: dict[str, Any],
    game_info: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    action_dict = _parse_action_dict(game_info.get("action_dict"))
    if not action_dict:
        raise FatalActionPlanError("sc2_missing_action_dict", "SC2 缺少 action_dict")

    num_actions = int(game_info.get("num_actions", 0))
    if num_actions <= 0:
        raise FatalActionPlanError("sc2_missing_num_actions", "SC2 缺少 num_actions")

    allowed = {str(k).upper() for k in action_dict.keys()}

    raw_actions = _extract_actions_list(plan)
    if raw_actions is None:
        raise RetryableActionPlanError("sc2_invalid_actions_type", "SC2 actions 必须是数组")

    if len(raw_actions) != num_actions:
        raise RetryableActionPlanError(
            "sc2_actions_length_mismatch",
            f"SC2 actions 长度必须等于 num_actions={num_actions}（当前={len(raw_actions)}）",
        )

    normalized_actions: list[str] = []
    invalid: list[str] = []
    for x in raw_actions:
        normalized = _normalize_action_name(x, allowed=allowed)
        if normalized is None:
            invalid.append(str(x))
        else:
            normalized_actions.append(normalized)

    if invalid:
        raise RetryableActionPlanError(
            "sc2_action_not_allowed",
            "存在不在 action_dict 的动作名: " + ", ".join(invalid),
        )

    # 构建返回结果：每一步输出 1 条字符串，其中包含 num_actions 行编号动作。
    normalized_plan = dict(plan)
    normalized_plan["actions"] = [{"actions": normalized_actions}]
    normalized_plan["chunk_size"] = 1

    lines = [f"{i + 1}: {name}" for i, name in enumerate(normalized_actions)]
    action_strs = ["\n".join(lines)]
    return normalized_plan, action_strs
