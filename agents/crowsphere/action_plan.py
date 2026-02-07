from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agents.crowsphere.errors import FatalActionPlanError, RetryableActionPlanError


@dataclass(frozen=True)
class ParseResult:
    plan: dict[str, Any]
    requires_tool_execution: bool = False


def _extract_top_level_json_objects(text: str) -> list[str]:
    """
    从文本中提取“顶层 JSON 对象”子串（形如 {...}）。

    仅做“提取/截取”，不做任何语义改写（例如不会替换引号、不会修补逗号）。
    目标是容忍模型输出的包裹文本，例如：
      - ```json ... ```
      - "Here is the JSON: {...}"
      - 多段输出（示例 + 最终 JSON）

    如果最后一个顶层对象没有闭合，也会把从 '{' 到结尾的片段作为候选返回，
    供后续的“补闭合括号”修复尝试。
    """
    candidates: list[str] = []
    start_idx: int | None = None
    depth = 0
    in_string = False
    escaped = False

    src = str(text or "")
    for i, ch in enumerate(src):
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start_idx = i
            depth += 1
            continue
        if ch == "}":
            if depth <= 0:
                continue
            depth -= 1
            if depth == 0 and start_idx is not None:
                candidates.append(src[start_idx : i + 1])
                start_idx = None
            continue

    if start_idx is not None:
        # Unclosed top-level object candidate.
        candidates.append(src[start_idx:].strip())

    return candidates


def _balance_unclosed_json(text: str) -> str:
    """为可能缺少闭合括号的 JSON 前缀补齐结尾的 '}' / ']'。

    该修复非常保守：只根据 opener 栈追加缺失的闭合符，不改写任何已有字符。
    """
    stack: list[str] = []
    in_str = False
    escape = False
    for ch in text:
        if in_str:
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == "\"":
                in_str = False
            continue

        if ch == "\"":
            in_str = True
            continue
        if ch in "{[":
            stack.append(ch)
            continue
        if ch == "}" and stack and stack[-1] == "{":
            stack.pop()
            continue
        if ch == "]" and stack and stack[-1] == "[":
            stack.pop()
            continue

    suffix: list[str] = []
    while stack:
        opener = stack.pop()
        suffix.append("}" if opener == "{" else "]")
    return text.rstrip() + "".join(suffix)


def _try_parse_action_plan_from_text(raw_text: str) -> dict[str, Any] | None:
    """
    尝试从 raw_text 中提取并解析一个 JSON 对象。
    选择策略：从后往前尝试（倾向把“最后一个 JSON 对象”当作最终答案，避免误取示例）。

    同时提供保守的“缺失闭合括号”修复：只会追加缺失的 '}' / ']'，不改写内容。
    """
    for chunk in reversed(_extract_top_level_json_objects(raw_text)):
        for cand in (chunk, _balance_unclosed_json(chunk)):
            try:
                data = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def parse_action_plan(raw_text: str) -> ParseResult:
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        repaired = _try_parse_action_plan_from_text(raw_text)
        if repaired is None:
            raise RetryableActionPlanError("json_parse_error", "模型输出不是合法 JSON") from exc
        data = repaired

    if not isinstance(data, dict):
        raise RetryableActionPlanError(
            "missing_required_field", "ActionPlan 根节点必须是对象"
        )

    plan_type = data.get("type")

    # 工具调用请求
    if plan_type == "tool_call":
        tool_name = data.get("tool_name")
        if not tool_name:
            raise RetryableActionPlanError(
                "missing_required_field", "tool_call 缺少 tool_name"
            )
        return ParseResult(plan=data, requires_tool_execution=True)

    # 兼容一些常见的“格式接近但字段名不对”的输出：只做结构补全，不改动作本身。
    # 例如：
    # - {"actions":[...]}（漏了 type/chunk_size）
    # - {"action_chunk":[...]}（把 type 写成 key）
    if "type" not in data:
        # 1) {"action_chunk": [...]}
        if "action_chunk" in data and isinstance(data.get("action_chunk"), list):
            payload = data.get("action_chunk")
            # 常见：payload 是 SC2 的 5 个动作字符串列表
            if payload and all(isinstance(x, str) for x in payload):
                return ParseResult(
                    plan={
                        "type": "action_chunk",
                        "chunk_size": 1,
                        "actions": [{"actions": payload}],
                    }
                )

        # 2) {"actions": [...]}
        if "actions" in data and isinstance(data.get("actions"), list):
            payload = data.get("actions")
            # SC2 常见：直接给出动作字符串列表（应当嵌在 actions[0].actions 里）
            if payload and all(isinstance(x, str) for x in payload):
                return ParseResult(
                    plan={
                        "type": "action_chunk",
                        "chunk_size": 1,
                        "actions": [{"actions": payload}],
                    }
                )
            # 其他情况（例如 [{"dir":"left"}]）无法在不猜测的情况下补全；要求模型重试。
            raise RetryableActionPlanError(
                "missing_required_field", "ActionPlan 缺少关键字段 type"
            )

        raise RetryableActionPlanError(
            "missing_required_field", "ActionPlan 缺少关键字段 type/actions"
        )

    if "actions" not in data:
        raise RetryableActionPlanError(
            "missing_required_field", "ActionPlan 缺少关键字段 type/actions"
        )

    return ParseResult(plan=data)


def normalize_action_plan(plan: dict[str, Any]) -> dict[str, Any]:
    plan_type = plan.get("type")
    if plan_type not in {"action_chunk", "skill_call"}:
        raise FatalActionPlanError(
            "unsupported_action_plan_type", f"不支持的 ActionPlan.type: {plan_type}"
        )

    if plan_type != "action_chunk":
        raise FatalActionPlanError(
            "unsupported_action_plan_type",
            "第一版在线只允许 type=action_chunk（可选 skill_call 需另行实现）",
        )

    actions = plan.get("actions")
    if not isinstance(actions, list):
        raise FatalActionPlanError("missing_required_field", "actions 必须是数组")
    if len(actions) == 0:
        raise FatalActionPlanError("empty_actions", "actions 不能为空")

    chunk_size = plan.get("chunk_size")
    normalized = dict(plan)
    normalized["chunk_size"] = len(actions) if not isinstance(chunk_size, int) else chunk_size
    if normalized["chunk_size"] != len(actions):
        normalized["chunk_size"] = len(actions)
    return normalized


def plan_to_action_strs(
    *,
    game: str,
    plan: dict[str, Any],
    config: dict[str, Any] | None = None,
    game_info: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    normalized = normalize_action_plan(plan)

    if game == "twenty_fourty_eight":
        from agents.crowsphere.adapters.twenty_fourty_eight import tfe_plan_to_action_strs

        return tfe_plan_to_action_strs(normalized)
    if game == "super_mario":
        from agents.crowsphere.adapters.super_mario import mario_plan_to_action_strs

        return mario_plan_to_action_strs(normalized, game_info or {})
    if game == "pokemon_red":
        from agents.crowsphere.adapters.pokemon_red import pokemon_plan_to_action_strs

        return pokemon_plan_to_action_strs(normalized, config or {})
    if game == "star_craft":
        from agents.crowsphere.adapters.star_craft import sc2_plan_to_action_strs

        return sc2_plan_to_action_strs(normalized, game_info or {}, config or {})

    raise FatalActionPlanError("unknown_game", f"未知游戏: {game}")
