"""2048 game-specific logic for CrowSphereEngine.

This module focuses on two things:
1) LLM makes the final move decision (up/down/left/right).
2) Tool-call compatibility: if the model requests a tool, only allow `calculator`.
"""

from __future__ import annotations

from typing import Any

from agents.crowsphere import action_plan
from agents.crowsphere.errors import FatalActionPlanError, RetryableActionPlanError
from agents.crowsphere.logging_jsonl import JsonlLogger
from agents.crowsphere.tool_definitions import ToolResult
from agents.crowsphere.vllm_client import OpenAICompatClient, VllmRequest


_DIRS = ("up", "down", "left", "right")


def pick_2048_dir_from_preproc(obs_text: str) -> str:
    """Pick a 2048 move deterministically from the preprocessed obs_text.

    The 2048 obs_text preprocessor already computes expectimax estimates per direction.
    This function parses those lines and chooses the best valid direction with the same
    tie-break rules used in preprocessing.
    """
    parsed = _parse_move_analysis(obs_text)
    candidates = [v for v in parsed.values() if bool(v.get("changed"))]
    if not candidates:
        # No valid moves; any direction is fine (the env will end soon).
        return "up"

    preferred_order = {"left": 0, "down": 1, "right": 2, "up": 3}

    def sort_key(item: dict[str, Any]) -> tuple[float, int, int, int]:
        d = str(item.get("direction") or "").lower()
        return (
            -float(item.get("expected_score_estimate", float("-inf"))),
            -int(item.get("merge_gain", 0)),
            -int(item.get("empty_after", 0)),
            int(preferred_order.get(d, 9)),
        )

    best = sorted(candidates, key=sort_key)[0]
    direction = str(best.get("direction") or "").lower()
    return direction if direction in _DIRS else "up"


def act_2048_with_tool(
    *,
    preproc: Any,
    game_info: dict[str, Any],
    params: dict[str, Any],
    image_data_url: str | None,
    image_meta: dict[str, Any],
    images_enabled: bool,
    prev_image_url: str | None,
    # Engine dependencies
    client: OpenAICompatClient,
    config: dict[str, Any],
    logger: JsonlLogger,
    step_id: int,
    action_queue: dict[str, list[str]],
    memory: dict[str, dict[str, Any]],
    build_prompt_fn: Any,
    build_generation_params_for_stage_fn: Any,
    build_messages_fn: Any,
    max_actions_per_pack_fn: Any,
    update_memory_fn: Any,
) -> str:
    """2048: model-driven decision (optional calculator tool call)."""
    game = "twenty_fourty_eight"
    obs_text = str(getattr(preproc, "text", "") or "")
    tool_results: list[ToolResult] = []
    last_error: str | None = None
    max_retries = 3
    max_attempts = 1 + max_retries

    # For 2048, the preprocessed text already contains full grid + move analysis.
    # Sending images to VLM is unnecessary and slows down inference.
    image_data_url = None
    prev_image_url = None

    # At most: one optional tool_call round + one final action round.
    stages = ["allow_tool_call", "action_only"]
    for stage in stages:
        retry_note = ""
        tool_executed_in_stage = False
        for attempt in range(1, max_attempts + 1):
            prompt = build_prompt_fn(
                game=game,
                obs_text=obs_text,
                game_info=game_info,
                tools_enabled=(stage == "allow_tool_call"),
                tool_results=tool_results if tool_results else None,
            )
            if retry_note:
                prompt = prompt + retry_note

            round_params = build_generation_params_for_stage_fn(
                game=game,
                game_info=game_info,
                stage=stage,
            )
            request = VllmRequest(
                messages=build_messages_fn(
                    prompt=prompt,
                    image_data_url=image_data_url,
                    prev_image_url=prev_image_url,
                    game=game,
                ),
                params=round_params,
            )

            try:
                raw = client.chat_completions(request)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                logger.write(
                    {
                        "event": "act",
                        "game": game,
                        "step_id": step_id,
                        "stage": stage,
                        "attempt": attempt,
                        "status": "request_error",
                        "error": last_error,
                    }
                )
                if attempt < max_attempts:
                    retry_note = (
                        "\n\n[重试]\n"
                        "上一次请求失败。请你重新输出：只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                    continue
                raise

            record: dict[str, Any] = {
                "event": "act",
                "game": game,
                "step_id": step_id,
                "stage": stage,
                "attempt": attempt,
                "model": {"name": config.get("model", {}).get("name")},
                "obs_text": obs_text,
                "obs_text_metadata": getattr(preproc, "metadata", {}),
                "prompt": prompt,
                "raw_model_output": raw,
                "tool_results": [{"name": tr.tool_name, "success": tr.success} for tr in tool_results],
                "image": image_meta,
                "status": "ok",
            }

            try:
                parsed_result = action_plan.parse_action_plan(raw)
                if parsed_result.requires_tool_execution:
                    if stage != "allow_tool_call":
                        last_error = "tool_call_not_allowed_in_action_only_stage"
                        record["status"] = "error"
                        record["error"] = last_error
                        logger.write(record)
                        if attempt < max_attempts:
                            retry_note = (
                                "\n\n[格式修复]\n"
                                "你刚才输出了 tool_call，但当前阶段禁止 tool_call。\n"
                                "请立刻重新输出：只输出 action_chunk（dir=up/down/left/right），只输出一个合法 JSON 对象。\n"
                            )
                            continue
                        raise RetryableActionPlanError("tool_call_not_allowed", str(last_error))

                    tool_result = execute_2048_tool(tool_call=parsed_result.plan, obs_text=obs_text)
                    tool_results = [tool_result]
                    record["status"] = "tool_executed"
                    record["tool_call"] = {
                        "tool_name": parsed_result.plan.get("tool_name"),
                        "arguments": parsed_result.plan.get("arguments"),
                    }
                    record["tool_result"] = {
                        "name": tool_result.tool_name,
                        "success": tool_result.success,
                        "error": tool_result.error,
                        "result": tool_result.result,
                    }
                    logger.write(record)
                    tool_executed_in_stage = True
                    break

                normalized, action_strs = action_plan.plan_to_action_strs(
                    game=game,
                    plan=parsed_result.plan,
                    config=config,
                    game_info=game_info,
                )

                # 2048 has stochastic spawns; queueing future moves is usually harmful.
                action_strs = action_strs[:1]
                final_action = action_strs[0]
                action_queue[game] = []
                update_memory_fn(game=game, obs_text=obs_text, action_str=final_action)

                record["normalized_action_plan"] = normalized
                record["final_action_str"] = final_action
                record["queue_len_after"] = 0
                log_2048_action_value(record, obs_text, final_action)
                logger.write(record)
                return final_action

            except (RetryableActionPlanError, FatalActionPlanError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                record["status"] = "error"
                record["error_kind"] = exc.kind
                record["error"] = last_error
                logger.write(record)
                if attempt < max_attempts:
                    retry_note = (
                        "\n\n[格式修复]\n"
                        "你刚才的输出无法解析或不符合要求。\n"
                        "请立刻重新输出：只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                    continue
                raise
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                record["status"] = "fatal_error"
                record["error"] = last_error
                logger.write(record)
                if attempt < max_attempts:
                    retry_note = (
                        "\n\n[格式修复]\n"
                        "你刚才的输出导致处理失败。请立刻重新输出：只输出一个合法 JSON 对象。\n"
                    )
                    continue
                raise

        if stage == "allow_tool_call" and tool_executed_in_stage:
            # Proceed to action_only stage with the calculator result injected into the prompt.
            continue

        # Stage exhausted without producing an action (or tool_call in allow_tool_call).
        raise RuntimeError(f"2048 推理失败（stage={stage}）：{last_error or 'unknown_error'}")

    raise RuntimeError("unreachable")


def log_2048_action_value(record: dict[str, Any], obs_text: str, final_action: str) -> None:
    """Log per-step action value (for debugging/analysis)."""
    try:
        parsed = _parse_move_analysis(obs_text)
        chosen = parsed.get(final_action) if parsed else None
        valid = [v for v in (parsed or {}).values() if v.get("changed")]
        best = max(valid, key=lambda x: float(x.get("expected_score_estimate", float("-inf")))) if valid else None

        if chosen and best and chosen.get("changed"):
            record["twenty_fourty_eight_action_value"] = {
                "chosen": {
                    "direction": final_action,
                    "changed": bool(chosen.get("changed")),
                    "immediate_score_gain": int(chosen.get("merge_gain", 0)),
                    "expected_score_estimate": float(chosen.get("expected_score_estimate")),
                },
                "best": {
                    "direction": str(best.get("direction")),
                    "expected_score_estimate": float(best.get("expected_score_estimate")),
                },
                "regret_estimate": float(best.get("expected_score_estimate")) - float(chosen.get("expected_score_estimate")),
            }
            return
    except Exception:
        # If parsing fails, fall back to recomputing (best-effort).
        pass

    try:
        import math

        from agents.crowsphere.twenty_fourty_eight_board import parse_board_from_obs_text
        from agents.crowsphere.twenty_fourty_eight_expected_score import (
            estimate_expected_score_for_direction,
            estimate_expected_scores_all_directions,
        )

        def safe_float(x: float) -> float | None:
            return float(x) if math.isfinite(x) else None

        board = parse_board_from_obs_text(obs_text)
        if board is not None and final_action in set(_DIRS):
            all_estimates = estimate_expected_scores_all_directions(board)
            best = max(
                (e for e in all_estimates if e.changed),
                key=lambda e: e.expected_score_estimate,
                default=None,
            )
            chosen = estimate_expected_score_for_direction(board, final_action)
            regret = (
                safe_float(best.expected_score_estimate - chosen.expected_score_estimate)
                if best is not None and chosen.changed
                else None
            )
            record["twenty_fourty_eight_action_value"] = {
                "chosen": {
                    "direction": chosen.direction,
                    "changed": chosen.changed,
                    "immediate_score_gain": chosen.immediate_score_gain,
                    "expected_score_estimate": safe_float(chosen.expected_score_estimate),
                },
                "best": {
                    "direction": best.direction,
                    "expected_score_estimate": safe_float(best.expected_score_estimate),
                }
                if best
                else None,
                "regret_estimate": regret,
            }
    except Exception as exc:
        record["twenty_fourty_eight_action_value_error"] = f"{type(exc).__name__}: {exc}"


def execute_2048_tool(
    *,
    tool_call: dict[str, Any],
    obs_text: str,
) -> ToolResult:
    """Execute 2048 tool calls.

    规则：只允许 calculator。
    """
    tool_name = str(tool_call.get("tool_name") or "")
    if tool_name != "calculator":
        return ToolResult(
            tool_name=tool_name,
            success=False,
            result=None,
            error=f"2048 只允许 calculator，收到: {tool_name or 'N/A'}",
        )

    args = tool_call.get("arguments")
    if not isinstance(args, dict):
        args = {}
    expr = str(args.get("expression") or args.get("expr") or "").strip()
    if not expr:
        return ToolResult(
            tool_name="calculator",
            success=False,
            result=None,
            error="calculator.arguments.expression 不能为空",
        )

    try:
        from agents.crowsphere.engine_2048_calculator import safe_calculator_eval

        value = safe_calculator_eval(expr)
        return ToolResult(tool_name="calculator", success=True, result=value)
    except Exception as exc:
        return ToolResult(
            tool_name="calculator",
            success=False,
            result=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _parse_move_analysis(obs_text: str) -> dict[str, dict[str, Any]]:
    """Parse Move Analysis lines emitted by obs_text_preproc_2048."""
    import re

    move_re = re.compile(
        r"^\s*-\s*(up|down|left|right)\s*:\s*change=(yes|no)\s*,\s*merge_gain=(\-?\d+)\s*,\s*"
        r"empty_after=(\-?\d+)\s*,\s*max_after=(\-?\d+)\s*,\s*expected_score=([\-0-9.]+|inf|\-inf)\s*$",
        re.IGNORECASE,
    )
    parsed: dict[str, dict[str, Any]] = {}
    for ln in (obs_text or "").splitlines():
        m = move_re.match(ln.strip())
        if not m:
            continue
        d = m.group(1).lower()
        changed = m.group(2).lower() == "yes"
        merge_gain = int(m.group(3))
        empty_after = int(m.group(4))
        max_after = int(m.group(5))
        es_raw = m.group(6).lower()
        if es_raw == "inf":
            expected_score = float("inf")
        elif es_raw == "-inf":
            expected_score = float("-inf")
        else:
            expected_score = float(es_raw)
        parsed[d] = {
            "direction": d,
            "changed": changed,
            "merge_gain": merge_gain,
            "empty_after": empty_after,
            "max_after": max_after,
            "expected_score_estimate": expected_score,
        }
    return parsed
