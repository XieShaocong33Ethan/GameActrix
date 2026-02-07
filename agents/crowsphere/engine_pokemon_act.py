"""Pokemon action loop (decision support + optional VLM).

Refactor (docs/POKEMON_AGENT_REFACTOR_PLAN.md):
- Environment output only: low-level keys OR official `use_tool(...)` macros.
- Remove custom `tool_call`: engine runs decision support; model only chooses keys/env_tool.
"""

from __future__ import annotations

import json
from typing import Any

from agents.crowsphere import action_plan
from agents.crowsphere.engine_memory import update_pokemon_subtask_from_tool_result
from agents.crowsphere.errors import FatalActionPlanError, RetryableActionPlanError
from agents.crowsphere.logging_jsonl import JsonlLogger
from agents.crowsphere.tool_definitions import ToolResult
from agents.crowsphere.vllm_client import OpenAICompatClient, VllmRequest


def _decision_support_tool(
    *,
    obs_text: str,
    memory: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> tuple[ToolResult, dict[str, Any] | None]:
    """Run deterministic decision support and update engine memory (subtask/progress)."""
    try:
        from agents.crowsphere.pokemon_tools import (
            decision_support_to_pretty_json,
            pokemon_decision_support,
        )

        ds = pokemon_decision_support(
            obs_text=obs_text,
            memory=memory.get("pokemon_red"),
            config=config,
            proposed_subtask=None,
        )
        ds_dict = ds.to_dict()
        update_pokemon_subtask_from_tool_result(memory=memory, tool_result_dict=ds_dict)
        return (
            ToolResult(tool_name="pokemon_decision_support", success=True, result=decision_support_to_pretty_json(ds)),
            ds_dict,
        )
    except Exception as exc:
        return (
            ToolResult(
                tool_name="pokemon_decision_support",
                success=False,
                result=None,
                error=f"{type(exc).__name__}: {exc}",
            ),
            None,
        )


def _candidate_sig_from_action(action: dict[str, Any]) -> tuple[str, str]:
    """Return a stable signature for comparing env_tool candidates."""
    env_tool = action.get("env_tool")
    if not isinstance(env_tool, dict):
        return ("", "")
    name = str(env_tool.get("name") or "").strip()
    args = env_tool.get("args")
    try:
        args_str = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        args_str = ""
    return (name, args_str)


def _action_matches_any_candidate(
    *,
    normalized_first_action: dict[str, Any],
    candidates: list[Any],
) -> bool:
    """Check whether the normalized first action matches a decision support candidate."""
    if not isinstance(normalized_first_action, dict):
        return False
    if not isinstance(candidates, list) or not candidates:
        return False

    if "env_tool" in normalized_first_action:
        sig = _candidate_sig_from_action(normalized_first_action)
        if sig == ("", ""):
            return False
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            if "env_tool" not in cand:
                continue
            if sig == _candidate_sig_from_action(cand):
                return True
        return False

    if "keys" in normalized_first_action:
        keys = normalized_first_action.get("keys")
        if not isinstance(keys, list):
            return False
        keys_norm = [str(k).strip().lower() for k in keys if isinstance(k, str) and str(k).strip()]
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            if "keys" not in cand or "env_tool" in cand:
                continue
            cand_keys = cand.get("keys")
            if not isinstance(cand_keys, list):
                continue
            cand_norm = [str(k).strip().lower() for k in cand_keys if isinstance(k, str) and str(k).strip()]
            if keys_norm == cand_norm:
                return True
        return False

    return False


def act_pokemon_with_tool(
    *,
    preproc: Any,
    game_info: dict[str, Any],
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
    """Pokemon Red agent step.

    - Always runs deterministic decision support to produce candidates/progress (as prompt guidance).
    - By default, the model must choose the final action (keys/env_tool). When candidates are available,
      we constrain the output to be one of the candidates (via retry on mismatch).
    - Optional: you can enable deterministic bypass to pick the highest-priority candidate, but it is
      disabled by default for compliance.
    - If no candidates are available, we still do a single model call (with retries on invalid output).
    - If model output is invalid, retries up to 3 times (max 4 calls), then raises.
    """
    game = "pokemon_red"
    max_retries = 3
    max_attempts = 1 + max_retries

    # Always compute decision support (candidates + progress). This also updates memory with
    # subtask/progress so milestone tracking remains monotonic across all states.
    decision_support_tr, decision_support_dict = _decision_support_tool(
        obs_text=preproc.text,
        memory=memory,
        config=config,
    )

    def _best_candidate_action() -> dict[str, Any] | None:
        """Pick the highest-priority action from decision support candidates (deterministic)."""
        if not isinstance(decision_support_dict, dict):
            return None
        candidates = decision_support_dict.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            return None

        best_key: tuple[int, int, int, str] | None = None
        best_action: dict[str, Any] | None = None
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            try:
                priority = int(cand.get("priority", 0) or 0)
            except Exception:
                priority = 0

            env_tool = cand.get("env_tool")
            if isinstance(env_tool, dict) and isinstance(env_tool.get("name"), str):
                action: dict[str, Any] = {"env_tool": env_tool}
                is_env_tool = 1
                key_len = 0
            else:
                keys = cand.get("keys")
                if not isinstance(keys, list):
                    keys = []
                action = {"keys": keys}
                is_env_tool = 0
                key_len = len(keys)

            try:
                action_sig = json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            except Exception:
                action_sig = ""

            # Sort: higher priority, prefer env_tool, prefer longer key sequences (more progress per step),
            # then stable by action JSON.
            sort_key = (priority, is_env_tool, key_len, action_sig)
            if best_key is None or sort_key > best_key:
                best_key = sort_key
                best_action = action

        return best_action

    pokemon_cfg = config.get("pokemon") if isinstance(config.get("pokemon"), dict) else {}
    tool_use_cfg = pokemon_cfg.get("tool_use") if isinstance(pokemon_cfg.get("tool_use"), dict) else {}
    bypass_model_when_candidates = bool(tool_use_cfg.get("bypass_model_when_candidates", False))

    if bypass_model_when_candidates:
        best_action = _best_candidate_action()
        if best_action is not None:
            plan = {"type": "action_chunk", "chunk_size": 1, "actions": [best_action]}
            normalized_plan, action_strs = action_plan.plan_to_action_strs(
                game=game,
                plan=plan,
                config=config,
                game_info=game_info,
            )
            action_strs = action_strs[:1]
            final_action = action_strs[0]
            action_queue[game] = []
            update_memory_fn(game=game, obs_text=preproc.text, action_str=final_action)

            logger.write(
                {
                    "event": "act",
                    "game": game,
                    "step_id": step_id,
                    "attempt": 0,
                    "model": {"name": None, "note": "bypass_model_when_candidates"},
                    "obs_text_metadata": getattr(preproc, "metadata", None),
                    "obs_text": preproc.text,
                    "game_info": game_info,
                    "image": image_meta,
                    "prompt": None,
                    "tool_results": [
                        {"tool_name": decision_support_tr.tool_name, "success": decision_support_tr.success, "error": decision_support_tr.error}
                    ],
                    "raw_model_output": None,
                    "status": "ok",
                    "normalized_action_plan": normalized_plan,
                    "final_action_str": final_action,
                    "queue_len_after": 0,
                }
            )
            return final_action

    # Model path: provide decision support result to the model (when available),
    # but do not allow tool_call protocol outputs.
    tool_results: list[ToolResult] | None = [decision_support_tr] if decision_support_tr.success else None
    tools_enabled = tool_results is not None
    params = build_generation_params_for_stage_fn(game=game, game_info=game_info, stage="action_only")

    def _request(prompt_text: str) -> str:
        request = VllmRequest(
            messages=build_messages_fn(
                prompt=prompt_text,
                image_data_url=image_data_url,
                prev_image_url=prev_image_url,
                game=game,
            ),
            params=params,
        )
        return client.chat_completions(request)

    def _allowed_action_json_lines() -> str:
        """Render candidates as fully-formed JSON outputs the model can copy verbatim."""
        candidates = decision_support_dict.get("candidates") if isinstance(decision_support_dict, dict) else None
        if not isinstance(candidates, list) or not candidates:
            return ""

        lines: list[str] = []
        for cand in candidates[:12]:
            if not isinstance(cand, dict):
                continue
            if "env_tool" in cand and isinstance(cand.get("env_tool"), dict):
                action = {"env_tool": cand["env_tool"]}
            else:
                keys = cand.get("keys")
                action = {"keys": keys if isinstance(keys, list) else []}
            plan = {"type": "action_chunk", "chunk_size": 1, "actions": [action]}
            lines.append(json.dumps(plan, ensure_ascii=False, separators=(",", ":")))

        if not lines:
            return ""
        return "\n".join(lines)


    retry_note = ""
    last_error: str | None = None
    for attempt in range(1, max_attempts + 1):
        prompt = build_prompt_fn(
            game=game,
            obs_text=preproc.text,
            game_info=game_info,
            tools_enabled=tools_enabled,
            tool_results=tool_results,
        )
        if retry_note:
            prompt = prompt + retry_note

        raw: str
        try:
            raw = _request(prompt)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            logger.write(
                {
                    "event": "act",
                    "game": game,
                    "step_id": step_id,
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
            "attempt": attempt,
            "model": {"name": config.get("model", {}).get("name")},
            "obs_text_metadata": getattr(preproc, "metadata", None),
            "obs_text": preproc.text,
            "game_info": game_info,
            "image": image_meta,
            "prompt": prompt,
            "tool_results": [
                {"tool_name": tr.tool_name, "success": tr.success, "error": tr.error}
                for tr in (tool_results or [])
            ],
            "raw_model_output": raw,
        }

        try:
            parsed = action_plan.parse_action_plan(raw)
            if parsed.requires_tool_execution:
                raise RetryableActionPlanError("tool_call_not_allowed", "不允许输出 tool_call（已移除 tool_call 协议）")

            normalized_plan, action_strs = action_plan.plan_to_action_strs(
                game=game,
                plan=parsed.plan,
                config=config,
                game_info=game_info,
            )

            # With decision support, require exactly one chosen action to avoid stale queues.
            action_strs = action_strs[:1]
            final_action = action_strs[0]

            # If decision support candidates exist, enforce "choose from candidates" by retrying.
            candidates = decision_support_dict.get("candidates") if isinstance(decision_support_dict, dict) else None
            normalized_actions = normalized_plan.get("actions") if isinstance(normalized_plan, dict) else None
            first_action = normalized_actions[0] if isinstance(normalized_actions, list) and normalized_actions else None
            if (
                isinstance(candidates, list)
                and first_action is not None
                and not _action_matches_any_candidate(normalized_first_action=first_action, candidates=candidates)
            ):
                raise RetryableActionPlanError("candidate_mismatch", "模型输出不在 candidates 中")

            action_queue[game] = []
            update_memory_fn(game=game, obs_text=preproc.text, action_str=final_action)
            record["status"] = "ok"
            record["normalized_action_plan"] = normalized_plan
            record["final_action_str"] = final_action
            record["queue_len_after"] = 0
            logger.write(record)
            return final_action

        except (RetryableActionPlanError, FatalActionPlanError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            record["status"] = "error"
            record["error_kind"] = exc.kind
            record["error"] = last_error
            logger.write(record)
            if attempt < max_attempts:
                if exc.kind == "candidate_mismatch":
                    allowed = _allowed_action_json_lines()
                    retry_note = (
                        "\n\n[候选约束]\n"
                        "你刚才的动作不在 candidates 中。\n"
                        "你必须从下面“允许输出的 JSON”中任选一行，原样拷贝输出（一字不改）。\n"
                        "不要输出其它任何内容。\n"
                    )
                    if allowed:
                        retry_note += "\n\n[允许输出的 JSON]\n" + allowed + "\n"
                    retry_note += "只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                elif exc.kind == "tool_call_not_allowed":
                    retry_note = (
                        "\n\n[协议修复]\n"
                        "禁止输出 tool_call。\n"
                        "请输出 action_chunk（keys 或 env_tool），只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                elif exc.kind == "json_parse_error":
                    retry_note = (
                        "\n\n[格式修复]\n"
                        "你刚才的输出不是合法 JSON。\n"
                        "请立刻重新输出：只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                else:
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

    raise RuntimeError(f"Pokemon 推理失败：{last_error or 'unknown_error'}")
