from __future__ import annotations

import time
from typing import Any, Callable

from agents.crowsphere import action_plan
from agents.crowsphere.errors import FatalActionPlanError, RetryableActionPlanError
from agents.crowsphere.vllm_client import VllmRequest


def act_generic_action(
    *,
    game: str,
    step_id: int,
    obs_text: str,
    game_info: dict[str, Any],
    prompt_game_info: dict[str, Any],
    image_data_url: str | None,
    image_meta: dict[str, Any],
    prev_image_url: str | None,
    client: Any,
    config: dict[str, Any],
    logger: Any,
    memory: dict[str, dict[str, Any]],
    action_queue: dict[str, list[str]],
    build_prompt_fn: Callable[..., str],
    build_generation_params_fn: Callable[..., dict[str, Any]],
    build_messages_fn: Callable[..., list[dict[str, Any]]],
    max_actions_per_pack_fn: Callable[..., int],
    update_memory_fn: Callable[..., None],
    is_data_inspection_failed_fn: Callable[[Exception], bool],
    bad_request_error_type: type | None,
) -> str:
    params = build_generation_params_fn(config=config, game=game, game_info=game_info)
    base_prompt = build_prompt_fn(
        game=game,
        obs_text=obs_text,
        game_info=prompt_game_info,
        tools_enabled=False,
    )

    max_retries = 3
    max_attempts = 1 + max_retries
    last_error: str | None = None
    force_text_only = False
    retry_note = ""

    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt + (retry_note or "")
        use_image = (image_data_url is not None) and (not force_text_only)
        request = VllmRequest(
            messages=build_messages_fn(
                prompt=prompt,
                image_data_url=image_data_url if use_image else None,
                prev_image_url=prev_image_url if use_image else None,
                game=game,
            ),
            params=params,
        )

        try:
            t0 = time.time()
            raw = client.chat_completions(request)
            latency_s = time.time() - t0
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
            if (
                bad_request_error_type
                and isinstance(exc, bad_request_error_type)
                and use_image
                and is_data_inspection_failed_fn(exc)
                and attempt < max_attempts
            ):
                force_text_only = True
                continue
            raise

        record: dict[str, Any] = {
            "event": "act",
            "game": game,
            "step_id": step_id,
            "attempt": attempt,
            "model": {"name": (config.get("model") or {}).get("name")},
            "obs_text": obs_text,
            "game_info": game_info,
            "image": image_meta,
            "raw_model_output": raw,
            "last_error": last_error,
            "request": {"used_image": bool(use_image), "latency_s": float(latency_s)},
        }

        try:
            parsed = action_plan.parse_action_plan(raw).plan
            enriched_game_info = dict(game_info) if game_info else {}
            enriched_game_info["obs_str"] = obs_text
            if game == "star_craft":
                enriched_game_info["sc2_context"] = memory.get("star_craft", {})
            normalized, action_strs = action_plan.plan_to_action_strs(
                game=game,
                plan=parsed,
                config=config,
                game_info=enriched_game_info,
            )
            max_pack = max_actions_per_pack_fn(game=game, obs_text=obs_text)
            action_strs = action_strs[:max_pack]
            final_action = action_strs[0]
            action_queue[game] = list(action_strs[1:])
            update_memory_fn(game=game, obs_text=obs_text, action_str=final_action)
            record.update(
                {
                    "status": "ok",
                    "normalized_action_plan": normalized,
                    "final_action_str": final_action,
                }
            )
            if game == "super_mario" and isinstance(normalized, dict):
                keys = (
                    "vlm_skill",
                    "vlm_jump_level",
                    "final_skill",
                    "final_jump_level",
                    "override_reason",
                    "candidate_table",
                    "risk_table",
                )
                record.update({k: normalized.get(k) for k in keys})
            logger.write(record)
            return final_action
        except RetryableActionPlanError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            record.update({"status": "error", "error": last_error, "error_kind": exc.kind})
            logger.write(record)
            if attempt < max_attempts:
                if exc.kind == "json_parse_error":
                    retry_note = (
                        "\n\n[格式修复]\n"
                        "你刚才的输出不是合法 JSON。\n"
                        "请立刻重新输出：只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                else:
                    reason = str(exc).strip()
                    detail = f"原因：{reason}\n" if reason else ""
                    retry_note = (
                        "\n\n[评估反馈]\n"
                        f"{detail}"
                        "请根据 obs_text 的约束修正并重新输出。\n"
                        "只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                    )
                continue
            raise
        except FatalActionPlanError as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            record.update({"status": "error", "error": last_error, "error_kind": exc.kind})
            logger.write(record)
            if attempt < max_attempts:
                retry_note = (
                    "\n\n[格式修复]\n"
                    "你刚才的输出不符合 ActionPlan 约束。\n"
                    "请立刻重新输出：只输出一个合法 JSON 对象（不要 markdown，不要解释，不要额外文本）。\n"
                )
                continue
            raise
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            record.update({"status": "fatal_error", "error": last_error})
            logger.write(record)
            raise

    raise RuntimeError("unreachable")

