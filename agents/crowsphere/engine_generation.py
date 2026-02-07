"""Generation parameters and guided JSON schema building for CrowSphereEngine.

This module handles:
- build_generation_params: Build generation parameters for model calls
- build_guided_json_schema: Build JSON schema for guided decoding
- build_generation_params_for_stage: Stage-specific generation params (B1 tool calling)
"""
from __future__ import annotations

import os
import ast
from typing import Any


def build_generation_params(
    *,
    config: dict[str, Any],
    game: str,
    game_info: dict[str, Any],
) -> dict[str, Any]:
    """Build generation parameters for model calls.
    
    Args:
        config: Engine configuration.
        game: The game name.
        game_info: Game info dictionary.
        
    Returns:
        Dictionary of generation parameters.
    """
    gen = config.get("generation") if isinstance(config.get("generation"), dict) else {}
    max_tokens = int(gen.get("max_tokens", 256))
    temperature = float(gen.get("temperature", 0))
    # Reproducibility: allow forcing temperature=0 via env var (default off).
    if str(os.getenv("CROWSPHERE_FORCE_TEMP0", "0")).strip().lower() in {"1", "true", "yes"}:
        temperature = 0.0
    # Clamp: temperature cannot be negative.
    temperature = max(0.0, float(temperature))
    top_p = float(gen.get("top_p", 1.0))
    top_p = min(1.0, max(0.0, top_p))

    params: dict[str, Any] = {
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        # Keep penalties at 0 for determinism and to avoid provider-specific drift.
        "presence_penalty": 0,
        "frequency_penalty": 0,
    }

    base_url = str(config.get("model", {}).get("base_url", "")).lower()
    is_dashscope = "dashscope.aliyuncs.com" in base_url
    is_local_vllm = ("127.0.0.1" in base_url) or ("localhost" in base_url)
    force_response_format = ("modelscope" in base_url) or is_dashscope
    guided_json = bool(gen.get("guided_json", False)) and not force_response_format
    if guided_json:
        schema = build_guided_json_schema(config=config, game=game, game_info=game_info)

        # vLLM 的 OpenAI 兼容服务在新版本中使用 structured_outputs.* 传递结构化输出约束。
        # 旧字段 guided_json/guided_decoding_backend 在部分版本里会触发 400。
        params["extra_body"] = {"structured_outputs": {"json": schema}}
    else:
        # For local vLLM, `response_format={"type":"json_object"}` may route through
        # structured-outputs backends (e.g. xgrammar) and can crash the engine core
        # after many requests. In non-guided mode we therefore *omit* response_format
        # for local vLLM and rely on prompt + parser robustness instead.
        if not is_local_vllm:
            params["response_format"] = {"type": "json_object"}

    # DashScope qwen3-vl 系列容易进入 repetition loop（_summary_summary_...），
    # 强制用较短的 max_tokens 来防止生成超长无效输出。
    # 注意：不使用 stop 参数，因为它可能在 JSON 闭合前就触发截断。
    if is_dashscope:
        # 限制 max_tokens 避免 repetition loop 生成超长无效输出
        # 对于 Mario 的简单 action_chunk JSON，256 tokens 足够
        params["max_tokens"] = min(params.get("max_tokens", 256), 256)

    return params


def build_guided_json_schema(
    *,
    config: dict[str, Any],
    game: str,
    game_info: dict[str, Any],
) -> dict[str, Any]:
    """Build JSON schema for guided decoding.
    
    Args:
        config: Engine configuration.
        game: The game name.
        game_info: Game info dictionary.
        
    Returns:
        JSON schema dictionary.
    """
    k = int(config.get("action_chunk", {}).get("k", {}).get(game, 1))
    k = max(1, min(16, k))
    # Super Mario macro-actions go stale quickly; the engine enforces 1 action per call.
    if game == "super_mario":
        k = 1

    if game == "twenty_fourty_eight":
        action_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"dir": {"type": "string", "enum": ["up", "down", "left", "right"]}},
            "required": ["dir"],
        }
    elif game == "super_mario":
        action_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "jump_level": {"type": "integer", "minimum": 0, "maximum": 6},
                # 可选：用于日志/校验的语义标签（不影响环境动作）。
                "skill": {
                    "type": "string",
                    "enum": [
                        "SAFE",
                        "FAST",
                        "STOMP",
                        "DENSE_SAFE",
                        "STAIRS_JUMP",
                        "SLOW_WALK",
                    ],
                },
                "why": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["jump_level"],
        }
    elif game == "pokemon_red":
        # Define env_tool action format (for macro actions like warp, dialog, battle)
        env_tool_action = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "env_tool": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "enum": [
                                "move_to",
                                "warp_with_warp_point",
                                "overworld_map_transition",
                                "interact_with_object",
                                "continue_dialog",
                                "select_move_in_battle",
                                "switch_pkmn_in_battle",
                                "run_away",
                                "use_item_in_battle",
                            ],
                        },
                        "args": {"type": "object"}  # Flexible args object
                    },
                    "required": ["name", "args"]
                }
            },
            "required": ["env_tool"]
        }

        # Define keys action format (for raw button presses)
        keys_action = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "keys": {
                    "type": "array",
                    "minItems": 0,
                    "maxItems": 8,
                    "items": {
                        "type": "string",
                        "enum": [
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
                        ],
                    },
                }
            },
            "required": ["keys"],
        }

        # Use oneOf to allow either env_tool OR keys (mutually exclusive)
        action_item = {
            "oneOf": [env_tool_action, keys_action]
        }
    elif game == "star_craft":
        num_actions = int(game_info.get("num_actions", 0))
        num_actions = max(1, min(64, num_actions))
        # Prefer constraining SC2 action strings to the env-provided action_dict keys.
        # This prevents common typos (e.g. BUILD_PYLLON) from slipping through and
        # causing avoidable retries/failures.
        allowed_actions: list[str] | None = None
        raw_action_dict = game_info.get("action_dict")
        parsed_action_dict: dict[str, Any] | None = None
        if isinstance(raw_action_dict, dict):
            parsed_action_dict = raw_action_dict
        elif isinstance(raw_action_dict, str):
            text = raw_action_dict.strip()
            if text:
                try:
                    parsed = ast.literal_eval(text)
                except (ValueError, SyntaxError):
                    parsed = None
                if isinstance(parsed, dict):
                    parsed_action_dict = parsed
        if parsed_action_dict:
            allowed_actions = [str(k).upper() for k in parsed_action_dict.keys()]
            # Keep stable order for reproducibility.
            allowed_actions = sorted(set(allowed_actions))
            # Encourage non-empty macro actions. The SC2 env accepts "EMPTY ACTION",
            # but letting the model pick it freely often leads to stalling and losses.
            allowed_actions = [a for a in allowed_actions if a != "EMPTY ACTION"] or allowed_actions

        action_item = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "actions": {
                    "type": "array",
                    "minItems": num_actions,
                    "maxItems": num_actions,
                    "items": (
                        {"type": "string", "enum": allowed_actions}
                        if allowed_actions
                        else {"type": "string"}
                    ),
                }
            },
            "required": ["actions"],
        }
    else:
        raise ValueError(f"未知游戏: {game}")

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "type": {"type": "string", "enum": ["action_chunk"]},
            "chunk_size": {"type": "integer", "minimum": 1, "maximum": k},
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": k,
                "items": action_item,
            },
        },
        "required": ["type", "chunk_size", "actions"],
    }


def build_generation_params_for_stage(
    *,
    config: dict[str, Any],
    game: str,
    game_info: dict[str, Any],
    stage: str,
) -> dict[str, Any]:
    """B1 guided_json handling for staged generation.
    
    - stage=allow_tool_call: only require JSON object (no guided_json schema that would block tool_call)
    - stage=action_only: use the normal action_chunk constraints (guided_json if enabled & supported)
    
    Args:
        config: Engine configuration.
        game: The game name.
        game_info: Game info dictionary.
        stage: Either "allow_tool_call" or "action_only".
        
    Returns:
        Dictionary of generation parameters.
    """
    if stage == "allow_tool_call":
        # Reuse the normal generation params (so provider-specific safety clamps still apply),
        # but disable guided_json so the model can output tool_call JSON.
        params = build_generation_params(config=config, game=game, game_info=game_info)
        params.pop("extra_body", None)  # drop guided decoding constraints
        params["response_format"] = {"type": "json_object"}
        return params

    return build_generation_params(config=config, game=game, game_info=game_info)
