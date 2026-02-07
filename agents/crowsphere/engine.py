"""CrowSphereEngine：统一编排四个游戏的提示词、推理、解析与日志。"""
from __future__ import annotations

import base64
import os
import re
import threading
from io import BytesIO
from pathlib import Path
from typing import Any

try:  # pragma: no cover
    from openai import BadRequestError
except Exception:  # pragma: no cover
    BadRequestError = None  # type: ignore[assignment]

from agents.crowsphere.config_loader import load_effective_config
from agents.crowsphere.eval_artifacts_autowrite import register_eval_artifacts_autowrite
from agents.crowsphere.llm_logging import reset_call_context, set_call_context
from agents.crowsphere.logging_jsonl import JsonlLogger
from agents.crowsphere.obs_text_preproc import preprocess_obs_text_for_model
from agents.crowsphere.prompt_builder import build_prompt
from agents.crowsphere.vllm_client import OpenAICompatClient
from agents.crowsphere.tool_definitions import ToolResult

from agents.crowsphere.engine_sc2 import maybe_reset_sc2_episode, update_sc2_context
from agents.crowsphere.engine_2048 import (
    act_2048_with_tool,
    log_2048_action_value,
    pick_2048_dir_from_preproc,
)
from agents.crowsphere.engine_pokemon import (
    act_pokemon_with_tool,
    pokemon_tool_use_enabled,
    pokemon_should_break_queue,
    max_actions_per_pack_pokemon,
)
from agents.crowsphere.engine_memory import (
    update_memory_after_action as _update_memory_general,
    build_memory_hint,
)
from agents.crowsphere.engine_generation import (
    build_generation_params,
    build_generation_params_for_stage,
)
from agents.crowsphere.engine_generic_action import act_generic_action
from agents.crowsphere.engine_helpers import (
    build_messages as _build_messages,
    is_data_inspection_failed as _is_data_inspection_failed,
    jpeg_bytes_to_data_url as _jpeg_bytes_to_data_url,
    resolve_api_key as _resolve_api_key,
)

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]

_GLOBAL_MODEL_LOCK = threading.Lock()


class CrowSphereEngine:
    def __init__(self) -> None:
        self._config = load_effective_config()
        self._local_lock = threading.Lock()
        self._step_id = 0
        self._memory: dict[str, dict[str, Any]] = {}
        self._action_queue: dict[str, list[str]] = {}
        register_eval_artifacts_autowrite()
        self._logger = self._build_logger()
        self._client = self._build_client()
        self._prev_image_url: dict[str, str | None] = {}

    def _parse_mario_world_x(self, game_info: dict[str, Any]) -> int | None:
        """从 game_info.mario_loc_history 提取 world x（用于 episode reset 检测）。"""
        hist = game_info.get("mario_loc_history")
        if not isinstance(hist, str) or not hist:
            return None
        # 兼容形如 "(2451, np.uint8(107))" 的日志格式：只抓取 x（第一个整数）。
        xs = re.findall(r"\((\d+),", hist)
        if not xs:
            return None
        try:
            return int(xs[-1])
        except Exception:
            return None

    def _get_or_create_mario_preproc_memory(self, game_info: dict[str, Any]) -> Any:
        """为 super_mario 预处理器维护跨 step 记忆；world x 大幅回退视为 reset。"""
        from agents.crowsphere.mario_preprocess import MarioPreprocessMemory

        mem = self._memory.setdefault("super_mario", {})
        memory_obj = mem.get("preproc_memory")
        if not isinstance(memory_obj, MarioPreprocessMemory):
            memory_obj = MarioPreprocessMemory()
            mem["preproc_memory"] = memory_obj
            mem["_last_world_x"] = None

        cur_x = self._parse_mario_world_x(game_info)
        last_x = mem.get("_last_world_x")
        if isinstance(last_x, int) and isinstance(cur_x, int):
            # 经验阈值：当 world x 大幅回退，基本可判定为新一局 reset。
            if cur_x < last_x - 200:
                memory_obj.reset()
        mem["_last_world_x"] = cur_x
        return memory_obj

    def _inject_mario_env_info(self, obs_text: str, game_info: dict[str, Any]) -> str:
        """给马里奥规则层补充最小环境信息（优先 x_pos/y_pos/time），避免误用屏幕坐标。"""
        text = str(obs_text or "")
        if "[Env Info]" in text:
            return text
        if not isinstance(game_info, dict):
            return text
        info = game_info.get("info") if isinstance(game_info.get("info"), dict) else {}

        def _to_int(v: Any) -> int | None:
            try:
                return int(v)
            except Exception:
                return None

        x_pos = _to_int(info.get("x_pos"))
        y_pos = _to_int(info.get("y_pos"))
        time_left = _to_int(info.get("time"))
        # Official starter-kit server does not expose `info`; fall back to mario_loc_history.
        if x_pos is None:
            x_pos = self._parse_mario_world_x(game_info)

        lines: list[str] = ["[Env Info]"]
        if x_pos is not None:
            lines.append(f"x_pos: {x_pos}")
        if y_pos is not None:
            lines.append(f"y_pos: {y_pos}")
        if time_left is not None:
            lines.append(f"time: {time_left}")

        if len(lines) <= 1:
            return text
        suffix = "\n" + "\n".join(lines) + "\n"
        return (text.rstrip() + suffix).strip()

    def act(self, game: str, obs: dict[str, Any]) -> str:
        lock = _GLOBAL_MODEL_LOCK if self._use_global_model_lock() else self._local_lock
        with lock:
            self._step_id += 1
            obs_text = str(obs.get("obs_str") or obs.get("obs_text") or "")
            game_info = obs.get("game_info") if isinstance(obs.get("game_info"), dict) else {}
            if game == "star_craft":
                # REMOTE runner may omit episode/map metadata from game_info; detect episode resets
                # from the observation stream (game_time) to avoid cross-episode contamination.
                maybe_reset_sc2_episode(memory=self._memory, game_info=game_info, obs_text=obs_text)
            # Extract image early so we can run optional Mario VLM sentinel before text preprocessing.
            raw_image = self._extract_image(obs) if self._images_enabled() else None
            image_data_url, image_meta = self._process_image(obs)

            ctx_tokens = set_call_context(
                game=game,
                step_id=self._step_id,
                image_sha256=image_meta.get("image_sha256") if isinstance(image_meta, dict) else None,
            )
            try:
                # Deterministic pit/lava detector (image-based, no network).
                # This is especially useful for RandomStages where template matching can miss castle lava pits.
                if game == "super_mario" and raw_image is not None:
                    try:
                        from agents.crowsphere.mario_pit_detector import analyze_mario_pit_from_image

                        pit_result = analyze_mario_pit_from_image(image=raw_image, obs_text=obs_text)
                        if pit_result.pit_ahead:
                            obs_text = obs_text + pit_result.to_obs_block()
                    except Exception:
                        # Keep deterministic behavior: if anything fails, do not modify obs_text.
                        pass

                # Optional: Mario VLM sentinel (external VLM, deterministic) to improve recall and tighten candidates.
                if game == "super_mario" and raw_image is not None and self._mario_vlm_sentinel_enabled():
                    try:
                        from agents.crowsphere.mario_vlm_sentinel import analyze as _sentinel_analyze

                        sentinel_result = _sentinel_analyze(raw_image, enabled=True)
                        # Only inject when the sentinel provides positive evidence.
                        # Rationale: a "no evidence" block would flip `vlm_sentinel.present=True` and
                        # suppress deterministic fallbacks, which can hurt robustness when the sentinel
                        # is unavailable/misconfigured or produces false negatives.
                        has_signal = bool(
                            getattr(sentinel_result, "on_platform", False)
                            or getattr(sentinel_result, "enemy_below_ahead", False)
                            or getattr(sentinel_result, "enemy_ahead", False)
                            or getattr(sentinel_result, "dense_enemies_ahead", False)
                            or getattr(sentinel_result, "pipe_ahead", False)
                            or getattr(sentinel_result, "pit_ahead", False)
                            or getattr(sentinel_result, "stairs_ahead", False)
                            or getattr(sentinel_result, "stairs_pit_ahead", False)
                            or getattr(sentinel_result, "low_ceiling", False)
                        )
                        if (getattr(sentinel_result, "error", None) is None) and has_signal:
                            obs_text = obs_text + sentinel_result.to_obs_block()
                    except Exception:
                        # Keep deterministic behavior: if anything fails, do not modify obs_text.
                        pass

                mario_preproc_memory = None
                if game == "super_mario":
                    mario_preproc_memory = self._get_or_create_mario_preproc_memory(game_info)
                    obs_text = self._inject_mario_env_info(obs_text, game_info)
                preproc = preprocess_obs_text_for_model(
                    game=game, obs_text=obs_text, koopa_memory=mario_preproc_memory, config=self._config
                )
                prompt_game_info = self._build_prompt_game_info(game, game_info)
                prev_image_url = self._prev_image_url.get(game)

                # Check action queue
                queued_result = self._check_action_queue(game, preproc.text, image_data_url)
                if queued_result is not None:
                    return queued_result

                # Game-specific dispatch
                if game == "twenty_fourty_eight":
                    return self._dispatch_2048(preproc, game_info, image_data_url, image_meta, prev_image_url)

                if game == "pokemon_red" and pokemon_tool_use_enabled(self._config):
                    return self._dispatch_pokemon(preproc, game_info, image_data_url, image_meta, prev_image_url)

                if game == "super_mario":
                    from agents.crowsphere.engine_mario_act import mario_use_model

                    if not mario_use_model(self._config):
                        action = self._dispatch_mario(preproc, game_info, image_meta)
                        if image_data_url:
                            self._prev_image_url[game] = image_data_url
                        return action

                # Default: generic model-based action
                return self._act_generic(
                    game=game,
                    preproc=preproc,
                    game_info=game_info,
                    prompt_game_info=prompt_game_info,
                    image_data_url=image_data_url,
                    image_meta=image_meta,
                    prev_image_url=prev_image_url,
                )
            finally:
                reset_call_context(ctx_tokens)

    def _process_image(self, obs: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """Extract and preprocess image from observation."""
        image_data_url = None
        image_meta: dict[str, Any] = {"image_present": False}
        if not self._images_enabled():
            return image_data_url, image_meta

        image = self._extract_image(obs)
        if image is None:
            return image_data_url, image_meta

        from agents.crowsphere.image_preproc import preprocess_image
        short_side = int(self._config.get("image_preproc", {}).get("short_side", 184))
        jpeg_quality = int(self._config.get("image_preproc", {}).get("jpeg_quality", 85))
        preproc_img = preprocess_image(image, short_side=short_side, jpeg_quality=jpeg_quality)
        image_data_url = _jpeg_bytes_to_data_url(preproc_img.jpeg_bytes)
        image_meta = {
            "image_present": True,
            "image_sha256": preproc_img.image_sha256,
            "image_size": preproc_img.image_size,
            "image_preproc": preproc_img.image_preproc,
            "jpeg_bytes_len": len(preproc_img.jpeg_bytes),
        }
        return image_data_url, image_meta

    def _build_prompt_game_info(self, game: str, game_info: dict[str, Any]) -> dict[str, Any]:
        """Build prompt game info with SC2 context if needed."""
        prompt_game_info = dict(game_info) if game_info else {}
        if game == "star_craft":
            prompt_game_info["sc2_context"] = self._memory.get("star_craft", {})
        return prompt_game_info

    def _check_action_queue(self, game: str, obs_text: str, image_data_url: str | None) -> str | None:
        """Check if we have queued actions, return next action or None."""
        queued = self._action_queue.get(game) or []
        if not queued:
            return None

        if game == "pokemon_red" and pokemon_should_break_queue(
            obs_text=obs_text, config=self._config, memory=self._memory
        ):
            self._action_queue[game] = []
            return None

        next_action = queued.pop(0)
        self._action_queue[game] = queued
        self._update_memory_after_action(game=game, obs_text=obs_text, action_str=next_action)
        self._logger.write({
            "event": "act", "game": game, "step_id": self._step_id,
            "status": "queue_hit", "final_action_str": next_action, "queue_len_after": len(queued),
        })
        if image_data_url:
            self._prev_image_url[game] = image_data_url
        return next_action

    def _dispatch_2048(self, preproc, game_info, image_data_url, image_meta, prev_image_url) -> str:
        """2048：默认走确定性 expectimax（无需模型），可选工具模式。"""
        game = "twenty_fourty_eight"
        cfg_2048 = self._config.get(game) if isinstance(self._config.get(game), dict) else {}
        use_model = bool(cfg_2048.get("use_model", True))

        if not use_model:
            final_action = pick_2048_dir_from_preproc(preproc.text)
            self._action_queue[game] = []
            self._update_memory_after_action(game=game, obs_text=preproc.text, action_str=final_action)

            record: dict[str, Any] = {
                "event": "act",
                "game": game,
                "step_id": self._step_id,
                "status": "deterministic",
                "policy": "expectimax_from_preproc",
                "final_action_str": final_action,
                "obs_text": preproc.text,
                "obs_text_metadata": getattr(preproc, "metadata", {}),
                "image": image_meta,
            }
            log_2048_action_value(record, preproc.text, final_action)
            self._logger.write(record)
            return final_action

        params = build_generation_params(config=self._config, game=game, game_info=game_info)
        return act_2048_with_tool(
            preproc=preproc,
            game_info=game_info,
            params=params,
            image_data_url=image_data_url,
            image_meta=image_meta,
            images_enabled=self._images_enabled(),
            prev_image_url=prev_image_url,
            client=self._client,
            config=self._config,
            logger=self._logger,
            step_id=self._step_id,
            action_queue=self._action_queue,
            memory=self._memory,
            build_prompt_fn=self._build_prompt,
            build_generation_params_for_stage_fn=self._build_generation_params_for_stage,
            build_messages_fn=_build_messages,
            max_actions_per_pack_fn=self._max_actions_per_pack,
            update_memory_fn=self._update_memory_after_action,
        )

    def _dispatch_pokemon(self, preproc, game_info, image_data_url, image_meta, prev_image_url) -> str:
        """Dispatch to Pokemon tool calling mode."""
        result = act_pokemon_with_tool(
            preproc=preproc, game_info=game_info,
            image_data_url=image_data_url, image_meta=image_meta,
            images_enabled=self._images_enabled(), prev_image_url=prev_image_url,
            client=self._client, config=self._config, logger=self._logger, step_id=self._step_id,
            action_queue=self._action_queue, memory=self._memory,
            build_prompt_fn=self._build_prompt, build_generation_params_for_stage_fn=self._build_generation_params_for_stage,
            build_messages_fn=_build_messages, max_actions_per_pack_fn=self._max_actions_per_pack,
            update_memory_fn=self._update_memory_after_action,
        )
        if image_data_url:
            self._prev_image_url["pokemon_red"] = image_data_url
        return result

    def _dispatch_mario(self, preproc, game_info, image_meta) -> str:
        """Super Mario: deterministic policy (candidates + risk report), optionally bypassing the model."""
        from agents.crowsphere.engine_mario_act import act_mario_deterministic

        return act_mario_deterministic(
            preproc=preproc,
            game_info=game_info,
            image_meta=image_meta,
            config=self._config,
            logger=self._logger,
            step_id=self._step_id,
            action_queue=self._action_queue,
            update_memory_fn=self._update_memory_after_action,
        )

    def _act_generic(
        self,
        *,
        game: str,
        preproc: Any,
        game_info: dict[str, Any],
        prompt_game_info: dict[str, Any],
        image_data_url: str | None,
        image_meta: dict[str, Any],
        prev_image_url: str | None,
    ) -> str:
        """通用动作路径：构造提示词 -> 模型推理 -> 解析 ActionPlan -> 下发动作。"""
        action = act_generic_action(
            game=game,
            step_id=self._step_id,
            obs_text=preproc.text,
            game_info=game_info,
            prompt_game_info=prompt_game_info,
            image_data_url=image_data_url,
            image_meta=image_meta,
            prev_image_url=prev_image_url,
            client=self._client,
            config=self._config,
            logger=self._logger,
            memory=self._memory,
            action_queue=self._action_queue,
            build_prompt_fn=self._build_prompt,
            build_generation_params_fn=build_generation_params,
            build_messages_fn=_build_messages,
            max_actions_per_pack_fn=self._max_actions_per_pack,
            update_memory_fn=self._update_memory_after_action,
            is_data_inspection_failed_fn=_is_data_inspection_failed,
            bad_request_error_type=BadRequestError,
        )
        if image_data_url:
            self._prev_image_url[game] = image_data_url
        return action

    def _resolve_output_dir(self) -> Path:
        """日志优先落到 $GAME_DATA_DIR；未设置则使用 logging.output_dir（默认 logs）。"""
        game_data_dir = (os.getenv("GAME_DATA_DIR") or "").strip()
        if game_data_dir:
            return Path(game_data_dir).expanduser().resolve()
        output_dir = str(self._config.get("logging", {}).get("output_dir", "logs"))
        return Path(output_dir).expanduser().resolve()

    def _build_logger(self) -> JsonlLogger:
        return JsonlLogger(path=self._resolve_output_dir() / "crowsphere_engine.jsonl")

    def _build_client(self) -> OpenAICompatClient:
        model_cfg = self._config.get("model", {}) if isinstance(self._config.get("model"), dict) else {}
        base_url = str(model_cfg.get("base_url") or "").strip()
        model_name = str(model_cfg.get("name") or "").strip()
        if not base_url or not model_name:
            raise ValueError("配置缺少 model.base_url 或 model.name")
        output_dir = self._resolve_output_dir()
        llm_call_logger = JsonlLogger(path=output_dir / "llm_calls.jsonl")
        tokenizer_name = str(self._config.get("logging", {}).get("tokenizer_name") or "cl100k_base")
        return OpenAICompatClient(
            base_url=base_url,
            model=model_name,
            api_key=_resolve_api_key(base_url=base_url),
            call_logger=llm_call_logger,
            tokenizer_name=tokenizer_name,
        )

    def _use_global_model_lock(self) -> bool:
        concurrency = self._config.get("concurrency") if isinstance(self._config.get("concurrency"), dict) else {}
        return bool(concurrency.get("vllm_global_lock", True))

    def _images_enabled(self) -> bool:
        image_cfg = self._config.get("image") if isinstance(self._config.get("image"), dict) else {}
        return bool(image_cfg.get("enabled", True))

    def _mario_vlm_sentinel_enabled(self) -> bool:
        """Enable Mario VLM sentinel via config or env (defaults to off)."""
        env = (os.getenv("CROWSPHERE_MARIO_VLM_SENTINEL") or "").strip().lower()
        if env in {"1", "true", "yes", "y", "on"}:
            return True
        if env in {"0", "false", "no", "n", "off"}:
            return False
        mario_cfg = self._config.get("super_mario") if isinstance(self._config.get("super_mario"), dict) else {}
        sentinel_cfg = mario_cfg.get("vlm_sentinel") if isinstance(mario_cfg.get("vlm_sentinel"), dict) else {}
        return bool(sentinel_cfg.get("enabled", False))

    def _build_prompt(self, *, game: str, obs_text: str, game_info: dict[str, Any],
                      tools_enabled: bool = False, tool_results: list[ToolResult] | None = None) -> str:
        memory_hint = build_memory_hint(memory=self._memory, game=game, obs_text=obs_text)
        return build_prompt(game=game, obs_text=obs_text, game_info=game_info,
                           memory_hint=memory_hint, tools_enabled=tools_enabled, tool_results=tool_results)

    def _update_memory_after_action(self, *, game: str, obs_text: str, action_str: str) -> None:
        if game == "star_craft":
            update_sc2_context(memory=self._memory, obs_text=obs_text, action_str=action_str)
        else:
            _update_memory_general(memory=self._memory, game=game, obs_text=obs_text, action_str=action_str)

    def _extract_image(self, obs: dict[str, Any]):
        if Image is None:
            return None
        image = obs.get("obs_image")
        if image is not None:
            return image
        obs_image_str = str(obs.get("obs_image_str") or "").strip()
        if not obs_image_str:
            return None
        try:
            return Image.open(BytesIO(base64.b64decode(obs_image_str)))
        except Exception:
            return None

    def _max_actions_per_pack(self, *, game: str, obs_text: str) -> int:
        if game == "pokemon_red":
            return max_actions_per_pack_pokemon(obs_text=obs_text, config=self._config)
        # Mario 场景变化快，跨步动作易过期；为稳定性与复现性固定为 1。
        if game == "super_mario":
            return 1
        k = int(self._config.get("action_chunk", {}).get("k", {}).get(game, 1))
        return max(1, min(16, k))

    def _build_generation_params_for_stage(self, *, game: str, game_info: dict[str, Any], stage: str) -> dict[str, Any]:
        return build_generation_params_for_stage(config=self._config, game=game, game_info=game_info, stage=stage)
