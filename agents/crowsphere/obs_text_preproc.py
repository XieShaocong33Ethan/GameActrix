from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ObsTextPreprocResult:
    text: str
    metadata: dict[str, Any]


def preprocess_obs_text_for_model(
    *,
    game: str,
    obs_text: str,
    max_bytes: int = 24000,
    koopa_memory: Any | None = None,
    config: dict[str, Any] | None = None,
) -> ObsTextPreprocResult:
    return preprocess_obs_str_for_model(
        game=game, obs_str=obs_text, max_bytes=max_bytes, koopa_memory=koopa_memory, config=config
    )


def preprocess_obs_str_for_model(
    *,
    game: str,
    obs_str: str,
    max_bytes: int = 24000,
    koopa_memory: Any | None = None,
    config: dict[str, Any] | None = None,
) -> ObsTextPreprocResult:
    if max_bytes <= 0:
        raise ValueError("max_bytes 必须是正整数")

    normalized = _normalize_text(obs_str)
    if game == "pokemon_red":
        from agents.crowsphere.obs_text_preproc_pokemon import (
            preprocess_pokemon_red_obs_text,
        )

        rendered = preprocess_pokemon_red_obs_text(normalized)
        strategy = "pokemon_red.v1"
    elif game == "twenty_fourty_eight":
        from agents.crowsphere.obs_text_preproc_2048 import (
            preprocess_twenty_fourty_eight_obs_text,
        )

        rendered = preprocess_twenty_fourty_eight_obs_text(normalized)
        strategy = "twenty_fourty_eight.v1"
    elif game == "super_mario":
        from agents.crowsphere.obs_text_preproc_mario import (
            preprocess_super_mario_obs_text,
        )

        mode_env = (os.getenv("CROWSPHERE_MARIO_PREPROC_MODE") or "").strip().lower()
        mode_cfg = ""
        if isinstance(config, dict):
            mario_cfg = config.get("super_mario") if isinstance(config.get("super_mario"), dict) else {}
            preproc_cfg = mario_cfg.get("preproc") if isinstance(mario_cfg.get("preproc"), dict) else {}
            mode_cfg = str(preproc_cfg.get("mode") or "").strip().lower()
        mode = mode_env or mode_cfg or "evidence_only"
        if mode not in {"full", "evidence_only", "raw"}:
            mode = "evidence_only"

        if mode == "raw":
            rendered = normalized.strip() or "N/A"
        else:
            # 传递 koopa_memory 用于处理 Koopa flicker（官方检测可能漏检）
            rendered = preprocess_super_mario_obs_text(normalized, koopa_memory=koopa_memory, mode=mode)
        # 传递 koopa_memory 用于处理 Koopa flicker（官方检测可能漏检）
        strategy = f"super_mario.v1.{mode}"
    else:
        rendered = normalized.strip() or "N/A"
        strategy = "generic.v1"

    original_bytes = len(normalized.encode("utf-8"))
    rendered_bytes = len(rendered.encode("utf-8"))
    truncated = rendered_bytes > max_bytes
    final_text = _truncate_middle_utf8(rendered, max_bytes=max_bytes) if truncated else rendered

    metadata: dict[str, Any] = {
        "game": game,
        "strategy": strategy,
        "max_bytes": max_bytes,
        "original_bytes": original_bytes,
        "rendered_bytes": rendered_bytes,
        "final_bytes": len(final_text.encode("utf-8")),
        "truncated": truncated,
    }
    return ObsTextPreprocResult(text=final_text, metadata=metadata)


def _normalize_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.rstrip() for ln in text.split("\n")]

    collapsed: list[str] = []
    blank_run = 0
    for ln in lines:
        if ln.strip() == "":
            blank_run += 1
            if blank_run <= 1:
                collapsed.append("")
            continue
        blank_run = 0
        collapsed.append(ln)
    return "\n".join(collapsed).strip()


def _truncate_middle_utf8(
    text: str,
    *,
    max_bytes: int,
    head_bytes: int | None = None,
    tail_bytes: int | None = None,
    cut_marker: str = "\n...[TRUNCATED]...\n",
) -> str:
    raw = text.encode("utf-8")
    if len(raw) <= max_bytes:
        return text

    marker_bytes = cut_marker.encode("utf-8")
    if max_bytes <= len(marker_bytes):
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore")

    available = max_bytes - len(marker_bytes)
    head = head_bytes if head_bytes is not None else int(available * 0.30)
    tail = tail_bytes if tail_bytes is not None else (available - head)

    head = max(0, min(head, available))
    tail = max(0, min(tail, available - head))
    if head + tail > available:
        tail = max(0, available - head)

    head_part = raw[:head].decode("utf-8", errors="ignore").rstrip()
    tail_part = raw[len(raw) - tail :].decode("utf-8", errors="ignore").lstrip()

    if not head_part and not tail_part:
        return cut_marker.strip()
    if not head_part:
        return cut_marker.lstrip("\n") + tail_part
    if not tail_part:
        return head_part + cut_marker.rstrip("\n")
    return head_part + cut_marker + tail_part
