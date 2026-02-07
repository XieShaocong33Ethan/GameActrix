"""Shared helpers for CrowSphereEngine.

Keep `engine.py` focused on orchestration. This module hosts small, reusable
helpers that are not game-specific.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any


def is_data_inspection_failed(exc: Exception) -> bool:
    """Detect OpenAI-compatible 'data_inspection_failed' errors (image moderation).

    Some providers reject certain images. When that happens, we retry the same
    request in text-only mode.
    """
    if "data_inspection_failed" in str(exc):
        return True
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
            return str(code) == "data_inspection_failed"
    return False


def resolve_api_key(*, base_url: str) -> str:
    """Resolve API key/token for the configured model provider."""
    if os.getenv("CROWSPHERE_API_KEY"):
        return str(os.getenv("CROWSPHERE_API_KEY"))
    base_url_lower = base_url.lower()
    if "dashscope.aliyuncs.com" in base_url_lower:
        value = os.getenv("DASHSCOPE_API_KEY")
        if value:
            return value
        if os.getenv("CROWSPHERE_ALLOW_EMPTY_API_KEY") == "1":
            return "EMPTY"
        raise RuntimeError(
            "DashScope 推理需要 API Key，请设置环境变量 DASHSCOPE_API_KEY（或设置 CROWSPHERE_ALLOW_EMPTY_API_KEY=1 仅用于本地无模型运行）"
        )
    if "modelscope" in base_url_lower:
        value = os.getenv("MODELSCOPE_API_KEY")
        if value:
            return value
        token_file = Path(".modelscope_token")
        if token_file.exists():
            return token_file.read_text(encoding="utf-8").strip()
        if os.getenv("CROWSPHERE_ALLOW_EMPTY_API_KEY") == "1":
            return "EMPTY"
        raise RuntimeError(
            "ModelScope 推理需要 Token，请设置环境变量 MODELSCOPE_API_KEY（或放到 .modelscope_token；或设置 CROWSPHERE_ALLOW_EMPTY_API_KEY=1 仅用于本地无模型运行）"
        )
    return os.getenv("OPENAI_API_KEY") or "EMPTY"


def jpeg_bytes_to_data_url(jpeg_bytes: bytes) -> str:
    return f"data:image/jpeg;base64,{base64.b64encode(jpeg_bytes).decode('utf-8')}"


def build_messages(
    *,
    prompt: str,
    image_data_url: str | None,
    prev_image_url: str | None = None,
    game: str = "",
) -> list[dict[str, Any]]:
    # Note: `prev_image_url` is intentionally unused for now. Some backends support
    # referencing previous images; we keep the signature stable for future use.
    system = {"role": "system", "content": "你是一个严格输出 JSON 的游戏智能体。"}
    content: list[dict[str, Any]] = []
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    content.append({"type": "text", "text": prompt})
    return [system, {"role": "user", "content": content}]

