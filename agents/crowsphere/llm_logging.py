from __future__ import annotations

import contextvars
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:  # pragma: no cover
    import tiktoken
except Exception:  # pragma: no cover
    tiktoken = None  # type: ignore[assignment]


_CTX_GAME: contextvars.ContextVar[str | None] = contextvars.ContextVar("crowsphere_game", default=None)
_CTX_STEP_ID: contextvars.ContextVar[int | None] = contextvars.ContextVar("crowsphere_step_id", default=None)
_CTX_IMAGE_SHA256: contextvars.ContextVar[str | None] = contextvars.ContextVar("crowsphere_image_sha256", default=None)

_ENCODING_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class CallContextTokens:
    game: contextvars.Token[str | None]
    step_id: contextvars.Token[int | None]
    image_sha256: contextvars.Token[str | None]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def set_call_context(*, game: str | None, step_id: int | None, image_sha256: str | None) -> CallContextTokens:
    return CallContextTokens(
        game=_CTX_GAME.set(game),
        step_id=_CTX_STEP_ID.set(step_id),
        image_sha256=_CTX_IMAGE_SHA256.set(image_sha256),
    )


def reset_call_context(tokens: CallContextTokens) -> None:
    _CTX_GAME.reset(tokens.game)
    _CTX_STEP_ID.reset(tokens.step_id)
    _CTX_IMAGE_SHA256.reset(tokens.image_sha256)


def get_call_context() -> dict[str, Any]:
    return {
        "game": _CTX_GAME.get(),
        "step_id": _CTX_STEP_ID.get(),
        "image_sha256": _CTX_IMAGE_SHA256.get(),
    }


def _redact_image_url(url: str) -> str:
    url = str(url or "")
    if url.startswith("data:image/"):
        return "data:image/<redacted>"
    if len(url) > 256:
        return "<redacted>"
    return url


def _extract_text_from_content(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        out: list[str] = []
        for item in content:
            if isinstance(item, str):
                out.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "")
            if item_type == "text":
                text = item.get("text")
                if isinstance(text, str) and text:
                    out.append(text)
        return out
    return []


def build_retoken_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        texts = _extract_text_from_content(content)
        for t in texts:
            t = str(t)
            if t:
                parts.append(t)
    return "\n".join(parts)


def sanitize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a JSON-serializable request body with image payloads redacted."""
    safe = copy.deepcopy(messages)
    for msg in safe:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if str(item.get("type") or "") != "image_url":
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and "url" in image_url:
                image_url["url"] = _redact_image_url(str(image_url.get("url") or ""))
    return safe


def request_includes_image(messages: list[dict[str, Any]]) -> bool:
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and str(item.get("type") or "") == "image_url":
                return True
    return False


def count_tokens(*, text: str, encoding_name: str = "cl100k_base") -> int | None:
    """Count tokens using tiktoken. Returns None if tiktoken is unavailable."""
    if tiktoken is None:  # pragma: no cover
        return None
    enc = _ENCODING_CACHE.get(encoding_name)
    if enc is None:
        enc = tiktoken.get_encoding(encoding_name)
        _ENCODING_CACHE[encoding_name] = enc
    return int(len(enc.encode(text)))

