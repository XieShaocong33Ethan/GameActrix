from __future__ import annotations

from dataclasses import dataclass
import itertools
import threading
import time
from typing import Any

from openai import OpenAI
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError

from agents.crowsphere.llm_logging import (
    build_retoken_text,
    count_tokens,
    get_call_context,
    request_includes_image,
    sanitize_messages,
    utc_now_iso,
)
from agents.crowsphere.logging_jsonl import JsonlLogger


@dataclass(frozen=True)
class VllmRequest:
    messages: list[dict[str, Any]]
    params: dict[str, Any]


_CALL_ID_GUARD = threading.Lock()
_CALL_ID = itertools.count(1)


def _next_call_id() -> int:
    with _CALL_ID_GUARD:
        return int(next(_CALL_ID))


class OpenAICompatClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str,
        timeout_s: float = 25.0,
        max_retries: int = 2,
        retry_backoff_s: float = 0.5,
        call_logger: JsonlLogger | None = None,
        tokenizer_name: str = "cl100k_base",
    ) -> None:
        self._base_url = base_url
        self._model = model
        # Disable the OpenAI SDK's own retries so we can strictly bound latency.
        # This is critical for environments like the official eval kit where the game session
        # expires if the agent does not send an action within ~120 seconds.
        try:
            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s, max_retries=0)
        except TypeError:  # pragma: no cover
            self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout_s)
        self._max_retries = max(1, int(max_retries))
        self._retry_backoff_s = max(0.0, float(retry_backoff_s))
        self._call_logger = call_logger
        self._tokenizer_name = str(tokenizer_name or "cl100k_base")

    def chat_completions(self, request: VllmRequest) -> str:
        params = dict(request.params or {})
        last_exc: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            call_id = _next_call_id()
            ts_utc = utc_now_iso()
            ctx = get_call_context()
            retoken_text = build_retoken_text(request.messages)
            prompt_tokens = count_tokens(text=retoken_text, encoding_name=self._tokenizer_name)
            safe_messages = sanitize_messages(request.messages)
            has_image = request_includes_image(request.messages)
            t0 = time.time()
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=request.messages,
                    **params,
                )
                content = response.choices[0].message.content
                latency_s = time.time() - t0
                if self._call_logger is not None:
                    self._call_logger.write(
                        {
                            "event": "llm_call",
                            "call_id": call_id,
                            "timestamp_utc": ts_utc,
                            "provider": {"base_url": self._base_url, "model": self._model},
                            "context": ctx,
                            "request": {
                                "messages": safe_messages,
                                "retoken_text": retoken_text,
                                "params": params,
                                "has_image": bool(has_image),
                                "image_redacted": True,
                            },
                            "tokenizer": {"name": self._tokenizer_name, "prompt_tokens": prompt_tokens},
                            "response": {"raw_text": str(content or "")},
                            "latency_s": float(latency_s),
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "status": "ok",
                        }
                    )
                return str(content or "")
            except (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError) as exc:
                last_exc = exc
                if self._call_logger is not None:
                    self._call_logger.write(
                        {
                            "event": "llm_call",
                            "call_id": call_id,
                            "timestamp_utc": ts_utc,
                            "provider": {"base_url": self._base_url, "model": self._model},
                            "context": ctx,
                            "request": {
                                "messages": safe_messages,
                                "retoken_text": retoken_text,
                                "params": params,
                                "has_image": bool(has_image),
                                "image_redacted": True,
                            },
                            "tokenizer": {"name": self._tokenizer_name, "prompt_tokens": prompt_tokens},
                            "response": None,
                            "latency_s": float(time.time() - t0),
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "status": "error",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                if attempt >= self._max_retries:
                    raise
                time.sleep(self._retry_backoff_s * (2 ** (attempt - 1)))
            except Exception as exc:
                # Non-retryable provider errors still count as an inference call; log them too.
                if self._call_logger is not None:
                    self._call_logger.write(
                        {
                            "event": "llm_call",
                            "call_id": call_id,
                            "timestamp_utc": ts_utc,
                            "provider": {"base_url": self._base_url, "model": self._model},
                            "context": ctx,
                            "request": {
                                "messages": safe_messages,
                                "retoken_text": retoken_text,
                                "params": params,
                                "has_image": bool(has_image),
                                "image_redacted": True,
                            },
                            "tokenizer": {"name": self._tokenizer_name, "prompt_tokens": prompt_tokens},
                            "response": None,
                            "latency_s": float(time.time() - t0),
                            "attempt": attempt,
                            "max_retries": self._max_retries,
                            "status": "error",
                            "error": {"type": type(exc).__name__, "message": str(exc)},
                        }
                    )
                raise

        raise RuntimeError(f"unreachable: last_exc={last_exc!r}")


VllmClient = OpenAICompatClient
