from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_default(value: Any):
    """
    gym / numpy 等库可能在 info 里返回 numpy scalar（例如 uint8/int64），
    直接 json.dumps 会失败。这里尽量转成 Python 内置类型，保证日志可写。
    """
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass
    if isinstance(value, Path):
        return str(value)
    return str(value)


_LOCKS_GUARD = threading.Lock()
_LOCKS_BY_PATH: dict[str, threading.Lock] = {}


def _get_shared_lock(path: Path) -> threading.Lock:
    # Multiple agent instances (4 games in parallel) can write to the same JSONL file.
    # Use a process-local shared lock keyed by resolved path to avoid line interleaving.
    key = str(path.expanduser().resolve())
    with _LOCKS_GUARD:
        lock = _LOCKS_BY_PATH.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS_BY_PATH[key] = lock
        return lock


@dataclass(frozen=True)
class JsonlLogger:
    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        object.__setattr__(self, "_lock", _get_shared_lock(self.path))

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
