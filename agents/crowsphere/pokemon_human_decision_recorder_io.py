from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


def save_screenshot(
    *,
    obs: Any,
    step: int,
    screenshots_dir: Path,
    timestamp: str,
    tag: str = "",
) -> str | None:
    """
    保存游戏截图。

    返回值是相对于输出目录的相对路径（用于写入 highlights.md）。
    """
    try:
        if not hasattr(obs, "image") or obs.image is None:
            return None

        tag_str = f"_{tag}" if tag else ""
        filename = f"step_{step:04d}{tag_str}.png"
        filepath = screenshots_dir / filename
        obs.image.save(filepath)
        return f"screenshots_{timestamp}/{filename}"
    except Exception:
        return None


def save_checkpoint(
    *,
    env: Any,
    step: int,
    tag: str,
    checkpoint_dir: Path,
    memory: dict[str, Any],
    stats: dict[str, Any],
    prev_obs_text: str,
    config: dict[str, Any],
) -> str | None:
    """
    保存 checkpoint（游戏状态 + 记录器状态）。

    Returns:
        checkpoint 目录名称；失败时返回 None。
    """
    try:
        tag_str = f"_{tag}" if tag else ""
        checkpoint_name = f"step_{step:04d}{tag_str}"
        checkpoint_path = checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_file = checkpoint_path / "game.state"
        with open(state_file, "wb") as f:
            runner = getattr(env, "runner", None)
            lock = getattr(runner, "lock", None) if runner is not None else None
            if lock is not None:
                with lock:
                    runner.pyboy.save_state(f)
            else:
                env.runner.pyboy.save_state(f)

        metadata = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "memory": dict(memory),
            "stats": {
                "env_tool_count": stats.get("env_tool_count", 0),
                "keys_count": stats.get("keys_count", 0),
                "fallback_count": stats.get("fallback_count", 0),
                "by_state": dict(stats.get("by_state", {})),
            },
            "prev_obs_text": prev_obs_text,
            "config": config,
        }
        metadata_file = checkpoint_path / "metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return checkpoint_name
    except Exception:
        return None


def load_checkpoint(
    *,
    env: Any,
    checkpoint_path: str,
) -> dict[str, Any] | None:
    """
    加载 checkpoint，返回 metadata（包含 step/memory/stats/prev_obs_text/config）。
    加载失败返回 None。
    """
    checkpoint_dir = Path(checkpoint_path)
    if not checkpoint_dir.exists():
        return None

    state_file = checkpoint_dir / "game.state"
    if not state_file.exists():
        return None

    try:
        with open(state_file, "rb") as f:
            runner = getattr(env, "runner", None)
            lock = getattr(runner, "lock", None) if runner is not None else None
            if lock is not None:
                with lock:
                    runner.pyboy.load_state(f)
            else:
                env.runner.pyboy.load_state(f)

        metadata_file = checkpoint_dir / "metadata.json"
        if not metadata_file.exists():
            return None
        with open(metadata_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

