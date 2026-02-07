from __future__ import annotations

import atexit
import os
import threading
from pathlib import Path


_REGISTER_GUARD = threading.Lock()
_REGISTERED = False


def _resolve_game_data_dir() -> Path | None:
    """
    官方 starter kit 会通过环境变量 GAME_DATA_DIR 指定日志目录（包含各游戏的 game_states.jsonl）。
    我们把评测交付物也写到同一个目录，便于线上收集。
    """
    raw = (os.getenv("GAME_DATA_DIR") or "").strip()
    if not raw:
        return None
    return Path(raw).expanduser().resolve()


def register_eval_artifacts_autowrite() -> None:
    """
    在进程退出时生成评测交付物（不修改官方 runner）。

    依赖：
    - GAME_DATA_DIR/<game>/game_states.jsonl（官方 runner 负责写）
    - GAME_DATA_DIR/llm_calls.jsonl（我们的 OpenAICompatClient 负责写）
    """
    global _REGISTERED
    with _REGISTER_GUARD:
        if _REGISTERED:
            return
        _REGISTERED = True

    def _on_exit() -> None:
        game_data_dir = _resolve_game_data_dir()
        if game_data_dir is None:
            return

        # 避免在没有跑评测的情况下产出一堆空文件（例如仅导入了包）。
        llm_calls = game_data_dir / "llm_calls.jsonl"
        has_game_states = any(
            (game_data_dir / game / "game_states.jsonl").exists()
            for game in ("twenty_fourty_eight", "super_mario", "pokemon_red", "star_craft")
        )
        if not has_game_states and not llm_calls.exists():
            return

        try:
            from agents.crowsphere.eval_artifacts import write_evaluation_artifacts

            write_evaluation_artifacts(
                output_dir=game_data_dir,
                game_data_dir=game_data_dir,
                llm_calls_path=llm_calls,
            )
        except Exception:
            # 交付物生成失败不应该影响评测主流程；线上由 organizer 兜底复算。
            return

    atexit.register(_on_exit)

