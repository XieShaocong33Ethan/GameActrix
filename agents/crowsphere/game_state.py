from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class GameState:
    game: str
    episode_id: str | int
    step_id: int
    obs_text: str
    game_info: dict[str, Any]
    last_action_str: str
    image_present: bool
    image_sha256: str
    image_size: dict[str, int]
    image_preproc: dict[str, Any]
    derived: dict[str, Any]
    memory_short: Any
    memory_retrieved: list[dict[str, Any]]

