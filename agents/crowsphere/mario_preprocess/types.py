from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    x: int
    y: int


@dataclass
class PipeInfo:
    x: int
    y: int
    height: int


@dataclass
class ThreatInfo:
    name: str
    position: Position
    distance: float
    height: int  # 障碍物高度，用于决定跳跃级别

