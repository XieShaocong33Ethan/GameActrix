from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agents.crowsphere.mario_jump_rules import KOOPA_SIZE, MARIO_SPEED_PER_STEP
from agents.crowsphere.mario_preprocess.types import Position, ThreatInfo


KOOPA_MEMORY_STEPS = 5
KOOPA_SHADOW_DECAY_PX = MARIO_SPEED_PER_STEP
KOOPA_SHADOW_MIN_DIST_PX = 35


@dataclass
class KoopaMemory:
    """
    跨 step 维护 Koopa 的短时记忆，用于在 flicker 时保持风险意识。

    使用方式：
    1. 创建实例：memory = KoopaMemory()
    2. 每步调用：memory.update(koopas_ahead, mario_x)
    3. 获取 shadow：shadow = memory.get_shadow_koopa()

    提交官方时，在 agent 类（如 CrowSphereEngine）中维护此实例，
    reset 时调用 memory.reset()。
    """

    last_seen_dx: float | None = None
    last_seen_y: int | None = None
    steps_since_seen: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.last_seen_dx = None
        self.last_seen_y = None
        self.steps_since_seen = 0
        self.history = []

    def update(self, koopas_ahead: list[Position], mario_x: int) -> None:
        if koopas_ahead:
            nearest = min(koopas_ahead, key=lambda p: p.x - mario_x)
            dx = nearest.x - mario_x
            if dx > 0:
                self.last_seen_dx = dx
                self.last_seen_y = nearest.y
                self.steps_since_seen = 0
                self.history.append({"step": len(self.history), "dx": dx, "seen": True})
                return

        if self.last_seen_dx is not None:
            self.steps_since_seen += 1
            self.last_seen_dx = max(0.0, float(self.last_seen_dx) - float(KOOPA_SHADOW_DECAY_PX))
            self.history.append(
                {"step": len(self.history), "dx": self.last_seen_dx, "seen": False, "shadow": True}
            )

    def get_shadow_koopa(self) -> ThreatInfo | None:
        if self.last_seen_dx is None:
            return None
        if self.steps_since_seen == 0:
            return None
        if self.steps_since_seen > KOOPA_MEMORY_STEPS:
            return None
        if self.last_seen_dx <= 0:
            return None
        # 当 shadow 已经非常接近时，如果当前帧仍完全检测不到 Koopa，
        # 更可能是“已经越过/被踩死/模板抖动误报”，继续把它当作最近威胁会误导策略。
        if float(self.last_seen_dx) < float(KOOPA_SHADOW_MIN_DIST_PX):
            return None

        return ThreatInfo(
            name="Koopa_shadow",
            position=Position(x=int(self.last_seen_dx), y=self.last_seen_y or 0),
            distance=float(self.last_seen_dx),
            height=KOOPA_SIZE,
        )
