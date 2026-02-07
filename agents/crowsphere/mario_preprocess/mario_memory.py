from __future__ import annotations

from dataclasses import dataclass, field

from agents.crowsphere.mario_preprocess.koopa_memory import KoopaMemory
from agents.crowsphere.mario_preprocess.low_ceiling_memory import LowCeilingMemory


@dataclass
class MarioPreprocessMemory:
    """
    Mario 预处理器的跨 step 记忆容器。

    说明：
    - 目前包含：
      1) KoopaMemory：处理 Koopa 模板抖动导致的 flicker
      2) LowCeilingMemory：处理低天花板段的短时召回
    - 该对象应由 Engine 在每个 episode 内持有，并在 reset 时调用 reset()。
    """

    koopa_memory: KoopaMemory = field(default_factory=KoopaMemory)
    low_ceiling_memory: LowCeilingMemory = field(default_factory=LowCeilingMemory)

    def reset(self) -> None:
        self.koopa_memory.reset()
        self.low_ceiling_memory.reset()

