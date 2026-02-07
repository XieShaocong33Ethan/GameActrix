from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LowCeilingMemory:
    """
    低天花板的短时记忆（用于解决模板匹配偶发漏检）。

    目标：
    - 当“前方/近处”确实进入低天花板段时，允许在接下来的少量 step 内保持 low_ceiling=true，
      即使某一帧 obs_text 里只检测到身后砖块或暂时没有检测到砖块。
    - 但“仅身后远处（例如 -30px）的一块低砖”不应单独触发 low ceiling（避免误报导致 clamp）。
    """

    ttl_steps: int = 6
    # 在 strong_seen 之后，允许弱证据（只剩身后低砖）短暂“续命”的次数。
    # 目的：覆盖模板匹配短时滞后，但避免仅凭单块身后低砖把 low-ceiling 永久锁死。
    weak_refresh_budget_init: int = 2
    active: bool = False
    steps_since_seen: int = 999
    weak_refresh_budget: int = 0

    def reset(self) -> None:
        self.active = False
        self.steps_since_seen = 999
        self.weak_refresh_budget = 0

    def update(self, *, strong_seen: bool, weak_seen: bool) -> None:
        """
        更新记忆状态。

        Args:
            strong_seen: 本步检测到“可信低天花板证据”（例如 Mario 前方/近处出现低砖，或 sentinel 明确标注）。
            weak_seen:   本步仅检测到“弱证据”（例如 Mario 身后较远处出现低砖），仅在 active 且预算未耗尽时用于短暂续命。
        """
        if bool(strong_seen):
            self.active = True
            self.steps_since_seen = 0
            self.weak_refresh_budget = max(0, int(self.weak_refresh_budget_init))
            return

        if self.active and bool(weak_seen) and int(self.weak_refresh_budget) > 0:
            # 弱证据只允许有限次续命，避免“仅一块身后低砖”让 low-ceiling 永久不退出。
            self.steps_since_seen = 0
            self.weak_refresh_budget = int(self.weak_refresh_budget) - 1
            return

        if self.active:
            self.steps_since_seen += 1
            if int(self.steps_since_seen) > int(self.ttl_steps):
                self.active = False
