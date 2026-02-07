"""2048 游戏 Tool 定义。

规则约束（重要）：
- 2048 模式下只允许 `calculator` 工具。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ToolResult:
    """工具执行结果"""
    tool_name: str
    success: bool
    result: Any
    error: str | None = None


# 2048 游戏可用工具列表（只允许 calculator）
AVAILABLE_TOOLS_2048 = ["calculator"]

# 工具描述（用于 Prompt）
TOOL_DESCRIPTIONS_2048 = {
    "calculator": "计算数学表达式（仅允许四则运算、括号、以及少量安全函数如 abs/min/max）",
}
