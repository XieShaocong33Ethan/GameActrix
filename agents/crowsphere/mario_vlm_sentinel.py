"""Mario VLM Sentinel - 确定性视觉提前告警模块

通过 VLM 视觉感知为 Mario agent 提供"高度差+敌人风险"的提前语义判断，
让 agent 在 600-700 Goomba 死亡段提前做出安全决策。

设计原则：
- 确定性输出：temperature=0，固定 prompt，JSON 格式
- 失败兜底：VLM 调用失败时返回全 False（不引入随机行为）
- 最小侵入：只在明确断言时介入，不影响原有逻辑
"""
from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment]


@dataclass
class SentinelResult:
    """VLM Sentinel 分析结果"""
    on_platform: bool = False           # Mario 是否在管道/砖块/台阶顶
    enemy_below_ahead: bool = False     # 前方低处是否有地面敌人风险
    enemy_ahead: bool = False           # 前方是否有敌人（作为 obs_str 的补充确认）
    enemy_count_ahead: str = "unknown"  # 前方敌人数量桶：none/one/two_plus/unknown
    dense_enemies_ahead: bool = False   # 前方是否“密集敌人”（>=2 且距离近）
    pipe_ahead: bool = False            # 前方是否有管道（Pipe）/高实体障碍
    pipe_distance: str = "unknown"      # 管道距离桶：very_close/close/mid/far/unknown
    pit_ahead: bool = False             # 前方是否有坑（缺口）
    pit_distance: str = "unknown"       # 坑距离桶：very_close/close/mid/far/unknown
    stairs_ahead: bool = False          # 前方是否有台阶（stairs）结构
    stairs_pit_ahead: bool = False      # 前方是否“台阶紧跟坑”（用于 2400-2500 台阶坑）
    low_ceiling: bool = False           # 前方是否低天花板（容易顶头）
    note: str = ""                      # 简短解释（仅用于日志）
    error: str | None = None            # 错误信息（如果有）
    raw_response: str = ""              # 原始 VLM 响应（用于调试）

    def to_obs_block(self) -> str:
        """生成追加到 obs_str 的文本块"""
        return f"""
[VLM Sentinel]
on_platform: {self.on_platform}
enemy_below_ahead: {self.enemy_below_ahead}
enemy_ahead: {self.enemy_ahead}
enemy_count_ahead: {self.enemy_count_ahead}
dense_enemies_ahead: {self.dense_enemies_ahead}
pipe_ahead: {self.pipe_ahead}
pipe_distance: {self.pipe_distance}
pit_ahead: {self.pit_ahead}
pit_distance: {self.pit_distance}
stairs_ahead: {self.stairs_ahead}
stairs_pit_ahead: {self.stairs_pit_ahead}
low_ceiling: {self.low_ceiling}
"""

    def to_dict(self) -> dict[str, Any]:
        """导出为字典（用于日志记录）"""
        return {
            "on_platform": self.on_platform,
            "enemy_below_ahead": self.enemy_below_ahead,
            "enemy_ahead": self.enemy_ahead,
            "enemy_count_ahead": self.enemy_count_ahead,
            "dense_enemies_ahead": self.dense_enemies_ahead,
            "pipe_ahead": self.pipe_ahead,
            "pipe_distance": self.pipe_distance,
            "pit_ahead": self.pit_ahead,
            "pit_distance": self.pit_distance,
            "stairs_ahead": self.stairs_ahead,
            "stairs_pit_ahead": self.stairs_pit_ahead,
            "low_ceiling": self.low_ceiling,
            "note": self.note,
            "error": self.error,
        }


# 固定的 System Prompt - 严格约束输出格式
SENTINEL_SYSTEM_PROMPT = """You are analyzing a Super Mario Bros game frame. Answer ONLY with valid JSON.

Your task is to determine:
1. on_platform: Is Mario (the small character on the left side) currently standing on an elevated surface? Elevated surfaces include: green pipes, brick platforms, stairs, or question blocks. Ground level is NOT elevated.
2. enemy_below_ahead: IMPORTANT - If Mario is on an elevated surface (on_platform=true) AND there is an enemy/monster on the ground ahead, then enemy_below_ahead MUST be true. This includes: Goomba, Koopa, or any other hostile creature walking on the ground while Mario is on a pipe/platform above.
3. enemy_ahead: Is there any enemy/monster visible anywhere ahead (to the right) of Mario? Count ANY hostile creature (not just Goomba/Koopa).
4. enemy_count_ahead: CAREFULLY COUNT how many enemies/monsters are visible ahead of Mario. Scan the ENTIRE screen to the right of Mario. Count Goombas, Koopas, AND any other hostile creatures. Use: "none" (0), "one" (exactly 1), "two_plus" (2 or more), "unknown".
5. dense_enemies_ahead: true if enemy_count_ahead is "two_plus" AND at least two enemies are within about 100-120 pixels ahead. A dense cluster is dangerous because there's no safe landing spot between them.
6. pipe_ahead: Is there a green pipe visible ahead (to the right) of Mario?
7. pipe_distance: Estimate how far the nearest pipe is: "very_close" (<30px), "close" (30-60px), "mid" (60-100px), "far" (>100px), "unknown".
8. pit_ahead: Is there a pit/gap (missing floor) visible ahead of Mario? Look for any break in the walkable floor line. IMPORTANT: In castle levels, lava gaps count as pits/gaps too.
9. pit_distance: Estimate pit distance: "very_close", "close", "mid", "far", "unknown".
10. stairs_ahead: Are there stairs (orange/brown stair-step blocks forming an ascending pattern) visible ahead of Mario? These typically appear near the end of a level.
11. stairs_pit_ahead: In Super Mario Bros 1-1, the end-game stairs ALWAYS have a pit/gap before them that Mario must jump over. If you see stairs (ascending block pattern) ahead, look VERY carefully at the base - if there's ANY gap in the ground before or beside the stairs, set this to true.
12. low_ceiling: Is there a low ceiling ahead (dense bricks/blocks overhead) that would make high jumps risky?

CRITICAL RULES:
- ENEMY COUNTING: Do not just glance - actually COUNT each enemy sprite visible to the right of Mario. Even small brown shapes on the ground are Goombas!
- STAIRS PIT: The classic end-level stairs in 1-1 have a pit. If stairs_ahead=true, assume stairs_pit_ahead=true unless you're 100% certain there's continuous ground.
- If on_platform is true AND enemy_ahead is true AND the enemy is on ground level, then enemy_below_ahead MUST be true.

IMPORTANT:
- Mario is the small character, usually on the left portion of the screen
- "Ahead" means to the RIGHT of Mario
- "Below" means the enemy is at ground level while Mario is elevated
- Goombas look like small brown mushrooms walking on ground
- Koopas look like turtles (green or red shell)
- Other enemies/monsters may have different shapes. Treat ANY hostile creature that can collide with Mario as an enemy.

Output ONLY this JSON format, no other text:
{"on_platform": true, "enemy_below_ahead": true, "enemy_ahead": true, "enemy_count_ahead": "two_plus", "dense_enemies_ahead": true, "pipe_ahead": false, "pipe_distance": "unknown", "pit_ahead": false, "pit_distance": "unknown", "stairs_ahead": false, "stairs_pit_ahead": false, "low_ceiling": false, "note": "Mario on brick, two Goombas close ahead"}"""


def _default_result() -> SentinelResult:
    """返回默认的全 False 结果（兜底）"""
    return SentinelResult(
        on_platform=False,
        enemy_below_ahead=False,
        enemy_ahead=False,
        enemy_count_ahead="unknown",
        dense_enemies_ahead=False,
        pipe_ahead=False,
        pipe_distance="unknown",
        pit_ahead=False,
        pit_distance="unknown",
        stairs_ahead=False,
        stairs_pit_ahead=False,
        low_ceiling=False,
        note="default fallback",
    )


def _image_to_base64_url(image: "Image.Image") -> str:
    """将 PIL Image 转换为 base64 data URL"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _parse_json_response(response_text: str) -> dict[str, Any] | None:
    """从 VLM 响应中解析 JSON"""
    # 尝试直接解析
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        pass

    # 尝试从 markdown code block 中提取
    json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # 尝试找到 JSON 对象
    json_match = re.search(r"\{[^{}]*\}", response_text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


class MarioVLMSentinel:
    """Mario VLM Sentinel - 视觉提前告警器
    
    使用 VLM 分析游戏画面，提供确定性的"高度差+敌人风险"判断。
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 10.0,
        enabled: bool = True,
    ):
        """初始化 Sentinel
        
        Args:
            base_url: OpenAI 兼容 API 的 base URL
            api_key: API key
            model: 模型名称
            timeout: 请求超时时间（秒）
            enabled: 是否启用（False 时直接返回默认结果）
        """
        self.enabled = enabled
        self.timeout = timeout

        # 从环境变量或参数获取配置
        # 注意：VLM Sentinel 优先使用 DashScope，因为需要 VLM 能力
        self.base_url = base_url or os.environ.get(
            "VLM_SENTINEL_BASE_URL",
            "https://dashscope.aliyuncs.com/compatible-mode/v1"  # 强制使用 DashScope
        )
        self.api_key = api_key or os.environ.get(
            "VLM_SENTINEL_API_KEY",
            os.environ.get("DASHSCOPE_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
        )
        self.model = model or os.environ.get(
            "VLM_SENTINEL_MODEL",
            "qwen3-vl-8b-instruct"  # 默认使用 qwen3-vl-8b-instruct（禁止 qwen-vl-plus）
        )

        self._client: "OpenAI | None" = None
        self._init_error: str | None = None

        # 竞赛约束：禁止使用 qwen-vl-plus，只允许 qwen3-vl-8b-instruct。
        requested = str(self.model or "").strip()
        if requested and requested.lower() != "qwen3-vl-8b-instruct":
            self._init_error = (
                f"VLM_SENTINEL_MODEL={requested} 不被允许；只允许使用 qwen3-vl-8b-instruct"
            )
            # 不要让 Sentinel 的配置错误影响主流程：禁用后，Mario 会回退到原来的确定性逻辑。
            self.enabled = False

        if self.enabled:
            self._init_client()

    def _init_client(self) -> None:
        """初始化 OpenAI 客户端"""
        if OpenAI is None:
            self._init_error = "openai package not installed"
            return

        if not self.api_key:
            self._init_error = "API key not configured"
            return

        try:
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
            )
        except Exception as e:
            self._init_error = f"Failed to init client: {e}"

    def analyze(self, image: "Image.Image") -> SentinelResult:
        """分析游戏画面，返回确定性判断
        
        Args:
            image: PIL Image 格式的游戏画面
            
        Returns:
            SentinelResult: 分析结果（确定性布尔值）
        """
        if not self.enabled:
            return _default_result()

        if self._init_error:
            return SentinelResult(error=self._init_error)

        if self._client is None:
            return SentinelResult(error="Client not initialized")

        if Image is None:
            return SentinelResult(error="PIL not installed")

        try:
            image_url = _image_to_base64_url(image)

            # 调用 VLM API - 确定性参数
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": SENTINEL_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url},
                            },
                            {
                                "type": "text",
                                "text": "Analyze this Super Mario Bros frame. Output JSON only.",
                            },
                        ],
                    },
                ],
                temperature=0,  # 确定性
                top_p=1,  # 关闭 nucleus sampling 的随机性
                presence_penalty=0,
                frequency_penalty=0,
                # qwen3-vl-8b-instruct 支持较大的上下文预算；这里给足 token，
                # 避免 JSON 被截断导致解析失败，且保持确定性（temperature=0）。
                # Sentinel 输出很短；512 足够且更稳定/便宜。
                max_tokens=512,
            )

            raw_response = response.choices[0].message.content or ""

            # 解析 JSON 响应
            parsed = _parse_json_response(raw_response)
            if parsed is None:
                return SentinelResult(
                    error=f"Failed to parse JSON from response",
                    raw_response=raw_response,
                )

            # 提取布尔值（宽容处理）
            def to_bool(val: Any) -> bool:
                if isinstance(val, bool):
                    return val
                if isinstance(val, str):
                    return val.lower() in ("true", "yes", "1")
                return bool(val)

            def to_bucket(val: Any) -> str:
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in {"very_close", "close", "mid", "far", "unknown"}:
                        return v
                return "unknown"

            def to_enemy_count(val: Any) -> str:
                if isinstance(val, str):
                    v = val.strip().lower()
                    if v in {"none", "one", "two_plus", "unknown"}:
                        return v
                return "unknown"

            return SentinelResult(
                on_platform=to_bool(parsed.get("on_platform", False)),
                enemy_below_ahead=to_bool(parsed.get("enemy_below_ahead", False)),
                enemy_ahead=to_bool(parsed.get("enemy_ahead", False)),
                enemy_count_ahead=to_enemy_count(parsed.get("enemy_count_ahead", "unknown")),
                dense_enemies_ahead=to_bool(parsed.get("dense_enemies_ahead", False)),
                pipe_ahead=to_bool(parsed.get("pipe_ahead", False)),
                pipe_distance=to_bucket(parsed.get("pipe_distance", "unknown")),
                pit_ahead=to_bool(parsed.get("pit_ahead", False)),
                pit_distance=to_bucket(parsed.get("pit_distance", "unknown")),
                stairs_ahead=to_bool(parsed.get("stairs_ahead", False)),
                stairs_pit_ahead=to_bool(parsed.get("stairs_pit_ahead", False)),
                low_ceiling=to_bool(parsed.get("low_ceiling", False)),
                note=str(parsed.get("note", ""))[:100],
                raw_response=raw_response,
            )

        except Exception as e:
            return SentinelResult(
                error=f"VLM call failed: {type(e).__name__}: {e}",
            )


# 全局单例（延迟初始化）
_global_sentinel: MarioVLMSentinel | None = None


def get_sentinel(enabled: bool = True) -> MarioVLMSentinel:
    """获取全局 Sentinel 实例"""
    global _global_sentinel
    if _global_sentinel is None:
        _global_sentinel = MarioVLMSentinel(enabled=enabled)
    return _global_sentinel


def analyze(image: "Image.Image", *, enabled: bool = True) -> SentinelResult:
    """便捷函数：分析游戏画面
    
    Args:
        image: PIL Image 格式的游戏画面
        enabled: 是否启用 VLM 分析（False 时返回默认结果）
        
    Returns:
        SentinelResult: 分析结果
    """
    sentinel = get_sentinel(enabled=enabled)
    return sentinel.analyze(image)
