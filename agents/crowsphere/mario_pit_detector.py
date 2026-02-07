from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PitDetectorResult:
    pit_ahead: bool = False
    pit_distance: str = "unknown"  # very_close/close/mid/far/unknown
    pit_start_dx: int | None = None
    pit_end_dx: int | None = None
    note: str = ""

    def to_obs_block(self) -> str:
        if not self.pit_ahead:
            return ""
        lines = [
            "",
            "[Mario Pit Detector]",
            f"pit_ahead: {bool(self.pit_ahead)}",
            f"pit_distance: {self.pit_distance}",
        ]
        if self.pit_start_dx is not None:
            lines.append(f"pit_start_dx: {int(self.pit_start_dx)}")
        if self.pit_end_dx is not None:
            lines.append(f"pit_end_dx: {int(self.pit_end_dx)}")
        if self.note:
            lines.append(f"note: {self.note}")
        return "\n".join(lines) + "\n"


def _parse_mario_screen_x(obs_text: str) -> int | None:
    text = str(obs_text or "")
    match = re.search(r"Position of Mario:\s*\((\d+)\s*,\s*(-?\d+)\)", text)
    if not match:
        return None
    try:
        return int(match.group(1))
    except Exception:
        return None


def _bucket_for_dx(dx: int) -> str:
    if dx < 20:
        return "very_close"
    if dx < 45:
        return "close"
    if dx < 90:
        return "mid"
    if dx < 140:
        return "far"
    return "unknown"


def analyze_mario_pit_from_image(*, image: Any, obs_text: str) -> PitDetectorResult:
    """
    确定性“坑/岩浆”检测器（不依赖模板匹配）。

    目标：
    - 在 RandomStages / castle lava 等模板漏检时，尽量检测到“前方有坑/岩浆”
    - 输出一个粗粒度的距离桶 + 估计的 pit_start_dx/pit_end_dx（屏幕坐标相对 Mario）

    设计取舍：
    - 只看屏幕底部区域（y≈190..240），避免把上方 UI/背景误判成坑
    - 默认只检测“岩浆坑”（红橙色）。普通黑洞坑在 1-1 的地面纹理里容易产生误报，
      这里宁可漏检，也不要在官方 1-1 里产生大量 false positive。
    """
    if np is None or Image is None:  # pragma: no cover
        return PitDetectorResult(pit_ahead=False, note="numpy_or_pil_missing")

    if image is None:
        return PitDetectorResult(pit_ahead=False, note="no_image")

    try:
        if isinstance(image, Image.Image):
            rgb = image.convert("RGB")
        else:
            rgb = Image.fromarray(np.asarray(image)).convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            return PitDetectorResult(pit_ahead=False, note="bad_image_shape")
    except Exception:  # pragma: no cover
        return PitDetectorResult(pit_ahead=False, note="image_decode_failed")

    mario_x = _parse_mario_screen_x(obs_text)
    if mario_x is None:
        # 官方 obs_text x 会 clamp 到 122；这里用它作为兜底不会太差。
        mario_x = 122

    height, width, _ = arr.shape
    # 只看底部 50px，覆盖地面坑与 castle lava
    y0 = max(0, min(height, 190))
    region = arr[y0:height, :, :]
    r = region[:, :, 0].astype(np.int16)
    g = region[:, :, 1].astype(np.int16)
    b = region[:, :, 2].astype(np.int16)

    # 岩浆：亮红/橙（R 很高、B 很低）。
    # 注意：1-1 的地面/砖块也可能是“偏红棕色”，因此阈值要足够严格，避免误报。
    # 经验：把阈值收紧到“接近纯红/亮橙”，可显著降低 1-1 的棕色地面/砖块误报。
    lava = (r >= 240) & (g <= 160) & (b <= 60)
    hazard = lava

    # 每列 hazard 占比
    col_ratio = hazard.mean(axis=0)
    # 阈值：要求底部区域至少 50% 是 hazard，避免零散像素/阴影误判
    hazard_cols = col_ratio > 0.50

    # 取足够长的连续段，过滤噪声
    segments: list[tuple[int, int]] = []
    # 最小宽度：普通关卡的地面纹理/管道阴影可能造成窄条误报；
    # lava 段通常远大于 1 tile，因此用更大的阈值过滤窄噪声。
    MIN_SEGMENT_WIDTH_PX = 16
    start: int | None = None
    for x, v in enumerate(hazard_cols):
        if v and start is None:
            start = x
        if (not v) and start is not None:
            end = x - 1
            if end - start + 1 >= MIN_SEGMENT_WIDTH_PX:
                segments.append((start, end))
            start = None
    if start is not None:
        end = width - 1
        if end - start + 1 >= MIN_SEGMENT_WIDTH_PX:
            segments.append((start, end))

    # 只关心 Mario 右侧的最近 hazard 段
    ahead = [seg for seg in segments if seg[1] > (mario_x + 10)]
    if not ahead:
        return PitDetectorResult(pit_ahead=False)
    seg = min(ahead, key=lambda s: max(0, s[0] - mario_x))
    pit_start_x, pit_end_x = seg
    pit_start_dx = int(pit_start_x) - int(mario_x)
    pit_end_dx = int(pit_end_x) - int(mario_x)
    if pit_end_dx <= 0:
        return PitDetectorResult(pit_ahead=False)

    bucket = _bucket_for_dx(max(0, pit_start_dx))
    # 如果 hazard 段延伸到屏幕右边缘，end 往往不可观测（继续延伸到屏幕外），
    # 将其视为 unknown，避免把“不可见的延伸”误当成超宽坑。
    pit_end_dx_out: int | None = None
    if int(pit_end_x) < int(width - 1):
        pit_end_dx_out = pit_end_dx if pit_end_dx > 0 else None
    return PitDetectorResult(
        pit_ahead=True,
        pit_distance=bucket,
        pit_start_dx=pit_start_dx if pit_start_dx > 0 else None,
        pit_end_dx=pit_end_dx_out,
        note="bottom_hazard",
    )
