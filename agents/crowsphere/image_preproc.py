from __future__ import annotations

import hashlib
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image


@dataclass(frozen=True)
class PreprocResult:
    jpeg_bytes: bytes
    image_sha256: str
    image_size: dict[str, int]
    image_preproc: dict[str, Any]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def preprocess_image(
    image: Image.Image,
    *,
    short_side: int,
    jpeg_quality: int,
) -> PreprocResult:
    if image.mode != "RGB":
        image = image.convert("RGB")

    width, height = image.size
    if width == 0 or height == 0:
        raise ValueError("图片尺寸非法")

    scale = short_side / min(width, height)
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = image.resize((new_w, new_h), resample=Image.BICUBIC)

    buf = BytesIO()
    resized.save(buf, format="JPEG", quality=jpeg_quality)
    jpeg_bytes = buf.getvalue()

    return PreprocResult(
        jpeg_bytes=jpeg_bytes,
        image_sha256=_sha256_hex(jpeg_bytes),
        image_size={"w": width, "h": height},
        image_preproc={"short_side": short_side, "jpeg_quality": jpeg_quality},
    )

