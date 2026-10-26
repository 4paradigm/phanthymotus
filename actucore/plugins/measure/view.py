"""Draw measured boxes and labels on the camera frame.

A JPEG in, a JPEG out. The card decides when to publish it; this module only
answers whether the picture shows the same boxes the JSON reports.

**Pillow, not OpenCV.** The jp5.11 actucore image has no `libffi.so.8`, which
OpenCV's Python wheel needs; importing cv2 there raises before a single frame is
drawn. Pillow uses the image codecs already present in the base image and is a
much smaller dependency for drawing four rectangles and some ASCII text.
"""

from __future__ import annotations

import io


def render(camera_jpeg: bytes, objects: list[dict], *, quality: int = 85) -> bytes:
    """Return a JPEG with every measured box and its W×H label drawn on it."""
    from PIL import Image, ImageDraw, ImageFont

    if not camera_jpeg:
        raise ValueError("empty camera frame")
    try:
        image = Image.open(io.BytesIO(camera_jpeg)).convert("RGB")
    except Exception as exc:                                  # noqa: BLE001
        raise ValueError(f"camera frame is not a decodable JPEG: {exc}") from exc

    width, height = image.size
    draw = ImageDraw.Draw(image)
    font = _font(ImageFont)

    for obj in objects or ():
        box = obj.get("box")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        if not all(v == v and abs(v) != float("inf") for v in (x1, y1, x2, y2)):
            continue
        x1 = max(0, min(width - 1, int(round(x1 * width))))
        y1 = max(0, min(height - 1, int(round(y1 * height))))
        x2 = max(x1 + 1, min(width - 1, int(round(x2 * width))))
        y2 = max(y1 + 1, min(height - 1, int(round(y2 * height))))

        # Yellow means the box touches a frame edge: the number is a lower
        # bound, not a complete object. Green means a normal visible box.
        colour = (255, 200, 0) if obj.get("clipped") else (80, 220, 0)
        draw.rectangle([x1, y1, x2, y2], outline=(10, 10, 10), width=4)
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)

        label = str(obj.get("label") or obj.get("name") or "object")
        # cv2/Pillow's default fonts have no CJK; keep this ASCII and keep prose
        # in info()/history where it renders properly.
        draw.text((x1 + 3, max(0, y1 - 17)), label, fill=colour,
                  font=font, stroke_width=2, stroke_fill=(10, 10, 10))

    quality = max(40, min(95, int(quality)))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()


def _font(ImageFont):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        return ImageFont.load_default()
