from __future__ import annotations

from PIL import Image, ImageDraw


def render_bboxes(
    image: Image.Image,
    boxes: list[list[float]],
    color: str = "red",
    line_width: int = 3,
    numbered_labels: bool = False,
) -> Image.Image:
    rendered = image.copy()
    draw = ImageDraw.Draw(rendered)

    for idx, box in enumerate(boxes, start=1):
        x, y, w, h = box
        x2 = x + w
        y2 = y + h
        draw.rectangle([x, y, x2, y2], outline=color, width=line_width)

        if numbered_labels:
            label = str(idx)
            text_bbox = draw.textbbox((x, y), label)
            pad = 2
            bg_box = [
                text_bbox[0] - pad,
                text_bbox[1] - pad,
                text_bbox[2] + pad,
                text_bbox[3] + pad,
            ]
            draw.rectangle(bg_box, fill=color)
            draw.text((x, y), label, fill="white")

    return rendered
