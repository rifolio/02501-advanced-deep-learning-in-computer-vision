"""Shared zero-shot localization instructions for VLM bbox models (InternVL, Qwen, etc.)."""

from __future__ import annotations


def zero_shot_detection_instructions(target_class: str) -> str:
    """
    Behavioral context only (format rules stay in model-specific tails where needed).

    Goal: discourage whole-frame collapse and sloppy boxes; encourage tight boxes around
    full instances across scale variation.
    """
    cls = target_class.strip()
    return (
        f"Find every visible {cls} in the image.\n"
        "Output one tight axis-aligned box per distinct instance—the box must wrap the "
        "entire object (all visible parts), not a limb or crop, and avoid large empty background.\n"
        "Scale varies (objects can be distant and small or close and large); each box "
        "should match the object's real footprint.\n"
        f"Do not return a single box that covers nearly the whole image unless one {cls} "
        "actually fills most of the frame.\n"
    )
