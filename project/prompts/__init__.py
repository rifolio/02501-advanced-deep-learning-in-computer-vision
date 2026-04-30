from .base_strategy import PromptStrategy
from .cropped_exemplars import CroppedExemplarsStrategy
from .set_of_mark import SetOfMarkStrategy
from .side_by_side import SideBySideStrategy
from .side_by_side_with_coords import SideBySideWithCoordsStrategy
from .text_from_vision import TextFromVisionStrategy
from .vlm_text_generation import VLMTextGenerationStrategy
from .verification_strategy import VerificationStrategy


def get_prompt_strategy(name: str) -> PromptStrategy:
    strategy_map = {
        "side_by_side": SideBySideStrategy,
        "side_by_side_with_coords": SideBySideWithCoordsStrategy,
        "cropped_exemplars": CroppedExemplarsStrategy,
        "text_from_vision": TextFromVisionStrategy,
        "set_of_mark": SetOfMarkStrategy,
        "vlm_text_generation": VLMTextGenerationStrategy,
        "verification": VerificationStrategy,
    }
    if name not in strategy_map:
        valid = ", ".join(sorted(strategy_map))
        raise ValueError(f"Unknown prompt strategy '{name}'. Valid values: {valid}")
    return strategy_map[name]()
