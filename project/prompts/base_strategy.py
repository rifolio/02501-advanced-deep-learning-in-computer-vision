from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from PIL import Image

from data.support_sampler import SupportExample


class PromptStrategy(ABC):
    @abstractmethod
    def build_prompt(
        self,
        query_image: Image.Image,
        support_by_cat: dict[int, list[SupportExample]],
        class_name: str,
        cat_id: int,
    ) -> dict[str, Any]:
        """
        Build a prompt bundle for one target category in one query image.

        Returns a dict with:
          - images: list[PIL.Image]
          - text: str
          - optional extra keys consumed by the experiment/model.
        """
        raise NotImplementedError
