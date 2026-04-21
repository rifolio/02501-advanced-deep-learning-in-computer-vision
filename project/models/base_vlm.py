from abc import ABC, abstractmethod

class BaseVLM(ABC):
    def __init__(self, device: str):
        self.device = device
        self.model_name = "BaseVLM"
        self._runtime_stats: dict[str, int] = {}

    @abstractmethod
    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        """
        Takes an image and a target class name.
        Must return a list of bounding boxes in COCO format: 
        [[x, y, width, height], [x, y, width, height], ...]
        """
        pass

    def predict_few_shot(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list:
        """
        Few-shot detection entrypoint.
        Models that do not support multi-image few-shot should override this.
        """
        raise NotImplementedError(f"{self.model_name} does not implement predict_few_shot")

    def _bump_runtime_stat(self, key: str, value: int = 1) -> None:
        self._runtime_stats[key] = self._runtime_stats.get(key, 0) + value

    def pop_runtime_stats(self) -> dict[str, int]:
        stats = dict(self._runtime_stats)
        self._runtime_stats.clear()
        return stats