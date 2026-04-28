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

    def predict_with_scores(
        self,
        image,
        target_class: str,
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        """
        Backward-compatible scored prediction API.

        Returns:
            [
                {
                    "bbox": [x, y, width, height],
                    "score": float in [0, 1],
                    "score_source": "model" | "provisional",
                    "score_policy": str,
                },
                ...
            ]

        Default implementation wraps `predict(...)` and assigns provisional
        scores for models that do not expose native confidence values.
        """
        boxes = self.predict(image, target_class, img_width, img_height)
        return self._build_provisional_scored_predictions(boxes)

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

    def predict_few_shot_with_scores(
        self,
        query_image,
        support_images: list,
        prompt_text: str,
        img_width: int,
        img_height: int,
    ) -> list[dict]:
        """
        Backward-compatible few-shot scored prediction API.

        Default implementation wraps `predict_few_shot(...)` and assigns
        provisional scores when the model does not expose native confidence.
        """
        boxes = self.predict_few_shot(
            query_image,
            support_images,
            prompt_text,
            img_width,
            img_height,
        )
        return self._build_provisional_scored_predictions(boxes)

    def _bump_runtime_stat(self, key: str, value: int = 1) -> None:
        self._runtime_stats[key] = self._runtime_stats.get(key, 0) + value

    def pop_runtime_stats(self) -> dict[str, int]:
        stats = dict(self._runtime_stats)
        self._runtime_stats.clear()
        return stats

    def _provisional_score_policy(self) -> str:
        """
        Policy used by default scored predictions when model confidence is absent.
        """
        return "provisional_rank_decay_v1"

    def _build_provisional_scored_predictions(self, boxes: list) -> list[dict]:
        """
        Build scored predictions with a clearly marked provisional policy.

        Policy details (`provisional_rank_decay_v1`):
        - score(rank) = max(0.05, 0.5 / (rank + 1))
        - rank is the model output order index (0-based)
        """
        policy = self._provisional_score_policy()
        scored_predictions = []
        for rank, box in enumerate(boxes):
            score = max(0.05, 0.5 / float(rank + 1))
            scored_predictions.append(
                {
                    "bbox": box,
                    "score": score,
                    "score_source": "provisional",
                    "score_policy": policy,
                }
            )
        return scored_predictions