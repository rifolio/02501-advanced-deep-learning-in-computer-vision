from abc import ABC, abstractmethod

class BaseVLM(ABC):
    def __init__(self, device: str):
        self.device = device
        self.model_name = "BaseVLM"
        print(self.device)

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