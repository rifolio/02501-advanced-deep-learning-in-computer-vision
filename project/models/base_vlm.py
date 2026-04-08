from abc import ABC, abstractmethod

class BaseVLM(ABC):
    def __init__(self, device: str):
        self.device = device
        self.model_name = "BaseVLM"

    @abstractmethod
    def predict(self, image, target_class: str, img_width: int, img_height: int) -> list:
        """
        Takes an image and a target class name.
        Must return a list of bounding boxes in COCO format: 
        [[x, y, width, height], [x, y, width, height], ...]
        """
        pass