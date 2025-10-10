from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from raiki_sdk.base.schemas import YoloResult, YoloValidation


class YoloBase(ABC):
    @abstractmethod
    def yolo_detect(self, image: np.ndarray) -> YoloResult:
        """
        Perform YOLO detection on the image and return the result.
        """
        pass

    def validate(self, result: YoloResult) -> YoloValidation:
        """
        Validate the YOLO detection result and return the validation result.
        """
        pass

    def crop(self, image: np.ndarray, position: list[int]) -> Image.Image:
        """
        Crop the image based on the position and return the cropped image.
        """
        pass