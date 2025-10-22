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

    @abstractmethod
    def validate(self, yolo_result: YoloResult) -> YoloValidation:
        """
        Validate the YOLO detection result and return the validation result.
        """
        pass

    @abstractmethod
    def crop(self, image: np.ndarray, position: list[int]) -> Image.Image:
        """
        Crop the image based on the position and return the cropped image.
        """
        pass

    @abstractmethod
    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        """
        Complete YOLO detection pipeline: detect and validate.
        """
        yolo_result = self.yolo_detect(image)
        yolo_validation = self.validate(yolo_result)

        if is_crop and yolo_validation.is_valid:
            cropped_image = self.crop(image, yolo_validation.position)
            yolo_validation.image_crop = cropped_image

        return yolo_validation