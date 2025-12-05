from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from .schemas import YoloResult, YoloValidation
from typing import Union, List
import supervision as sv


class DetectorBase(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> Union[sv.Detections, List[sv.Detections]] | YoloResult:
        """
        Detect parts in the image
        """
        pass
    
    @abstractmethod
    def validate(self, results: Union[sv.Detections, List[sv.Detections]] | YoloResult) -> YoloValidation:
        """
        Validate the results
        """
        pass

    @abstractmethod
    def crop(self, image: np.ndarray, position: List[int]) -> Image.Image:
        """
        Crop the image based on the position
        """
        pass

    @abstractmethod
    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        """
        Run the pipeline

        Args:
            image: np.ndarray (cv2) BGR image
            is_crop: bool, whether to crop the image

        Returns:
            validation: Validation, validation results
        """

        results = self.detect(image)
        validation = self.validate(results)

        if is_crop and validation.is_valid:
            cropped_image = self.crop(image, validation.position)
            validation.image_crop = cropped_image

        return validation