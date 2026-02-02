from raiki_sdk import DetectorBase, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo, normalize_position_to_square
from PIL import Image
import cv2
import numpy as np
from abc import ABC, abstractmethod
from .config import DETECTOR_WEIGHTS, COLOR_PADDING
from rfdetr import RFDETRBase
import supervision as sv
from typing import Union, List


class OMGP02Detector(DetectorBase):
    """
    RFDETR detector for Omega P02 part detection
    """

    def __init__(self, device: str):
        self.model = RFDETRBase(num_classes=3, pretrain_weights=DETECTOR_WEIGHTS)

    def detect(self, image) -> Union[sv.Detections, List[sv.Detections]]:
        """
        Detect Omg P02 parts in the image

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            Union[sv.Detections, List[sv.Detections]]: Detection results
        """
        # max_hw = max(image.shape[:2])
        # self.img_rsed_square = resize_pwd(
        #     image,
        #     output_size=max_hw,
        #     color_padding=COLOR_PADDING
        # )
        self.img_rsed_square = image
        pil_image = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB))
        results = self.model.predict(pil_image)
        return results

    def validate(self, results: Union[sv.Detections, List[sv.Detections]]) -> YoloValidation:

        if len(results.confidence) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        best_class_id = None

        for i in range(len(results.confidence)):
            if results.confidence[i] > max_confidence:
                max_confidence = results.confidence[i]
                best_class_id = results.class_id[i]
                best_result = results.xyxy[i]

        # check validate
        valid = False
        position = None
        if best_result is not None:
            valid = True
            position = [int(value) for value in best_result]
        
        square_position = normalize_position_to_square(position)
        x0, y0, x1, y1 = square_position
        x0_p, y0_p, x1_p, y1_p = x0 - 200, y0 - 200, x1 + 200, y1 + 200
        square_position_padded = [x0_p, y0_p, x1_p, y1_p]

        return YoloValidation(is_valid=valid, position=square_position_padded, class_id=best_class_id)


    def crop(self, image, position) -> Image.Image:
        """
        Crop the image based on the position

        Args:
            image: np.ndarray (cv2) BGR image
            position: The position of the detection

        Returns:
            image: Image.Image (PIL) cropped image
        """
        cropped_image = padding_crop_yolo(self.img_rsed_square, position)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)