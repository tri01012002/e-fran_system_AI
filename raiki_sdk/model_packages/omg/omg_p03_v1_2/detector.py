from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo, normalize_position_to_square
from ultralytics import YOLO
from PIL import Image
import cv2
import torch
import numpy as np
from .config import DETECTOR_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, COLOR_PADDING

class OMGP03Detector(DetectorBase):
    """
    YOLO detector for Omega P03 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(DETECTOR_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.img_preparing_yolo = None
        self.color_padding = COLOR_PADDING

    def detect(self, image) -> YoloResult:
        """
        Detect Omg P03 parts in the image

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            YoloResult: Detection results
        """
        # max_hw = max(image.shape[:2])
        # self.img_preparing_yolo = resize_pwd(image, output_size=max_hw, color_padding=self.color_padding)
        self.img_preparing_yolo = image
        pil_image = Image.fromarray(cv2.cvtColor(self.img_preparing_yolo, cv2.COLOR_BGR2RGB))
        results = self.model.predict(source= pil_image, verbose=False, show=False, save=False, save_crop=False,
                                        conf=self.yolo_conf, line_width=0, show_labels=False, show_conf=False,
                                        imgsz=self.yolo_imgsz)
        return results[0]

    def validate(self, yolo_results: YoloResult) -> YoloValidation:
        if len(yolo_results.boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)

        boxesn = [box for box in yolo_results.boxes]
        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        for box in boxesn:
            if box.conf > max_confidence:
                max_confidence = box.conf
                best_result = box
        # check validate
        valid = False
        position = None
        if best_result:
            valid = True
            x1, y1, x2, y2 = best_result.xyxy[0].cpu().to(torch.int)
            position = [x1, y1, x2, y2]
        square_position = normalize_position_to_square(position)

        return YoloValidation(is_valid=valid, position=square_position, class_id=None)



    def crop(self, image: np.ndarray = None, position: list = None) -> Image.Image:
        """
        Crop the image based on the position

        Args:
            image: np.ndarray (cv2) BGR image
            position: The position of the detection

        Returns:
            image: Image.Image (PIL) cropped image
        """
        cropped_image = padding_crop_yolo(self.img_preparing_yolo, position)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)