"""
YOLO Detector for Louis Vuitton P01

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2
import torch
import numpy as np
import logging

_logger = logging.getLogger(__name__)


class LVP01Detector(DetectorBase):
    """
    YOLO detector for Louis Vuitton P01 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.rotate_180 = False
        self.img_rsed_square = None

    def detect(self, image) -> YoloResult:
        """
        Detect Louis Vuitton P01 parts in the image

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            YoloResult: Detection results
        """
        max_hw = max(image.shape[:2])
        self.img_rsed_square = resize_pwd(image, output_size=max_hw, color_padding=(255, 255, 255))
        pil_image = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB))
        pil_image_gray = pil_image.convert("L")
        results = self.model.predict(source=pil_image_gray, verbose=False, show=False, save=False, save_crop=False,
                                        conf=self.yolo_conf, line_width=0, show_labels=False, show_conf=False,
                                        imgsz=self.yolo_imgsz)
        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:
        """
        Validate detection results

        Args:
            yolo_results: YOLO detection results

        Returns:
            is_valid: True if the detection is valid, False otherwise
            position: The position of the detection
            class_id: The class id of the detection
        """

        self.rotate_180 = False
        
        boxes = results.boxes
        if boxes is None:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        boxesn = boxes.xyxyn.cpu().numpy()
        classes = boxes.cls.cpu().int().numpy()
        confs = boxes.conf.cpu().numpy()
        
        # Check if no detections found
        if len(classes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)  
        
        # {0: 'CITY', 1: 'FULL', 2: 'LOUIS', 3: 'R', 4: 'VUITTON',
        #  5: 'CITY_r180', 6: 'FULL_r180', 7: 'LOUIS_r180', 8: 'R_r180', 9: 'VUITTON_r180'}
        
        # Find FULL (1) or FULL_r180 (6) to crop image for CMAL model
        boxs_valid = [(i, boxesn[i], confs[i]) for i in range(len(classes)) if classes[i] == 1]
        
        if len(boxs_valid) == 0:
            # Not found FULL, try to find FULL_r180
            boxs_valid = [(i, boxesn[i], confs[i]) for i in range(len(classes)) if classes[i] == 6]
            if len(boxs_valid) > 0:
                _logger.info("BagLVP01 detected FULL_r180")
                self.rotate_180 = True
        
        if len(boxs_valid) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        # Select the box with the highest confidence
        best_box = max(boxs_valid, key=lambda x: x[2])  # (index, box, conf)
        _, box_coords, _ = best_box
        x1n, y1n, x2n, y2n = box_coords
        
        
        # Convert normalized coords to pixel coords
        image_h, image_w = map(int, boxes.orig_shape)
        x1, y1, x2, y2 = int(x1n * image_w), int(y1n * image_h), int(x2n * image_w), int(y2n * image_h)
        
        position = [x1, y1, x2, y2]

        print(f"Rotate 180: {self.rotate_180}")
        
        return YoloValidation(is_valid=True, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        """
        Crop the image based on the position

        Args:
            image: np.ndarray (cv2) BGR image
            position: The position of the detection

        Returns:
            image: Image.Image (PIL) cropped image
        """
        cropped_image = padding_crop_yolo(self.img_rsed_square, position, rotate_180=self.rotate_180)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

