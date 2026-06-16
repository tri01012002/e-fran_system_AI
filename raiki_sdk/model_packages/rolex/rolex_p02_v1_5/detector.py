"""
YOLO Detector for Rolex P02

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, COLOR_PADDING
from PIL import Image
import cv2
import numpy as np


class RolexP02Detector(DetectorBase):
    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.img_rsed_square = None

    def detect(self, image) -> YoloResult:
        """
        Detect object using YOLO
        """

        # resize + padding
        self.img_rsed_square = resize_pwd(
            image,
            output_size=self.yolo_imgsz,
            color_padding=COLOR_PADDING
        )

        pil_img = Image.fromarray(
            cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB)
        )

        results = self.model.predict(
            source=pil_img,
            verbose=False,
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz
        )

        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:
        """
        Validate detection results
        """

        if results.boxes is None or len(results.boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None, confidence=None)

        best_box = None
        best_conf = -1
        best_cls = None

        for box in results.boxes:
            cls_id = int(box.cls)
            conf = float(box.conf.item())

            # lọc class + conf
            if cls_id in [0, 1] and conf >= self.yolo_conf:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])

                if conf > best_conf:
                    best_conf = conf
                    best_box = [x_min, y_min, x_max, y_max]
                    best_cls = cls_id

        if best_box is None:
            return YoloValidation(is_valid=False, position=None, class_id=None, confidence=None)

        return YoloValidation(
            is_valid=True,
            position=best_box,
            class_id=best_cls,
            confidence=best_conf
        )

    def crop(self, image, position) -> Image.Image:
        """
        Crop from resized image (IMPORTANT)
        """

        if position is None:
            return None

        x_min, y_min, x_max, y_max = position

        h, w = self.img_rsed_square.shape[:2]

        # clamp
        x_min = max(0, int(x_min))
        y_min = max(0, int(y_min))
        x_max = min(w, int(x_max))
        y_max = min(h, int(y_max))

        if x_max <= x_min or y_max <= y_min:
            return None

        cropped = self.img_rsed_square[y_min:y_max, x_min:x_max]

        if cropped is None or cropped.size == 0:
            return None

        cropped_pil = Image.fromarray(
            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        )

        return cropped_pil

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        """
        Full pipeline: detect -> validate -> crop
        """
        return super().run_pipeline(image, is_crop)

