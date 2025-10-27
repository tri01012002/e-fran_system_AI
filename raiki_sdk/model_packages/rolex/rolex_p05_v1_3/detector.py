from raiki_sdk.base import YoloBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2
import numpy as np
import torch


class RolexP05Detector(YoloBase):

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.position = {}
        self.image_detect = None
        self.iqa_image = None
        self.output_size = 224
        self.color_padding = (85, 85, 85)

    def yolo_detect(self, image) -> YoloResult:
        img_input_yolo_cv2 = image
        img_input_yolo_pil = Image.fromarray(cv2.cvtColor(img_input_yolo_cv2, cv2.COLOR_BGR2GRAY))

        results = self.model.predict(source=img_input_yolo_pil, verbose=False, show=False, save=False,
                                     save_crop=False, conf=self.yolo_conf, line_width=0,
                                     show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)

        return results[0]

    def validate(self, yolo_results: YoloResult) -> YoloValidation:
        p05_boxesn = list(yolo_results[0].boxes)

        max_confidence = 0
        best_result = None
        for box in p05_boxesn:
            conf = box.conf.item() if hasattr(box.conf, "item") else float(box.conf)
            if conf > max_confidence:
                max_confidence = conf
                best_result = box

        # ✅ Validation
        valid = best_result is not None
        position = {}

        if valid:
            position = best_result.xyxy.cpu().numpy()[0].astype(int)

        return YoloValidation(is_valid=valid, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        cropped_image = padding_crop_yolo(image, position)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)