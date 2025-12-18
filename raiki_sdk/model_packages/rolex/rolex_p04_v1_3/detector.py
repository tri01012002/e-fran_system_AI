from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo, resize_pwd
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, COLOR_PADDING, TARGET_SIZE
from PIL import Image
import cv2
import numpy as np
import torch


class RolexP04Detector(DetectorBase):

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.output_size = TARGET_SIZE
        self.color_padding = COLOR_PADDING

    def detect(self, image) -> YoloResult:
        img_input_yolo_cv2 = image
        img_input_yolo_pil = Image.fromarray(cv2.cvtColor(img_input_yolo_cv2, cv2.COLOR_BGR2GRAY))

        results = self.model.predict(source=img_input_yolo_pil, verbose=False, show=False, save=False,
                                     save_crop=False, conf=self.yolo_conf, line_width=0,
                                     show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)

        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:
        if len(results.boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        p04_boxesn = [box for box in results.boxes]

        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        for box in p04_boxesn:
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

        return YoloValidation(is_valid=valid, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        img_padding_yolo = padding_crop_yolo(image, position)
        img_padding_yolo = resize_pwd(img_padding_yolo, output_size=self.output_size, color_padding=self.color_padding)
        return Image.fromarray(cv2.cvtColor(img_padding_yolo, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)