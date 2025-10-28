from raiki_sdk.base import YoloBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo,resize_pwd
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2
import numpy as np
import torch


class RolexP04Detector(YoloBase):

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
        P10_boxesn = [box for box in yolo_results[0].boxes]

        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        for box in P10_boxesn:
            if box.conf > max_confidence:
                max_confidence = box.conf
                best_result = box
        # check validate
        valid = False
        position = {}
        if best_result:
            valid = True
            h, w = best_result.orig_shape
            x1, y1, x2, y2 = best_result.xyxy[0].cpu().to(torch.int)
            x1n, y1n, x2n, y2n = best_result.cpu().xyxyn[0]
            py_top = 0
            py_bot = 0

            pyn_top = 0
            pyn_bot = 0

            position = [x1, max(y1 - py_top, 0), x2, min(h, (y2 + py_bot))]


        return YoloValidation(is_valid=valid, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        img_padding_yolo = padding_crop_yolo(image, position)
        self.image_detect = img_padding_yolo
        return Image.fromarray(cv2.cvtColor(self.image_detect, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)