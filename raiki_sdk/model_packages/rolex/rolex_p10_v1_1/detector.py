from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo, resize_pwd
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, TARGET_SIZE, COLOR_PADDING
from PIL import Image
import cv2
import numpy as np
import torch


class RolexP10Detector(DetectorBase):

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.image_detect = None
        self.iqa_image = None
        self.output_size = TARGET_SIZE
        self.color_padding = COLOR_PADDING

    def detect(self, image) -> YoloResult:
        max_hw = max(image.shape[:2])
        # self.img_rsed_square = resize_pwd(image, output_size=max_hw, color_padding=(255, 255, 255))
        self.img_rsed_square = image
        pil_img_rsed_square = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB))

        results = self.model.predict(source=pil_img_rsed_square, verbose=False, show=False, save=False,
                                     save_crop=False, conf=self.yolo_conf, line_width=0,
                                     show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)


        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:
        P10_boxesn = [box for box in results.boxes]

        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        for box in P10_boxesn:
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
        img_padding_yolo = padding_crop_yolo(self.img_rsed_square, position)
        self.image_detect = img_padding_yolo
        return Image.fromarray(cv2.cvtColor(self.image_detect, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)