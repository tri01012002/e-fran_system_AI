from raiki_sdk import DetectorBase, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo, normalize_position_to_square
# from rfdetr import RFDETRBase
from .config import DETECTOR_WEIGHTS, TARGET_SIZE, YOLO_CONF
from PIL import Image
import cv2
import numpy as np
import supervision as sv
from typing import Union, List

from ultralytics import YOLO


class CTRP02Detector(DetectorBase):

    def __init__(self, device: str):
        self.model = YOLO(DETECTOR_WEIGHTS)

        self.device = device
        self.yolo_conf = YOLO_CONF
        self.output_size = TARGET_SIZE
        self.img_preparing = None

    def detect(self, image):

        print("Image shape:", image.shape)

        results = self.model.predict(
            source=image,
            conf=0.1,
            verbose=True
        )

        print(results)

        detections = sv.Detections.from_ultralytics(results[0])

        print("Num detections:", len(detections))

        return detections

    def validate(self, results: sv.Detections) -> YoloValidation:

        if results is None or len(results.confidence) == 0:
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None,
                confidence=None
            )

        best_idx = int(np.argmax(results.confidence))

        if results.confidence[best_idx] < self.yolo_conf:
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None,
                confidence=None
            )

        best_box = results.xyxy[best_idx]

        class_id = int(results.class_id[best_idx])

        position = [int(v) for v in best_box]

        square_pos = normalize_position_to_square(position)

        x0, y0, x1, y1 = square_pos

        padding_px = 200

        padded_pos = [
            x0 - padding_px,
            y0 - padding_px,
            x1 + padding_px,
            y1 + padding_px
        ]

        return YoloValidation(
            is_valid=True,
            position=padded_pos,
            class_id=class_id,
            confidence=float(results.confidence[best_idx])
        )

    def crop(self, image, position, class_id=None) -> Image.Image:
        """
        Crop box from image
        """
        # image argument is not used here as we use self.img_preparing stored in detect
        # but to be safe and follow interface, we use image if self.img_preparing is None
        input_img = self.img_preparing if self.img_preparing is not None else image
        
        try:
            crop_img = padding_crop_yolo(input_img, position)
            if crop_img is None or crop_img.size == 0:
                raise ValueError("Empty crop")
            # Convert to PIL
            return Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            # Fallback in case of invalid crop
            return Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image, is_crop: bool = True) -> YoloValidation:
        """Run full detection pipeline"""
        return super().run_pipeline(image, is_crop)