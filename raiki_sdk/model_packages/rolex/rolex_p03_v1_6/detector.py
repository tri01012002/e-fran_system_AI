"""
Detector for Rolex P03.

Standalone version.
No raiki_sdk dependency.
"""
from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import padding_crop_yolo
from dataclasses import dataclass
from typing import Optional

from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

from .config import (
    YOLO_WEIGHTS,
    YOLO_CONF,
    YOLO_IMGSZ,
    COLOR_PADDING,
    RESIZE_OUTPUT_SIZE,
)


@dataclass
class YoloValidation:
    is_valid: bool
    position: Optional[list]
    class_id: Optional[int]
    confidence: Optional[float]


class RolexP03Detector:

    def __init__(self, device: str = "cpu"):
        self.model = YOLO(YOLO_WEIGHTS).to(device)

        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ

    def detect(self, image):
        """
        Run YOLO detection.
        """

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img_pil = Image.fromarray(img_gray)

        results = self.model.predict(
            source=img_pil,
            verbose=False,
            show=False,
            save=False,
            save_crop=False,
            conf=self.yolo_conf,
            line_width=0,
            show_labels=False,
            show_conf=False,
            imgsz=self.yolo_imgsz,
        )

        return results[0]

    def validate(self, results) -> YoloValidation:
        """
        Validate YOLO detection.

        Class:
            0 = inner
            1 = outer
        """

        if results.boxes is None or len(results.boxes) == 0:
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None,
                confidence=None,
            )

        boxesn = results.boxes.xyxyn.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        outer_conf = None

        for i in range(len(classes)):
            if classes[i] == 1:
                outer_conf = float(confs[i])

                if outer_conf < 0.8:
                    return YoloValidation(
                        is_valid=False,
                        position=None,
                        class_id=None,
                        confidence=outer_conf,
                    )

        bbox_class_inner = [
            boxesn[i]
            for i in range(len(classes))
            if classes[i] == 0
        ]

        bbox_class_outer = [
            boxesn[i]
            for i in range(len(classes))
            if classes[i] == 1
        ]

        p03_boxesn = []

        if bbox_class_inner and bbox_class_outer:

            for bbox1 in bbox_class_inner:

                x_min1, y_min1, x_max1, y_max1 = bbox1

                for bbox2 in bbox_class_outer:

                    x_min2, y_min2, x_max2, y_max2 = bbox2

                    if (
                        x_min1 >= x_min2
                        and y_min1 >= y_min2
                        and x_max1 <= x_max2
                        and y_max1 <= y_max2
                    ):
                        p03_boxesn.append([
                            max(0, x_min2),
                            max(0, y_min2),
                            min(x_max2, 1),
                            min(y_max2, 1),
                        ])

        if len(p03_boxesn) == 0:
            return YoloValidation(
                is_valid=False,
                position=None,
                class_id=None,
                confidence=outer_conf,
            )

        image_h, image_w = results.boxes.orig_shape

        x1n, y1n, x2n, y2n = p03_boxesn[0]

        position = [
            int(x1n * image_w),
            int(y1n * image_h),
            int(x2n * image_w),
            int(y2n * image_h),
        ]

        return YoloValidation(
            is_valid=True,
            position=position,
            class_id=None,
            confidence=outer_conf,
        )

    def crop(self, image, position):
        """
        Crop detected Rolex P03 region.
        """

        x1, y1, x2, y2 = position

        cropped = image[y1:y2, x1:x2]

        if cropped.size == 0:
            return None

        cropped = cv2.resize(
            cropped,
            (RESIZE_OUTPUT_SIZE, RESIZE_OUTPUT_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )

        return Image.fromarray(
            cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        )

    def run(self, image):
        """
        Complete detector pipeline.
        """

        results = self.detect(image)

        validation = self.validate(results)

        if not validation.is_valid:
            return validation, None

        cropped_image = self.crop(
            image,
            validation.position,
        )

        return validation, cropped_image