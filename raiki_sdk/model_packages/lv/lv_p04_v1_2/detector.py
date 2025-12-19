"""
YOLO Detector for Louis Vuitton P04

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS_1, YOLO_WEIGHTS_2, YOLO_CONF, YOLO_IMGSZ, RESIZE_OUTPUT_SIZE, COLOR_PADDING
from PIL import Image
import cv2
import numpy as np
import io
import logging

_logger = logging.getLogger(__name__)

class LVP04Detector(DetectorBase):
    """
    YOLO detector for Louis Vuitton P04 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS_1).to(device)
        self.model_v2 = YOLO(YOLO_WEIGHTS_2).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ

    def detect(self, image) -> YoloResult:
        """
        Detect Louis Vuitton P04 parts in the image
        - Convert the image to grayscale for YOLO detection.
        - Perform YOLO detection.
        - Return yolo results.

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            YoloResult: Detection results
        """
        pil_image_gray = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).convert("L")
        detect_results = self.model.predict(
            source=pil_image_gray, verbose=False, show=False, save=False, 
            save_crop=False, conf=self.yolo_conf, line_width=0, 
            show_labels=False, show_conf=False, imgsz=self.yolo_imgsz) 

        results = self.model_v2.predict(
            source=pil_image_gray, verbose=False, show=False, 
            save=False, save_crop=False, conf=self.yolo_conf, line_width=0, 
            show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)

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
        if len(results.boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)

        # Extract boxes and classes
        best_box = None
        best_conf = -1
        valid = False
        # Iterate over detected boxes
        for box in results.boxes:
            valid = True
            conf = box.conf.item()

            # Get bounding box coordinates
            x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            # If a frame is defined, ensure the box is within it

            # Keep the highest confidence valid box
            if conf > best_conf:
                best_conf = conf
                best_box = [x_min, y_min, x_max, y_max]

        return YoloValidation(is_valid=valid, position=best_box, class_id=None)

    def crop(self, image, position) -> Image.Image:
        """
        Crop the image based on the position

        Args:
            image: np.ndarray (cv2) BGR image
            position: The position of the detection

        Returns:
            image: Image.Image (PIL) cropped image
        """
        cropped_image = padding_crop_yolo(image, position)
        pil_img_temp = Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
        buffer = io.BytesIO()
        pil_img_temp.save(buffer, format="JPEG")
        buffer.seek(0)
        file_bytes = np.asarray(bytearray(buffer.read()), dtype=np.uint8)
        img_padding_yolo = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        ### End convett cv2 -> PIL -> cv2

        self.image_detect = resize_pwd(img_padding_yolo, output_size=RESIZE_OUTPUT_SIZE, color_padding=COLOR_PADDING)
        return Image.fromarray(cv2.cvtColor(self.image_detect, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

