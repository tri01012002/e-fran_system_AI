"""
YOLO Detector for Louis Vuitton P02

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2


class LVP02Detector(DetectorBase):
    """
    YOLO detector for Louis Vuitton P02 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.img_rsed_square = None

    def detect(self, image) -> YoloResult:
        """
        Detect Louis Vuitton P02 parts in the image

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

    def validate(self, results) -> YoloValidation:
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
        best_conf = 0.0
        position = None
        valid = False
        # Process detected boxes
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()

            # Filter for valid classes and confidence
            if cls_id in [0, 2, 3]:
                valid = True
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # If frame is given, filter box

                # Update best box
                if conf > best_conf:
                    best_conf = conf
                    position = [x_min, y_min, x_max, y_max]

        return YoloValidation(is_valid=valid, position=position, class_id=None)

    def crop(self, image, position) -> Image.Image:
        """
        Crop the image based on the position

        Args:
            image: np.ndarray (cv2) BGR image
            position: The position of the detection

        Returns:
            image: Image.Image (PIL) cropped image
        """
        cropped_image = padding_crop_yolo(self.img_rsed_square, position, p=0, remove_padding=True)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

