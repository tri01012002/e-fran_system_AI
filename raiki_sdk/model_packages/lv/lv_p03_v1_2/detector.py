"""
YOLO Detector for Louis Vuitton P03

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS_1, YOLO_WEIGHTS_2, YOLO_CONF, YOLO_IMGSZ, PAD
from PIL import Image
import cv2
import logging

_logger = logging.getLogger(__name__)


class LVP03Detector(DetectorBase):
    """
    YOLO detector for Louis Vuitton P03 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS_1).to(device)
        self.model_v2 = YOLO(YOLO_WEIGHTS_2).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.img_rsed_square = None

    def detect(self, image) -> YoloResult:
        """
        Detect Louis Vuitton P03 parts in the image

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            YoloResult: Detection results
        """
        max_hw = max(image.shape[:2])
        self.img_rsed_square = resize_pwd(image, output_size=max_hw, color_padding=(255, 255, 255))

        pil_img_rsed_square = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2RGB))
        pil_img_rsed_square_gray = pil_img_rsed_square.convert("L")
        results_flip = self.model_v2.predict(
            source=pil_img_rsed_square_gray, verbose=False, 
            show=False, save=False, save_crop=False, 
            conf=self.yolo_conf, line_width=0, 
            show_labels=False, show_conf=False, 
            imgsz=self.yolo_imgsz)
        
        # 0: 'flip', 1: 'hook-logo', 2: 'louis', 3: 'lu', 4: 'lv_circle', 5: 'lv_logo_thick', 6: 'lv_logo_thin', 7: 'paris', 8: 'ton', 9: 'vui', 10: 'vuitton'
        if len(results_flip[0].boxes) > 0 and (results_flip[0].boxes.cls == 0).any():
            _logger.info("Image rotated 180 degrees")
            self.img_rsed_square = cv2.rotate(self.img_rsed_square, cv2.ROTATE_180)

        pil_image_gray  = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2GRAY))

        results = self.model.predict(source=pil_image_gray, verbose=False, show=False, save=False, save_crop=False, conf=self.yolo_conf, line_width=0, show_labels=False, show_conf=False, imgsz=self.yolo_imgsz)

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
        # Extract boxes and classes
        if len(results.boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)

        best_box = None
        best_conf = -1
        valid = False
        # Iterate over all detected boxes
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()

            # Check if class is in the allowed list and confidence is above threshold
            # 0: 'hex_crew_big', 1: 'hook-logo', 2: 'louis', 3: 'lu', 4: 'lv_circle', 5: 'lv_logo_thick', 6: 'lv_logo_thin', 7: 'paris', 8: 'ton', 9: 'vui', 10: 'vuitton'
            if cls_id in [1, 2, 4, 5, 6]:
                valid = True
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Keep the box with the highest confidence
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
        cropped_image = padding_crop_yolo(self.img_rsed_square, position, p=PAD, remove_padding=True)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

