"""
YOLO Detector for Rolex P02

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import padding_crop_yolo, resize_pwd
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, TARGET_SIZE, COLOR_PADDING
from PIL import Image
import cv2


class RolexP02Detector(DetectorBase):
    """
    YOLO detector for Rolex P02 part detection
    """
    
    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
    
    def detect(self, image) -> YoloResult:
        """
        Detect Rolex P02 parts in the image
        - Convert the image to grayscale for YOLO detection.
        - Perform YOLO detection.
        - Return yolo results.
        
        Args:
            image: np.ndarray (cv2) BGR image 
            
        Returns:
            YoloResult: Detection results
        """

        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        results = self.model.predict(source=image, verbose=False, show=False, save=False, save_crop=False, 
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
        boxesn = results.boxes.xyxyn.cpu().numpy()  # Bounding boxesn
        classes = results.boxes.cls.cpu().numpy()  # Class IDs

        # filter bounding box of class 0 (rolex_p02) and class 1 (rolex_text)
        bbox_class_rolex_p02 = [boxesn[i] for i in range(len(classes)) if classes[i] == 0]
        bbox_class_rolex_text = [boxesn[i] for i in range(len(classes)) if classes[i] == 1]
        # check validate
        valid = False
        position = None
        
        p02_boxesn = []
        if bbox_class_rolex_p02 and bbox_class_rolex_text:
            for bbox1 in bbox_class_rolex_text:
                x_min1, y_min1, x_max1, y_max1 = bbox1
                for bbox2 in bbox_class_rolex_p02:
                    x_min2, y_min2, x_max2, y_max2 = bbox2
                    is_text_in = x_min1 >= x_min2 and y_min1 >= y_min2 and x_max1 <= x_max2 and y_max1 <= y_max2
                    if is_text_in:
                        p02_boxesn.append([x_min1, y_min2, x_max1, y_max1])

        if len(p02_boxesn) > 0:
            valid = True
            image_h, image_w = results.boxes.orig_shape
            x1n, y1n, x2n, y2n = p02_boxesn[0]
            x1, y1, x2, y2 = int(x1n * image_w), int(y1n * image_h), int(x2n * image_w), int(y2n * image_h)
            position = [x1, y1, x2, y2]

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
        cropped_image = padding_crop_yolo(image, position)
        cropped_image = resize_pwd(cropped_image, output_size=TARGET_SIZE, color_padding=COLOR_PADDING)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

