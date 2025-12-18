"""
YOLO Detector for Rolex P01

Handles object detection logic using YOLO model.
"""

from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ
from PIL import Image
import cv2


class RolexP01Detector(DetectorBase):
    """
    YOLO detector for Rolex P01 part detection
    """
    
    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
    
    def detect(self, image) -> YoloResult:
        """
        Detect Rolex P01 parts in the image
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
        boxesn = results.boxes.xyxyn.cpu().numpy()  # Bounding boxesn
        classes = results.boxes.cls.cpu().tolist()  # Class IDs

        # check validate
        valid = False
        position = None

        # Modified: Check if both ROLEX (1) and TOP (2) are in the results
        have_rolex_and_top = 1 in classes and 2 in classes
        p01_boxesn = []

        if have_rolex_and_top:
            # Modified: Get indices for ROLEX and TOP
            rolex_index = classes.index(1)
            top_index = classes.index(2)

            # Modified: Extract bounding boxes for ROLEX and TOP
            box_rolex = boxesn[rolex_index]
            box_top = boxesn[top_index]

            # Modified: Calculate the combined bounding box
            top = box_top[1]
            bottom = box_rolex[3]
            left = min(box_rolex[0], box_top[0])
            right = max(box_rolex[2], box_top[2])
            p01_boxesn.append([left, top, right, bottom])

        if len(p01_boxesn) > 0:
            valid = True
            image_h, image_w = results.boxes.orig_shape
            x1n, y1n, x2n, y2n = p01_boxesn[0]
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
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)