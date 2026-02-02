from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo, resize_pwd
from ultralytics import YOLO
from .config import (
    YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, 
    TARGET_SIZE, COLOR_PADDING, ROTATION_CONFIG
)
from PIL import Image
import cv2
import numpy as np


class OMGP06Detector(DetectorBase):
    """
    YOLO detector for Omega P06 back cover detection
    Detects 4 orientations and normalizes to bottom orientation
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.output_size = TARGET_SIZE
        self.color_padding = COLOR_PADDING
        self.rotation_config = ROTATION_CONFIG
        self.img_preparing_yolo = None
        self.last_class_id = None  

    def detect(self, image) -> YoloResult:
        """
        Detect back cover in image
        
        Args:
            image: np.ndarray (cv2) BGR image
            
        Returns:
            YoloResult: Detection results
        """
        # max_hw = max(image.shape[:2])
        # self.img_preparing_yolo = resize_pwd(
        #     image, 
        #     output_size=max_hw, 
        #     color_padding=(255, 255, 255)
        # )
        self.img_preparing_yolo = image
        pil_image = Image.fromarray(
            cv2.cvtColor(self.img_preparing_yolo, cv2.COLOR_BGR2RGB)
        )
        
        results = self.model.predict(
            source=pil_image, 
            verbose=False, 
            show=False, 
            save=False,
            save_crop=False, 
            conf=self.yolo_conf, 
            line_width=0,
            show_labels=False, 
            show_conf=False,
            imgsz=self.yolo_imgsz
        )
        
        return results[0]

    def validate(self, results: YoloResult) -> YoloValidation:
        """
        Validate detection results and select best detection
        
        Args:
            results: detection results
            
        Returns:
            YoloValidation with is_valid, position, class_id
        """
        
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        boxesn = boxes.xyxyn.cpu().numpy()
        classes = boxes.cls.cpu().int().numpy()
        confs = boxes.conf.cpu().numpy()
        
        if len(classes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        best_idx = np.argmax(confs)
        best_box = boxesn[best_idx]
        best_class = int(classes[best_idx])
        
        self.last_class_id = best_class
    
        image_h, image_w = map(int, boxes.orig_shape)
        x1n, y1n, x2n, y2n = best_box
        x1, y1, x2, y2 = (
            int(x1n * image_w), 
            int(y1n * image_h), 
            int(x2n * image_w), 
            int(y2n * image_h)
        )
        
        position = [x1, y1, x2, y2]
        
        return YoloValidation(is_valid=True, position=position, class_id=best_class)

    def crop(self, image, position, class_id=None) -> Image.Image:
        """
        Crop and normalize image to square, then rotate to bottom orientation
        
        Args:
            image: np.ndarray (cv2) BGR image (not used, uses self.img_preparing_yolo)
            position: [x1, y1, x2, y2]
            class_id: Detection class ID for rotation (optional)
        Returns:
            PIL Image cropped, rotated to bottom orientation
        """
        if class_id is None:
            class_id = self.last_class_id
        
        x1, y1, x2, y2 = position
        box_w = x2 - x1
        box_h = y2 - y1
        
        size = max(box_w, box_h)
        
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        new_x1 = center_x - size // 2
        new_y1 = center_y - size // 2
        new_x2 = new_x1 + size
        new_y2 = new_y1 + size
        
        h, w = self.img_preparing_yolo.shape[:2]
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(w, new_x2)
        new_y2 = min(h, new_y2)
        
        new_position = [new_x1, new_y1, new_x2, new_y2]
        
        cropped_image = padding_crop_yolo(
            self.img_preparing_yolo, 
            new_position
        )

        if class_id is not None:
            cropped_image = self.normalize_orientation(cropped_image, class_id)

        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    
    def normalize_orientation(self, image: np.ndarray, class_id: int) -> np.ndarray:
        """
        Rotate image to normalize orientation to bottom
        
        Args:
            image: BGR image
            class_id: Detection class ID (0=bottom, 1=left, 2=right, 3=top)
            
        Returns:
            Rotated BGR image normalized to bottom orientation
        """
        if image is None or class_id is None:
            return image
        
        config = self.rotation_config.get(class_id)
        if not config or config.get("rotate") is None:
            return image
        
        rotated = cv2.rotate(image, config["rotate"])
        return rotated

    def run_pipeline(self, image, is_crop: bool = True) -> YoloValidation:
        """Run full detection pipeline"""
        results = self.detect(image)
        validation = self.validate(results)


        if is_crop and validation.is_valid:
            cropped_image = self.crop(image, validation.position, validation.class_id)
            validation.image_crop = cropped_image
        return validation