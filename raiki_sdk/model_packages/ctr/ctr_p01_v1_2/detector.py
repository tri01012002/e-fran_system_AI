from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import padding_crop_yolo, resize_pwd
from ultralytics import YOLO
from .config import (
    YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, 
    TARGET_SIZE,COLOR_PADDING
)
from PIL import Image
import cv2
import numpy as np


class CTRP01Detector(DetectorBase):
    """
    YOLO detector for Omega P01
    Classes: 0=omega, 1=seamaster, 2=speedmaster
    Supports single and multi-box detection with combined box strategy
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.output_size = TARGET_SIZE
        self.img_preparing_yolo = None
        self.class_names = self.model.names

    def detect(self, image) -> YoloResult:
        """
        Detect objects in image
        """
        # max_hw = max(image.shape[:2])
        # self.img_preparing_yolo = resize_pwd(
        #     image, 
        #     output_size=max_hw, 
        #     color_padding=COLOR_PADDING
        # )
        self.img_preparing_yolo = image
        pil_image = Image.fromarray(
            cv2.cvtColor(self.img_preparing_yolo, cv2.COLOR_BGR2RGB)
        )
        
        results = self.model.predict(
            source=pil_image, 
            verbose=False, 
            conf=self.yolo_conf,
            imgsz=self.yolo_imgsz
        )
        return results[0]

    def _get_best_box_per_class(self, boxes, classes, confs, image_w: int, image_h: int) -> dict:
        """
        Get the best box for each class (highest confidence)
        Logic from crop.py
        """
        boxes_by_class = {}
        
        for i, class_id in enumerate(classes):
            class_id = int(class_id)
            conf = float(confs[i])
            
            x1n, y1n, x2n, y2n = boxes[i]
            box = [
                int(x1n * image_w),
                int(y1n * image_h),
                int(x2n * image_w),
                int(y2n * image_h)
            ]
            
            if class_id not in boxes_by_class or conf > boxes_by_class[class_id]['conf']:
                boxes_by_class[class_id] = {
                    'box': box,
                    'conf': conf,
                    'y1': box[1],
                    'class_name': self.class_names.get(class_id, f'class_{class_id}')
                }
        
        return boxes_by_class

    def validate(self, results: YoloResult) -> YoloValidation:
        """
        Validate detection results
        """
        boxes = results.boxes
        
        if boxes is None or len(boxes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        boxesn = boxes.xyxyn.cpu().numpy()
        classes = boxes.cls.cpu().int().numpy()
        confs = boxes.conf.cpu().numpy()
        
        if len(classes) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        image_h, image_w = map(int, boxes.orig_shape)
        boxes_by_class = self._get_best_box_per_class(boxesn, classes, confs, image_w, image_h)
        
        # Case 1: Single box
        if len(boxes_by_class) == 1:
            class_id = list(boxes_by_class.keys())[0]
            x1, y1, x2, y2 = boxes_by_class[class_id]['box']
            position = [x1, y1, x2, y2]
            return YoloValidation(is_valid=True, position=position, class_id=class_id)
        
        # Case 2: Multi boxes - create combined box
        sorted_boxes = sorted(boxes_by_class.items(), key=lambda x: x[1]['y1'])
        
        top_class_id, top_info = sorted_boxes[0]
        bottom_class_id, bottom_info = sorted_boxes[1]
        
        top_box = top_info['box']
        bottom_box = bottom_info['box']
        
        combined_box = [
            min(top_box[0], bottom_box[0]),
            min(top_box[1], bottom_box[1]),
            max(top_box[2], bottom_box[2]),
            max(top_box[3], bottom_box[3])
        ]
        
        return YoloValidation(is_valid=True, position=combined_box, class_id=999)

    def crop(self, image, position, class_id=None) -> Image.Image:
        """
        Crop box from image
        
        Args:
            image: np.ndarray (not used, uses self.img_preparing_yolo)
            position: [x1, y1, x2, y2]
            class_id: Detection class ID
            
        Returns:
            PIL Image - raw cropped image
        """
        x1, y1, x2, y2 = position
        cropped_original = self.img_preparing_yolo[y1:y2, x1:x2]
        
        # Convert to PIL
        return Image.fromarray(cv2.cvtColor(cropped_original, cv2.COLOR_BGR2RGB))


    def run_pipeline(self, image, is_crop: bool = True) -> YoloValidation:
        """Run full detection pipeline"""
        return super().run_pipeline(image, is_crop)