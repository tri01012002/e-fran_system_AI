"""
Detector for Omega P01 v1.2

YOLO detector for detecting watch components.
"""

from raiki_sdk.base import DetectorBase, YoloResult, YoloValidation
from raiki_sdk.preprocessing import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import (
    YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ,
    TARGET_SIZE, COLOR_PADDING
)

MULTI_BOX_CLASS_ID = 999
from PIL import Image
import cv2
import numpy as np


class OMGP01Detector(DetectorBase):
    """
    YOLO detector for Omega P01
    
    Classes: 0=omega, 1=seamaster, 2=speedmaster
    - Single class: crop box if valid
    - Multiple classes: return dict of cropped boxes
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ
        self.output_size = TARGET_SIZE
        self.img_preparing_yolo = None
        self.class_names = self.model.names

    def detect(self, image: np.ndarray) -> YoloResult:
        """
        Detect objects in image
        
        Args:
            image: Input image as numpy array (BGR)
            
        Returns:
            YoloResult: Detection results
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

    def _get_best_box_per_class(
        self, 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        confs: np.ndarray, 
        image_w: int, 
        image_h: int
    ) -> dict:
        """
        Get best box per class (highest confidence)
        
        Args:
            boxes: Normalized box coordinates [N, 4]
            classes: Class IDs [N]
            confs: Confidence scores [N]
            image_w: Image width
            image_h: Image height
            
        Returns:
            dict: {class_id: {'box': [x1,y1,x2,y2], 'conf': float, 'class_name': str}}
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
        
        Args:
            results: YOLO detection results
            
        Returns:
            YoloValidation: Validation result with position and class_id
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
        if len(boxes_by_class) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)

        # Single box case
        if len(boxes_by_class) == 1:
            class_id = list(boxes_by_class.keys())[0]
            x1, y1, x2, y2 = boxes_by_class[class_id]['box']
            return YoloValidation(is_valid=True, position=[x1, y1, x2, y2], class_id=class_id)

        # Multi boxes case
        boxes_dict = {
            class_id: {"box": box_info['box']}
            for class_id, box_info in boxes_by_class.items()
        }
        return YoloValidation(is_valid=True, position=boxes_dict, class_id=MULTI_BOX_CLASS_ID)

    def crop(self, image: np.ndarray, position, class_id: int = None) -> Image.Image:
        """
        Crop box and expand to square with padding
        
        Args:
            image: Input image (unused, kept for base class compatibility)
            position: [x1, y1, x2, y2] or dict of boxes
            class_id: Detection class ID
            
        Returns:
            PIL Image or dict of PIL Images
        """
        if position is None:
            return None

        # Multi box case
        if isinstance(position, dict):
            if len(position) == 0:
                return None
            return {
                cls_id: self._crop_single_box(*box_info['box'])
                for cls_id, box_info in position.items()
            }

        # Single box case
        if not isinstance(position, (list, tuple)) or len(position) != 4:
            return None

        return self._crop_single_box(*position)

    def _crop_single_box(self, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
        """Crop a single box and pad to square"""
        position = [x1, y1, x2, y2]
        cropped = padding_crop_yolo(self.img_preparing_yolo, position)
        max_hw = max(cropped.shape[:2])
        image_pwd = resize_pwd(cropped, output_size=max_hw, color_padding=COLOR_PADDING)
        return Image.fromarray(cv2.cvtColor(image_pwd, cv2.COLOR_BGR2RGB))
    
    
    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        """Run full detection pipeline"""
        return super().run_pipeline(image, is_crop)
