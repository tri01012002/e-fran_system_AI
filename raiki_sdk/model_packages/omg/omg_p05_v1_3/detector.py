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

MULTI_BOX_CLASS_ID = 999

class OMGP05Detector(DetectorBase):
    """
    YOLO detector for Omega P05 back cover detection
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
        self.class_names = self.model.names

    def detect(self, image) -> YoloResult:
        """
        Detect back cover in image
        
        Args:
            image: np.ndarray (cv2) BGR image
            
        Returns:
            YoloResult: Detection results
        """
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

    def _get_best_box_per_class(
        self, 
        boxes: np.ndarray, 
        classes: np.ndarray, 
        confs: np.ndarray, 
        image_w: int, 
        image_h: int
    ) -> dict:
        """
        Get best box per class based on confidence
        
        Args:
            boxes: Detected boxes (normalized)
            classes: Detected class IDs
            confs: Detected confidences
            image_w: Original image width
            image_h: Original image height
            
        Returns:
            Dict of best box per class with pixel coordinates and confidence
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
        # Select detection with highest confidence
        image_h, image_w = map(int, boxes.orig_shape)
        boxes_by_class = self._get_best_box_per_class(boxesn, classes, confs, image_w, image_h)

        if len(boxes_by_class) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)

        # Nhiều box → trả về dict cho từng box, có thể vẫn giữ combined
        all_boxes = [info['box'] for info in boxes_by_class.values()]
        combined_x1 = min(b[0] for b in all_boxes)
        combined_y1 = min(b[1] for b in all_boxes)
        combined_x2 = max(b[2] for b in all_boxes)
        combined_y2 = max(b[3] for b in all_boxes)
        combined_position = [combined_x1, combined_y1, combined_x2, combined_y2]

        if len(boxes_by_class) == 1:
            class_id = list(boxes_by_class.keys())[0]
            return YoloValidation(is_valid=True, position=combined_position, class_id=class_id)

        # Multi-box: position = combined (cho RedFrame), class_id = MULTI_BOX_CLASS_ID
        boxes_dict = {
            class_id: {"box": box_info['box']}
            for class_id, box_info in boxes_by_class.items()
        }
        return YoloValidation(
            is_valid=True,
            position=boxes_dict,       
            class_id=MULTI_BOX_CLASS_ID,
            # image_crop=boxes_dict             
        )

    def crop(self, image: np.ndarray, position, class_id: int = None) -> Image.Image:
        if position is None:
            return None

        if isinstance(position, dict):
            if len(position) == 0:
                return None
            return {
                cls_id: self._crop_single_box(*box_info['box'])
                for cls_id, box_info in position.items()
            }

        if not isinstance(position, (list, tuple)) or len(position) != 4:
            return None

        return self._crop_single_box(*position)

    def _crop_single_box(self, x1: int, y1: int, x2: int, y2: int) -> Image.Image:
        position = [x1, y1, x2, y2]
        cropped = padding_crop_yolo(self.img_preparing_yolo, position)
        max_hw = max(cropped.shape[:2])
        image_pwd = resize_pwd(cropped, output_size=max_hw, color_padding=COLOR_PADDING)
        return Image.fromarray(cv2.cvtColor(image_pwd, cv2.COLOR_BGR2RGB))
    
    def normalize_orientation(self, image: np.ndarray, class_id: int) -> np.ndarray:
        """
        Rotate image to normalize orientation to bottom
        
        Args:
            image: BGR image
            class_id: Detection class ID (2=down, 3=left, 4=right)
            
        Returns:
            Rotated BGR image normalized to bottom orientation
        """
        if image is None or class_id is None:
            return image
        
        config = self.rotation_config.get(class_id)
        if not config or config.get("rotate") is None:
            return image
        print("Rotate OMG P05: ",config)
        return cv2.rotate(image, config["rotate"])

    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
            return super().run_pipeline(image, is_crop)