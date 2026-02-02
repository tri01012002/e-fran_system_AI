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


class OMGP01Detector(DetectorBase):
    """
    YOLO detector for Omega P01
    Classes: 0=omega, 1=seamaster, 2=speedmaster
    Hỗ trợ single và multi-box detection với combined box strategy
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
        Detect objects trong ảnh
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
        Lấy box tốt nhất cho mỗi class (confidence cao nhất)
        Logic từ crop.py
        """
        boxes_by_class = {}
        
        for i, class_id in enumerate(classes):
            class_id = int(class_id)
            conf = float(confs[i])
            
            # Chuyển normalized coords sang pixel coords
            x1n, y1n, x2n, y2n = boxes[i]
            box = [
                int(x1n * image_w),
                int(y1n * image_h),
                int(x2n * image_w),
                int(y2n * image_h)
            ]
            
            # Chỉ giữ box có confidence cao nhất cho mỗi class
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
        Validate 
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
        
        # Case 2: Multi boxes - tạo combined box (logic từ crop.py)
        sorted_boxes = sorted(boxes_by_class.items(), key=lambda x: x[1]['y1'])
        
        top_class_id, top_info = sorted_boxes[0]
        bottom_class_id, bottom_info = sorted_boxes[1]
        
        top_box = top_info['box']
        bottom_box = bottom_info['box']
        
        # Tạo combined box
        combined_box = [
            min(top_box[0], bottom_box[0]),
            min(top_box[1], bottom_box[1]),
            max(top_box[2], bottom_box[2]),
            max(top_box[3], bottom_box[3])
        ]
        
        # class_id = 999 để đánh dấu combined box
        return YoloValidation(is_valid=True, position=combined_box, class_id=999)

    def crop(self, image, position, class_id=None) -> Image.Image:
        """
        Crop box và expand thành hình vuông với PADDING
        
        Logic:
        1. Tính size vuông từ max(box_w, box_h)
        2. LUÔN LUÔN thêm padding để box thành vuông (không crop trên ảnh gốc)
        3. Crop box vuông từ ảnh đã padding
        
        Args:
            image: np.ndarray (not used, uses self.img_preparing_yolo)
            position: [x1, y1, x2, y2]
            class_id: Detection class ID
            
        Returns:
            PIL Image - square cropped image (luôn có padding nếu box không vuông)
        """
        x1, y1, x2, y2 = position
        box_w = x2 - x1
        box_h = y2 - y1
        
        # BƯỚC 1: Crop box GỐC trước (chữ nhật)
        cropped_original = self.img_preparing_yolo[y1:y2, x1:x2]
        
        # BƯỚC 2: Tính padding cần thiết để thành vuông
        size = max(box_w, box_h)
        
        if box_w == box_h:
            # Đã vuông rồi, không cần padding
            cropped_image = cropped_original
        else:
            # Cần padding để thành vuông
            if box_w > box_h:
                # Box ngang → padding trên/dưới
                pad_vertical = (size - box_h) // 2
                pad_top = pad_vertical
                pad_bottom = size - box_h - pad_vertical  # Đảm bảo tổng đúng
                pad_left = 0
                pad_right = 0
            else:
                # Box dọc → padding trái/phải
                pad_horizontal = (size - box_w) // 2
                pad_left = pad_horizontal
                pad_right = size - box_w - pad_horizontal  # Đảm bảo tổng đúng
                pad_top = 0
                pad_bottom = 0
            
            # Thêm padding trắng
            cropped_image = cv2.copyMakeBorder(
                cropped_original,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=(255, 255, 255)  # Padding trắng
            )
        
        # BƯỚC 3: Convert to PIL
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image, is_crop: bool = True) -> YoloValidation:
        """Run full detection pipeline"""
        return super().run_pipeline(image, is_crop)