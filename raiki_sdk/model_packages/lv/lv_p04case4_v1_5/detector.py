"""
YOLO Detector for Rolex P04

Handles object detection logic using YOLO model.
"""

from raiki_sdk import DetectorBase, YoloResult, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo
from ultralytics import YOLO
from .config import YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ, COLOR_PADDING
from PIL import Image
import cv2
import numpy as np
import io

class LVP04Case4Detector(DetectorBase):
    """
    YOLO detector for Louis Vuitton P04 Case 4 part detection
    """

    def __init__(self, device: str):
        self.model = YOLO(YOLO_WEIGHTS).to(device)
        self.yolo_conf = YOLO_CONF
        self.yolo_imgsz = YOLO_IMGSZ

    def detect(self, image) -> YoloResult:
        """
        Detect Louis Vuitton P04 Case 4 parts in the image

        Args:
            image: np.ndarray (cv2) BGR image

        Returns:
            YoloResult: Detection results
        """
        self.img_rsed_square = resize_pwd(image, output_size=YOLO_IMGSZ, color_padding=COLOR_PADDING)
        pil_img_rsed_square_gray = Image.fromarray(cv2.cvtColor(self.img_rsed_square, cv2.COLOR_BGR2GRAY))
        results = self.model.predict(source=pil_img_rsed_square_gray, verbose=False, show=False, save=False, save_crop=False,
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

        best_box = None
        best_conf = -1
        valid = False
        # Iterate over all detected boxes
        for box in results.boxes:
            cls_id = int(box.cls)
            conf = box.conf.item()
            valid = True
            # Check if class is in the allowed list and confidence is above threshold
            # 0: 'circle', 1: 'itt', 2: 'material'
            if cls_id == 2:
                x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

                # Keep the box with the highest confidence
                if conf > best_conf:
                    best_conf = conf
                    best_box = [x_min, y_min, x_max, y_max]

        return YoloValidation(is_valid=valid, position=best_box, class_id=None)

    def crop(self, image, position) -> Image.Image:
        h, w = image.shape[:2]

        crop_ratio = 0.6 

        crop_size = int(min(h, w) * crop_ratio)

        center_x = w // 2
        center_y = h // 2

        x_min = max(center_x - crop_size // 2, 0)
        y_min = max(center_y - crop_size // 2, 0)
        x_max = min(center_x + crop_size // 2, w)
        y_max = min(center_y + crop_size // 2, h)

        cropped = image[y_min:y_max, x_min:x_max]

        output_size = 256  
        processed = resize_pwd(
            cropped,
            output_size=output_size,
            color_padding=COLOR_PADDING
        )

        processed_pil = Image.fromarray(
            cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        )
        return processed_pil

    def run_pipeline(self, image:  any, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)

if __name__ == "__main__":
    from raiki_sdk import load_image
    import matplotlib.pyplot as plt

    detector = LVP04Case4Detector(device="cpu")
    image = load_image(r"\\192.168.10.19\DataHandle\PERSONAL_DATA\QUANG\LV\LV_P04\lv_p04_case4_data\lv_p04_case4_data-padding\data\data-training\real\LV_P04_real_material_case4_1771632707092_2025-02-07_train_frames")
    validation_result = detector.run_pipeline(image)

    print(validation_result)

    if validation_result.is_valid:
        plt.imshow(detector.image_detect)
        plt.axis('off')
        plt.show()
