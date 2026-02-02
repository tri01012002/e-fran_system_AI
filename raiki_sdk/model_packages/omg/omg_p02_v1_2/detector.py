from raiki_sdk import DetectorBase, YoloValidation
from raiki_sdk import resize_pwd, padding_crop_yolo, normalize_position_to_square
from PIL import Image
import cv2
import numpy as np
from abc import ABC, abstractmethod
from .config import DETECTOR_WEIGHTS, COLOR_PADDING, NEEDLE_ALIGNER_WEIGHTS,YOLO_NEEDLE_ALIGNER_CONF
from ultralytics import YOLO
from rfdetr import RFDETRBase
import supervision as sv
from typing import Union, List
import math
import logging

logger = logging.getLogger(__name__)

class OMGP02Detector(DetectorBase):
    """
    RFDETR detector for Omega P02 part detection
    """

    def __init__(self, device: str):
        self.model = RFDETRBase(num_classes=3, pretrain_weights=DETECTOR_WEIGHTS)
        self.model_needle_aligner = YOLO(NEEDLE_ALIGNER_WEIGHTS).to(device)
        self.resized_image = None

    @staticmethod
    def calculate_rotation_angle(current_angle, target_angle=-90):
        """
        Calculate rotation angle to align vector from current_angle to target_angle.
        Logic: Keep unchanged as per user requirements.
        """
        angle_diff = target_angle - current_angle
        
        # Normalize to [-180, 180] to select shortest rotation direction
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        
        return -angle_diff

    @staticmethod
    def rotate_image(image, angle):
        """Rotate image around center with given angle (degrees). Positive = CCW."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(
            image, rot_mat, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)  
        )

    def needle_aligner(self, img):
        try:
            results = self.model_needle_aligner.predict(img, conf=YOLO_NEEDLE_ALIGNER_CONF, verbose=False)
            # Check if results are empty
            if not results or len(results) == 0:
                return img
            
            result = results[0]
            
            # Check if keypoints exist and have data
            if (result.keypoints is None or 
                not hasattr(result.keypoints, 'xy') or 
                result.keypoints.xy is None or
                result.keypoints.xy.shape[0] == 0 or
                not result.keypoints.has_visible):
                return img
            
            kpts = result.keypoints.xy[0].cpu().numpy()
            
            # Check if there are at least 3 keypoints
            if len(kpts) < 3:
                return img

            # Extract keypoints
            tip = kpts[0].astype(int)      # Tip
            left = kpts[1].astype(int)     # Left base
            right = kpts[2].astype(int)    # Right base
            
            # Calculate current angle of needle
            base_cx = (left[0] + right[0]) // 2
            base_cy = (left[1] + right[1]) // 2
            dx = tip[0] - base_cx
            dy = tip[1] - base_cy
            current_angle = math.degrees(math.atan2(dy, dx))
            
            # Calculate required rotation angle
            angle_to_rotate = self.calculate_rotation_angle(current_angle)
            
            # Rotate image
            return self.rotate_image(img, angle_to_rotate)
        
        except (IndexError, AttributeError, RuntimeError) as e:
            logger.warning(f"Needle aligner error: {str(e)}, returning original image")
            return img

    def detect(self, image) -> Union[sv.Detections, List[sv.Detections]]:
        """
        Detect Omg P02 parts in the image
        """
        aligned_image = self.needle_aligner(image)
        
        # max_hw = max(aligned_image.shape[:2])
        # self.resized_image = resize_pwd(aligned_image, output_size=max_hw, color_padding=COLOR_PADDING)
        self.resized_image = aligned_image
        pil_image = Image.fromarray(cv2.cvtColor(self.resized_image, cv2.COLOR_BGR2RGB))
        results = self.model.predict(pil_image)
        return results

    def validate(self, results: Union[sv.Detections, List[sv.Detections]]) -> YoloValidation:

        if len(results.confidence) == 0:
            return YoloValidation(is_valid=False, position=None, class_id=None)
        
        # Select the object with the highest probability.
        max_confidence = 0
        best_result = None
        best_class_id = None

        for i in range(len(results.confidence)):
            if results.confidence[i] > max_confidence:
                max_confidence = results.confidence[i]
                best_class_id = results.class_id[i]
                best_result = results.xyxy[i]

        # check validate
        valid = False
        position = None
        if best_result is not None:
            valid = True
            position = [int(value) for value in best_result]
        
        square_position = normalize_position_to_square(position)
        x0, y0, x1, y1 = square_position
        x0_p, y0_p, x1_p, y1_p = x0 - 200, y0 - 200, x1 + 200, y1 + 200
        square_position_padded = [x0_p, y0_p, x1_p, y1_p]

        return YoloValidation(is_valid=valid, position=square_position_padded, class_id=best_class_id)


    def crop(self, image, position) -> Image.Image:
        """
        Crop the image based on the position
        """
        # Crop from self.resized_image (already rotated in detect)
        cropped_image = padding_crop_yolo(self.resized_image, position)
        return Image.fromarray(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))

    def run_pipeline(self, image: np.ndarray, is_crop: bool = True) -> YoloValidation:
        return super().run_pipeline(image, is_crop)