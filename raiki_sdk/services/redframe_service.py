import joblib
from .config import get_redframe_model_path
import numpy as np
import xgboost as xgb
import pandas as pd
from raiki_sdk.base import RedFrameResult
from .iqa_service import LIQEServices, LIQECustomServices
from typing import Callable, Optional
from dataclasses import dataclass
from PIL import Image

class RedFrameService:
    def __init__(self, category: str, part: str, class_labels: list, advanced_mode: bool, device: str = "cuda", threshold: float = 0.1):
        self.category = category
        self.part = part
        self.model_path = get_redframe_model_path(self.category.lower(), self.part.lower())
        self.model_redframe = joblib.load(self.model_path)
        self.class_labels = class_labels
        self.advanced_mode = advanced_mode
        self.iqa_service = LIQEServices(device=device, threshold=threshold) if not advanced_mode else LIQECustomServices(device=device, threshold=threshold)

    def calculate_area(self, image: np.ndarray, frame):
        if isinstance(frame, list):
            x1_f = int(frame[0])
            y1_f = int(frame[1])
            x2_f = int(frame[2])
            y2_f = int(frame[3])
        else:
            x1_f, y1_f, x2_f, y2_f = frame.split(",")
            x1_f = int(float(x1_f) * image.shape[1])
            y1_f = int(float(y1_f) * image.shape[0])
            x2_f = int(float(x2_f) * image.shape[1])
            y2_f = int(float(y2_f) * image.shape[0])
        return (x2_f - x1_f) * (y2_f - y1_f)

    def calculate_area_ratio(self, image: np.ndarray, frame: str = None, position: list = None):
        area_frame = self.calculate_area(image, frame)
        area_position = self.calculate_area(image, position)

        return float(area_position / area_frame)

    def forward(
        self, 
        image: np.ndarray, 
        frame: str | None = None, 
        position: list | None = None,
        crop_method: Optional[Callable[[np.ndarray, list], Image.Image]] = None
    ) -> RedFrameResult:

        if not self.advanced_mode:
            iqa_result = self.iqa_service.iqa_detect(image)
            if iqa_result.image_quality_validate == "fail":
                return RedFrameResult(result="failed", image_quality_validate="fail", iqa_score=iqa_result.iqa_score, distortion_type=None, error_type=None)
            else:
                return RedFrameResult(result="ok", image_quality_validate="pass", iqa_score=iqa_result.iqa_score, distortion_type=None, error_type=None)

        if frame is None or position is None:
            raise ValueError("Frame and position are required")

        if not isinstance(frame, str) or not isinstance(position, list):
            raise ValueError("Frame and position must be a string and a list")

        if len(position) != 4:
            raise ValueError("Position must be a list of 4 elements")

        # Coordinate of frame
        x1_f, y1_f, x2_f, y2_f = frame.split(",")
        x1_f = int(float(x1_f) * image.shape[1])
        y1_f = int(float(y1_f) * image.shape[0])
        x2_f = int(float(x2_f) * image.shape[1])
        y2_f = int(float(y2_f) * image.shape[0])

        # Coordinate of position
        x1_p = position[0]
        y1_p = position[1]
        x2_p = position[2]
        y2_p = position[3]

        if position is not None and (x1_p >= x1_f and y1_p >= y1_f and x2_p <= x2_f and y2_p <= y2_f):
            ratio_position_frame = self.calculate_area_ratio(image, frame, position)
        else:
            ratio_position_frame = 0
            return RedFrameResult(result="failed", image_quality_validate=None, iqa_score=None, distortion_type=None, error_type="out_of_frame_error")

        # Create a dictionary with the correct feature names that match the training data
        input_data = {
            'ration_position_frame': [ratio_position_frame],
            'x1': [float(position[0] / image.shape[1])],
            'y1': [float(position[1] / image.shape[0])],
            'x2': [float(position[2] / image.shape[1])],
            'y2': [float(position[3] / image.shape[0])]
        }

        # Convert to DMatrix with feature names
        dnew = xgb.DMatrix(pd.DataFrame(input_data))

        pred = self.model_redframe.predict(dnew)
        result = self.class_labels[int(pred[0])]

        # Step 1: Check the result of the redframe model
        if result == "good":
            pass
        elif result == "wide":
            return RedFrameResult(result="failed", image_quality_validate=None, iqa_score=None, distortion_type=None, error_type="wide_error")
        elif result == "angle":
            return RedFrameResult(result="failed", image_quality_validate=None, iqa_score=None, distortion_type=None, error_type="angle_error")
        elif result == "close":
            return RedFrameResult(result="failed", image_quality_validate=None, iqa_score=None, distortion_type=None, error_type="close_error")

        # Step 2: Check the result of the iqa model
        # Crop image using crop_image_func if provided
        if not crop_method:
            raise ValueError("crop_image_func is required to perform iqa detection")
        
        cropped_image = crop_method(image, position)

        iqa_result = self.iqa_service.iqa_detect(cropped_image)

        if iqa_result.image_quality_validate == "fail":
            return RedFrameResult(result="failed", image_quality_validate="fail", iqa_score=iqa_result.iqa_score, distortion_type=iqa_result.distortion_type, error_type=f"{iqa_result.distortion_type}_error")
        else:
            return RedFrameResult(result="ok", image_quality_validate="pass", iqa_score=iqa_result.iqa_score, distortion_type=None, error_type=None)