import joblib
from .config import get_redframe_model_path
import numpy as np
import xgboost as xgb
from xgboost.core import Booster
import pandas as pd
from raiki_sdk.base import RedFrameResult
from .iqa_service import LIQEServices
from typing import Callable, Optional
from PIL import Image


class RedFrameService:
    def __init__(
        self,
        category: str,
        part: str,
        class_labels: list,
        advanced_mode: bool,
        device: str = "cuda",
        threshold: float = 0.1
    ):
        """Initialize RedFrameService."""
        self.category = category
        self.part = part
        self.model_path = None
        self.model_redframe = None

        if advanced_mode:
            self.model_path = get_redframe_model_path(self.category.lower(), self.part.lower())

        if self.model_path:
            self.model_redframe: Booster = joblib.load(self.model_path)
            
        self.class_labels = class_labels

        if "good" not in self.class_labels:
            raise ValueError("Class labels must include 'good'")

        self.advanced_mode = advanced_mode
        self.iqa_service = LIQEServices(
            device=device, threshold=threshold, use_advance_iqa=advanced_mode
        )

    # ----------------------------------------------------------------------
    # Geometry Utilities
    # ----------------------------------------------------------------------
    def _parse_frame_coordinates(self, image: np.ndarray, frame: str) -> tuple[int, int, int, int]:
        """Convert frame string (normalized) to pixel coordinates."""
        try:
            x1, y1, x2, y2 = frame.split(",")
            h, w = image.shape[0:2]
            return (
                int(float(x1) * w),
                int(float(y1) * h),
                int(float(x2) * w),
                int(float(y2) * h),
            )
        except ValueError:
            return (0.0, 0.0, w, h)

    def _calculate_area(self, image: np.ndarray, frame) -> float:
        """Calculate the pixel area of a frame."""
        if isinstance(frame, list):
            x1, y1, x2, y2 = frame
        else:
            x1, y1, x2, y2 = self._parse_frame_coordinates(image, frame)
        return abs((x2 - x1) * (y2 - y1))

    def _calculate_area_ratio(self, image: np.ndarray, frame: str, position: list) -> float:
        """Compute area ratio between position and frame."""
        area_frame = self._calculate_area(image, frame)
        if area_frame == 0:
            raise ValueError("Frame area is zero — invalid coordinates")
        area_position = self._calculate_area(image, position)
        return float(area_position / area_frame)

    # ----------------------------------------------------------------------
    # Validation Utilities
    # ----------------------------------------------------------------------
    def _validate_inputs(self, frame: Optional[str], position: Optional[list]):
        """Ensure inputs are valid and in the correct format."""
        if frame is None or position is None:
            print(f"Frame and position are required frame {frame} position {position}")
            return False
        if not isinstance(frame, str) or not isinstance(position, list):
            print(f"Frame must be a string and position must be a list frame {frame} position {position}")
            return False
        if len(position) != 4:
            print(f"Position must contain exactly 4 elements position {position}")
            return False
        return True
    def _is_position_inside_frame(
        self, image: np.ndarray, frame: str, position: list
    ) -> bool:
        """Check whether position is fully within frame boundaries."""
        x1_f, y1_f, x2_f, y2_f = self._parse_frame_coordinates(image, frame)
        x1_p, y1_p, x2_p, y2_p = position
        return x1_p >= x1_f and y1_p >= y1_f and x2_p <= x2_f and y2_p <= y2_f

    # ----------------------------------------------------------------------
    # Prediction Utilities
    # ----------------------------------------------------------------------
    def _prepare_features(
        self, image: np.ndarray, frame: str, position: list
    ) -> pd.DataFrame:
        """Prepare model input feature DataFrame."""
        ratio = self._calculate_area_ratio(image, frame, position)
        features = {
            "ratio_position_frame": [ratio],
            "x1": [float(position[0] / image.shape[1])],
            "y1": [float(position[1] / image.shape[0])],
            "x2": [float(position[2] / image.shape[1])],
            "y2": [float(position[3] / image.shape[0])],
        }
        return pd.DataFrame(features)

    def _predict_redframe_result(self, features: pd.DataFrame) -> str:
        """Run XGBoost redframe model and map to label."""
        dnew = xgb.DMatrix(features)
        pred = self.model_redframe.predict(dnew)
        return self.class_labels[int(pred[0])]

    # ----------------------------------------------------------------------
    # IQA Evaluation
    # ----------------------------------------------------------------------
    def _evaluate_iqa(self, image: np.ndarray, position: list, crop_method: Callable) -> RedFrameResult:
        """Evaluate image quality on cropped region."""
        if not crop_method:
            raise ValueError("crop_method is required for IQA evaluation")

        cropped_image = crop_method(image, position)
        if not isinstance(cropped_image, Image.Image):
            raise TypeError("crop_method must return a PIL.Image.Image object")

        iqa_result = self.iqa_service.iqa_detect(cropped_image)
        error_type = (
            None if iqa_result.distortion_type is None else f"{iqa_result.distortion_type}_error"
        )

        return RedFrameResult(
            result=bool(iqa_result.image_quality_validate),
            iqa_result=iqa_result,
            error_type=error_type,
        )

    # ----------------------------------------------------------------------
    # Main Inference Flow
    # ----------------------------------------------------------------------
    def forward(
        self,
        image: np.ndarray,
        frame: Optional[str] = None,
        position: Optional[list] = None,
        crop_method: Optional[Callable[[np.ndarray, list], Image.Image]] = None,
    ) -> RedFrameResult:
        """
        Perform RedFrame detection and (optionally) IQA quality validation.
        """
        # Step 1: Check frame inclusion
        is_valid_fr_position = self._validate_inputs(frame, position)
        if is_valid_fr_position:
            if not self._is_position_inside_frame(image, frame, position):
                return RedFrameResult(result=False, iqa_result=None, error_type="out_of_frame_error")

        # Mode 1: Simple IQA-only
        if not self.advanced_mode:
            iqa_result = self.iqa_service.iqa_detect(image)
            return RedFrameResult(
                result=iqa_result.image_quality_validate,
                iqa_result=iqa_result,
                error_type=None,
            )

        # Mode 2: Full redframe + IQA


        # Step 2: Redframe model prediction
        features = self._prepare_features(image, frame, position)
        prediction_label = self._predict_redframe_result(features)

        if prediction_label != "good":
            return RedFrameResult(result=False, iqa_result=None, error_type=f"{prediction_label}_error")

        # Step 3: IQA evaluation
        return self._evaluate_iqa(image, position, crop_method)
