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
import cv2

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
            
        if advanced_mode and self.model_redframe is None:
            raise RuntimeError("RedFrame model not loaded")
        
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
    def _parse_frame_coordinates(self, image: np.ndarray, frame: tuple) -> tuple[int, int, int, int]:
        """Convert frame string (normalized) to pixel coordinates."""
        h, w = image.shape[0:2]
        try:
            x1, y1, x2, y2 = frame
            return (
                int(float(x1) * w),
                int(float(y1) * h),
                int(float(x2) * w),
                int(float(y2) * h),
            )
        except ValueError:
            return (0.0, 0.0, max(w,h), max(w,h))

    def _get_center_frame(self, image: np.ndarray, scale: float = 0.6) -> tuple[int, int, int, int]:
        h, w = image.shape[0:2]

        frame_w, frame_h = int(w * scale), int(h * scale)
        
        x1 = (w - frame_w) // 2
        y1 = (h - frame_h) // 2
        x2 = x1 + frame_w    
        y2 = y1 + frame_h

        return (x1, y1, x2, y2)

    def _calculate_area(self, image: np.ndarray, frame) -> float:
        """Calculate the pixel area of a frame."""
        if isinstance(frame, (list, tuple)):
            x1, y1, x2, y2 = frame
        else:
            x1, y1, x2, y2 = self._parse_frame_coordinates(image, frame)
        return abs((x2 - x1) * (y2 - y1))

    def _calculate_area_ratio(self, image: np.ndarray, frame: tuple, position: list) -> float:
        """Compute area ratio between position and frame."""
        area_frame = self._calculate_area(image, frame)
        if area_frame == 0:
            raise ValueError("Frame area is zero — invalid coordinates")
        area_position = self._calculate_area(image, position)
        return float(area_position / area_frame)

    # ----------------------------------------------------------------------
    # Validation Utilities
    # ----------------------------------------------------------------------
    def _validate_inputs(self, frame: Optional[tuple], position: Optional[list]):
        if not isinstance(position, (list, tuple)):
            raise ValueError("Position must be list/tuple")
        if len(position) != 4:
            raise ValueError("Position must have 4 elements")
    def _is_position_inside_frame(
        self, frame: tuple, position: list
    ) -> bool:
        """Check whether position is fully within frame boundaries."""
        x1_f, y1_f, x2_f, y2_f = frame
        x1_p, y1_p, x2_p, y2_p = position
        return x1_p >= x1_f and y1_p >= y1_f and x2_p <= x2_f and y2_p <= y2_f

    # ----------------------------------------------------------------------
    # Prediction Utilities
    # ----------------------------------------------------------------------
    def _prepare_features(
        self, image: np.ndarray, frame: tuple, position: list
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
    def _evaluate_iqa(self, image: np.ndarray, position: list, crop_method: Callable, redframe: tuple) -> RedFrameResult:
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
            red_frame_bbox=redframe,
            error_type=error_type,
        )

    # ----------------------------------------------------------------------
    # Main Inference Flow
    # ----------------------------------------------------------------------
    def crop_method_default(self, image: np.ndarray, position: list) -> Image.Image:
        """Default crop method for RedFrameService."""
        x1, y1, x2, y2 = position
        # Padding
        w, h = image.shape[1], image.shape[0] 
        px, py =  int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
        x1 = max(0, x1 - px)
        y1 = max(0, y1 - py)
        x2 = min(w, x2 + px)
        y2 = min(h, y2 + py)
        cropped = image[y1:y2, x1:x2]

        # Convert BGR -> RGB
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        return Image.fromarray(cropped)

    def forward(
        self,
        image: np.ndarray,
        position: Optional[list] = None,
        crop_method: Optional[Callable[[np.ndarray, list], Image.Image]] = None,
    ) -> RedFrameResult:
        """
        Perform RedFrame detection (center-based) and IQA validation.
        """

        # Validate input
        if position is None or len(position) != 4:
            return RedFrameResult(
                result=False,
                red_frame_bbox=None,
                error_type="detect_error"
            )

        # Step 1: get center red frame
        redframe = self._get_center_frame(image)
        # Step 2: check xem position và redframe có hợp lệ không
        self._validate_inputs(redframe, position)
        # Step 3: Check bbox có nằm trong red frame không   
        if not self._is_position_inside_frame(redframe, position):
            return RedFrameResult(
                result=False,
                red_frame_bbox=redframe,
                error_type="out_of_frame_error"
            )

        # --------------------------------------------------
        # Mode 1: Simple IQA-only
        # --------------------------------------------------
        if not self.advanced_mode:
            if crop_method:
                image_crop = crop_method(image, position)
            else:
                image_crop = self.crop_method_default(image, position)

            iqa_result = self.iqa_service.iqa_detect(image_crop)

            return RedFrameResult(
                result=bool(iqa_result.image_quality_validate),
                red_frame_bbox=redframe,
                iqa_result=iqa_result,
                error_type=None if iqa_result.image_quality_validate else "iqa_error",
            )

        # --------------------------------------------------
        # Mode 2: Full redframe + XGBoost + IQA
        # --------------------------------------------------

        # Step 4: Feature cho model (dùng redframe bbox)
        features = self._prepare_features(image, redframe, position)

        prediction_label = self._predict_redframe_result(features)

        if prediction_label != "good":
            return RedFrameResult(
                result=False,
                red_frame_bbox=redframe,
                error_type=f"{prediction_label}_error"
            )

        # Step 4: IQA
        return self._evaluate_iqa(
            image=image,
            position=position,
            crop_method=crop_method or self.crop_method_default,
            redframe=redframe
        )