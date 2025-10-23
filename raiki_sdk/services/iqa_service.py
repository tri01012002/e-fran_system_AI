from PIL import Image
from raiki_sdk.base import IqaBase, IqaResult
import pyiqa
from .config import LIQE_CUSTOM_WEIGHTS
from .iqa_core import LIQE, pil_to_tensor
import torch
import numpy as np
import cv2


class LIQEServices(IqaBase):
    _base_model = None  # pyiqa model cache
    _custom_model = None  # custom model cache

    def __init__(self, device: str = "cuda", threshold: float = 0.1, use_advance_iqa: bool = False):
        """
        Unified LIQE IQA service supporting both standard and custom models.

        Args:
            device (str): "cuda" or "cpu".
            threshold (float): Quality threshold.
            use_advance_iqa (bool): If True, use custom LIQE model. If False, use pyiqa model.
        """
        self.device = device
        self.threshold = threshold
        self.use_advance_iqa = use_advance_iqa
        self.resize_target = (224, 224)

        # Lazy load the models
        if use_advance_iqa:
            if LIQEServices._custom_model is None:
                LIQEServices._custom_model = LIQE(LIQE_CUSTOM_WEIGHTS, device)
        else:
            if LIQEServices._base_model is None:
                LIQEServices._base_model = pyiqa.create_metric('liqe', as_loss=False)

    def iqa_detect(self, image: Image.Image) -> IqaResult:
        """
        Check the quality of the image using LIQE metric.
        """
        # Ensure image is a valid PIL image
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                raise ValueError("Input must be a PIL Image or a valid numpy array.")

        if not self.use_advance_iqa:
            # --- Standard LIQE via pyiqa ---
            img_rsed = image.resize(self.resize_target, Image.LANCZOS)
            score = LIQEServices._base_model(img_rsed).item()
            return IqaResult(iqa_score=score, iqa_threshold=self.threshold, image_quality_validate=score >= self.threshold)
        else:
            # --- Advanced LIQE via custom model ---
            img_tensor = pil_to_tensor(image, device=self.device)
            with torch.no_grad():
                quality, _, distortion = LIQEServices._custom_model(img_tensor)

            score = quality.item()

            return IqaResult(
                iqa_score=score,
                iqa_threshold=self.threshold,
                distortion_type=distortion,
                image_quality_validate=score >= self.threshold
            )
