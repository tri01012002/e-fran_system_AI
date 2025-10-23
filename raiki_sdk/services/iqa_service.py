from PIL import Image
from raiki_sdk.base import IqaBase
from raiki_sdk.base import IqaResult
import pyiqa
from .config import LIQE_CUSTOM_WEIGHTS
from .iqa_core import LIQE, pil_to_tensor
import torch
import numpy as np
import cv2

class LIQEServices(IqaBase):
    def __init__(self, device: str = "cuda", threshold: float = 0.1):
        self.resize_target = (224, 224)
        self.device = device
        self.model = pyiqa.create_metric('liqe', as_loss=False)
        self.threshold = threshold
    
    def iqa_detect(self, image: Image.Image) -> IqaResult:
        """
        Check the quality of the image using LIQE metric.
        
        Args:
            image: PIL.Image.Image
        Returns:
            IqaResult: Result of the IQA metric.
        """

        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if isinstance(image, np.ndarray) else None
            if image is None:
                raise ValueError("Input must be a PIL Image or a valid numpy array.")

        # Resize the image to 224x224 for IQA
        img_rsed = image.resize(self.resize_target, Image.LANCZOS)
        score = self.model(img_rsed).item()

        if score < self.threshold:
            return IqaResult(iqa_score=score, image_quality_validate="fail")
        else:
            return IqaResult(iqa_score=score, image_quality_validate="pass")


class LIQECustomServices(IqaBase):
    def __init__(self, device: str = "cuda", threshold: float = 0.1):
        self.device = device
        self.threshold = threshold
        self.model = LIQE(LIQE_CUSTOM_WEIGHTS, device)

    def iqa_detect(self, image: Image.Image) -> IqaResult:
        """
        Check the quality of the image using LIQE metric.
        
        Args:
            image: PIL.Image.Image
        Returns:
            IqaResult: Result of the IQA metric.
        """
        if not isinstance(image, Image.Image):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) if isinstance(image, np.ndarray) else None
            if image is None:
                raise ValueError("Input must be a PIL Image or a valid numpy array.")

        # Convert the image to tensor
        img_tensor = pil_to_tensor(image, device=self.device)

        # Forward pass through the model
        with torch.no_grad():
            quality, _, distortion = self.model(img_tensor)

        if quality < self.threshold:
            return IqaResult(iqa_score=quality.item(), distortion_type=distortion, image_quality_validate="fail")
        else:
            return IqaResult(iqa_score=quality.item(), distortion_type=distortion, image_quality_validate="pass")

