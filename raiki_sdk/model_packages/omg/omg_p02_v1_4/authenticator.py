"""
Authenticator for Omega P02 v1.4

Handles authentication logic using trained ConvNeXt V2 Base model.
"""

from raiki_sdk import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import WatchClassifier
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image
import numpy as np


class OMGP02Authenticator(AuthenticateBase):
    """
    Authentication model for Omega P02 v1.4
    Uses ConvNeXt V2 Base 384 backbone (2-class: real vs fake)
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = WatchClassifier(
            num_classes=AUTH_NUM_CLASSES,
            model_name=AUTH_MODEL_NAME,
            pretrained=False,
            drop_rate=0.0,
            checkpoint=AUTH_WEIGHTS,
            device=device,
        ).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION,
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Omega P02 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """
        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            probs_val, label = self.get_prob(probs)
            metadata = self.metadata

            return AuthResult(probability=probs_val, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        """
        AUTH_CLASS_LABELS = ["real", "fake"]
        index 0 = real, index 1 = fake
        """
        index = torch.argmax(probs, -1).item()
        label = AUTH_CLASS_LABELS[index]

        if index == 1:  # fake
            label = "fake"
            return min(0.55, 1.0 - probs[index].item()), label
        # real
        label = "real"
        return max(0.65, probs[index].item()), label

    def generate_heatmap(self) -> Image.Image:
        pass
