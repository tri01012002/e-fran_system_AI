"""
Authenticator for Omega P03

Handles authentication logic using trained model.
"""

from raiki_sdk import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import ConvnextSmall
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image

class OMGP03Authenticator(AuthenticateBase):
    """
    Authentication model for Omega P03
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = ConvnextSmall(num_classes=AUTH_NUM_CLASSES,checkpoint=AUTH_WEIGHTS, device=device, model_name=AUTH_MODEL_NAME).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Omega P03 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """

        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            probs, label = self.get_prob(probs)
            metadata = self.metadata

            return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]

        if index in [0, 2, 6]:  # fake - must be C
            return min(0.55, 1.0 - probs[index].item()), label
        return max(0.65, probs[index].item()), label

    def generate_heatmap(self) -> Image.Image:
        pass