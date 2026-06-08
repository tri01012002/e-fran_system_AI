"""
Authenticator for Rolex P05

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import Maxvit
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image

class RolexP05Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P05
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: RolexP05Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = Maxvit(
            num_classes=AUTH_NUM_CLASSES,
            model_name=AUTH_MODEL_NAME,
            pretrained=False,
            drop_rate=0.4,
            checkpoint=AUTH_WEIGHTS,
            device=device,
        ).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Rolex P05 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """

        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            raw = self.model(tensor_image)
            logits = raw['main'] if isinstance(raw, dict) else raw
            probs = torch.softmax(logits, -1).squeeze(0)
            probs, label = self.get_prob(probs)
            metadata = self.metadata

            return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]

        # Determine if it's real or fake based on label prefix
        if label.lower().startswith("fake"):
            # Return "inverse" probability if it's fake (user's preferred logic from P01)
            return 1 - probs[index].item(), label
        else:
            return max(0.6, probs[index].item()), label

    def generate_heatmap(self) -> Image.Image:
        pass
