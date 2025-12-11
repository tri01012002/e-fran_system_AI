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
        self.model = Maxvit(num_classes=AUTH_NUM_CLASSES,checkpoint=AUTH_WEIGHTS, device=device).to(self.device).eval()

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
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            probs, label = self.get_prob(probs)
            metadata = self.metadata

            return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]

        if index == 1:  # real_a - must be A
            # 0.7 - 1
            # 0.8 => 0.8; 0.5 => 0.6
            return max(0.7, probs[index].item()), f"{probs[index]}-{label}"
        elif index == 0:  # fake - must be C
            # 0.01 -  0.5
            # 0.4 => 0.5; 0.8 => 0.2
            return max(0.5, 1.0 - probs[index].item()), f"{probs[index]}-{label}"
        elif index == 2:  # real_c - can be B or C
            # 0.6 - 0.75
            # 0.9 => 0.68; # 0.7 => 0.6; 0.5 => 0.6
            return max(0.6, probs[index].item()), f"{probs[index]}-{label}"
        else:  # superfake - can be B or C
            # 0.5 - 0.75
            # 0.9 => 0.68; # 0.7 => 0.52; 0.4 => 0.5
            return min(0.5, 1 - probs[index].item()), f"{probs[index]}-{label}"

    def generate_heatmap(self) -> Image.Image:
        pass


        

