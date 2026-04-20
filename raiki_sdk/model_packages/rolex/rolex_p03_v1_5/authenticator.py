"""
Authenticator for Rolex P03 - v2 (2-class: fake / real)

get_prob now returns a single real_prob float and one of two labels:
  - 'fake'  (index 0)
  - 'real'  (index 1)
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS,
)
from .architecture import create_rolex_model
from .preprocessor import get_transform

import torch
from typing import Tuple
from PIL import Image


class RolexP03Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P03 v2 (2-class).

    Labels: fake (0) | real (1)
    """

    def __init__(self, device: str):
        self.device = device
        self.model = create_rolex_model(
            num_classes=AUTH_NUM_CLASSES,
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
        Authenticate a cropped Rolex P03 image.

        Args:
            image: PIL Image (RGB) — output of RolexP03Detector.crop()

        Returns:
            AuthResult: probability (real_prob float), label str, metadata dict
        """
        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, dim=-1).squeeze(0)
            real_prob, label = self.get_prob(probs)

        return AuthResult(probability=real_prob, label=label, metadata=self.metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        """
        Derive a single real_prob score and a label from the 2-class softmax output.

        Class mapping:
            index 0 -> fake   (real_prob = prob[1], capped at 0.5)
            index 1 -> real   (real_prob = prob[1])

        Args:
            probs: Tensor of shape [2] — softmax probabilities [fake, real]

        Returns:
            (real_prob, label)  where label in {'fake', 'real'}
        """
        index = torch.argmax(probs, dim=-1).item()
        label = AUTH_CLASS_LABELS[index]           # 'fake' or 'real'

        if index == 0:   # predicted fake
            # Real probability is low; cap at 0.5 to avoid misleading high scores
            real_prob = min(0.5, probs[1].item())
        else:            # index == 1, predicted real
            real_prob = probs[1].item()

        return real_prob, label

    def generate_heatmap(self) -> Image.Image:
        pass
