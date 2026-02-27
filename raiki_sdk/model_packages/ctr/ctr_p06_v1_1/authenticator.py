"""
Authenticator for Omega P02

Handles authentication logic using trained model.
"""

from raiki_sdk import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import DinoBinary
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image
import numpy as np

class CTRP06Authenticator(AuthenticateBase):
    """
    Authentication model for CTR P06
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model_name: name of model
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = DinoBinary(num_classes=AUTH_NUM_CLASSES, checkpoint=AUTH_WEIGHTS, device=device, model_name=AUTH_MODEL_NAME).to(self.device).eval()
        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate CTR P06 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probability, label, metadata
        """

        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            
            # Using binary classification probabilities
            probs = torch.softmax(logits, -1).squeeze(0)
            
            # Get result using common get_prob logic
            prob, label = self.get_prob(probs)
            
            # Get specific class name from label index
            idx = torch.argmax(probs, -1).item()
            fine_label = AUTH_CLASS_LABELS[idx]
            
            metadata = {}
            metadata.update({
                "fine_label": fine_label,
                "fine_probability": probs[idx].item(),
            })

            return AuthResult(probability=prob, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        # Matches P04 binary logic: return probabilities for Real class if needed or direct class_name
        # Here we follow the user's previously specified logic for return values
        index = torch.argmax(probs, -1).item()
        class_name = AUTH_CLASS_LABELS[index]

        # Determine if it's real or fake based on label prefix
        if class_name.lower().startswith("fake"):
            # Return "inverse" probability if it's fake (user's preferred logic from P01)
            return 1 - probs[index].item(), class_name
        else:
            return probs[index].item(), class_name

    def generate_heatmap(self) -> Image.Image:
        pass