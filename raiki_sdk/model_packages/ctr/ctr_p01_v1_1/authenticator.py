"""
Authenticator for Omega P02

Handles authentication logic using trained model.
"""

from raiki_sdk import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import MultiStageDINO
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image
import numpy as np

class CTRP01Authenticator(AuthenticateBase):
    """
    Authentication model for CTR P01
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
        self.model = MultiStageDINO(num_classes=AUTH_NUM_CLASSES,checkpoint=AUTH_WEIGHTS, device=device, model_name=AUTH_MODEL_NAME).to(self.device).eval()
        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Omega P01 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probability, label, metadata
        """

        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            type_logits, fine_logits = self.model(tensor_image)
            
            # Using fine classification for main result
            fine_probs = torch.softmax(fine_logits, -1).squeeze(0)
            
            # Type classification as additional info
            type_probs = torch.softmax(type_logits, -1).squeeze(0)
            type_idx = torch.argmax(type_probs).item()
            type_label = "Type1" if type_idx == 0 else "Type2"
            
            probs, label = self.get_prob(fine_probs)
            
            # Get the actual class name
            fine_idx = torch.argmax(fine_probs).item()
            fine_label = AUTH_CLASS_LABELS[fine_idx]
            
            metadata = {}
            metadata.update({
                "type_prediction": type_label,
                "fine_label": fine_label,
                "fine_probability": fine_probs[fine_idx].item(),
                "type_probability": type_probs[type_idx].item()
            })

            return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        # return real probability and label
        index = torch.argmax(probs, -1).item()
        class_name = AUTH_CLASS_LABELS[index]

        # Determine if it's real or fake based on label prefix
        if class_name.lower().startswith("fake"):
            return 1 - probs[index].item(), class_name
        else:
            return probs[index].item(), class_name

    def generate_heatmap(self) -> Image.Image:
        pass