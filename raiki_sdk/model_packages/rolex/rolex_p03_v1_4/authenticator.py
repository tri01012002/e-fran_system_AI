"""
Authenticator for Rolex P01

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS)
from .architecture import create_rolex_model
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image

class RolexP03Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P01
    """
    
    def __init__(self, device: str):
        """
        Initialize authenticator
        
        Args:
            model: RolexP01Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = create_rolex_model(num_classes=AUTH_NUM_CLASSES,checkpoint=AUTH_WEIGHTS, device=device).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }
    
    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Rolex P01 part
        
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
        if index == 0:
            real_prob = (probs[1] + probs[2]).item()
            real_prob = min(0.5, real_prob)
        elif index == 1:  # real_normal
            real_prob = probs[1].item()
        elif index == 2:  # real_old
            original_prob = probs[2].item()
            real_prob = original_prob
        else:
            real_prob = 0.0
        return real_prob, label

    def generate_heatmap(self) -> Image.Image:
        pass


        

