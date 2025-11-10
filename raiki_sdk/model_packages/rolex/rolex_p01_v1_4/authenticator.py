"""
Authenticator for Rolex P01

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS)
from .architecture import EfficientNetLoG
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image

class RolexP01Authenticator(AuthenticateBase):
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
        self.model = EfficientNetLoG(model_name=AUTH_MODEL_NAME, 
                                    checkpoint=AUTH_WEIGHTS,
                                    num_classes=AUTH_NUM_CLASSES,
                                    in_chans=4,
                                    log_sigma=1.0).to(self.device).eval()

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
        """
        Process the probability

        Args:
            probs: torch.Tensor
            
        Returns:
            Tuple[float, str]: Probability and label
        """
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]
        if index == 0:
            return min(0.5, 1.0 - probs[0].item()), label
        else:
            return torch.max(probs).item(), label

    def generate_heatmap(self) -> Image.Image:
        pass


        

