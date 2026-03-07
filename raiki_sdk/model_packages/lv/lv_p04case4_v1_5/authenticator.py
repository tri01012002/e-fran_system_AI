"""
Authenticator for Louis Vuitton P04 Case 4 V1.4

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import MaxvitTiny
from .preprocessor import get_transform
import torch
import numpy as np
from typing import Tuple
from PIL import Image

class LVP04Case4Authenticator(AuthenticateBase):
    """
    Authentication model for Louis Vuitton P04 Case 4 V1.5
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: LVP04Case4Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = MaxvitTiny(num_classes=AUTH_NUM_CLASSES, model_name=AUTH_MODEL_NAME, pretrained=True).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Louis Vuitton P04 Case 4 V1.4 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """
        # ensure metadata stays up-to-date
        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            probs, label = self.get_prob(probs)
            metadata = self.metadata

            return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)

        return 1 - probs[0].item(), AUTH_CLASS_LABELS[index]

    def generate_heatmap(self) -> Image.Image:
        pass


if __name__ == "__main__":
    import argparse
    from raiki_sdk import load_image
    
    parser = argparse.ArgumentParser(description="Test MaxvitTiny authenticator on a single image")
    parser.add_argument("--image", required=True, help="Path to input image file")
    parser.add_argument("--device", default="cpu", help="Device to run on (cpu or cuda)")
    args = parser.parse_args()
    
    auth = LVP04Case4Authenticator(device=args.device)
    img = load_image(r"\\192.168.10.19\DataHandle\PERSONAL_DATA\QUANG\LV\LV_P04\lv_p04_case4_data\lv_p04_case4_data-padding\data\data-training\real\LV_P04_real_material_case4_1771632707092_2025-02-07_train_frames")
    result = auth.forward(img)
    print("AuthResult:", result)


        

