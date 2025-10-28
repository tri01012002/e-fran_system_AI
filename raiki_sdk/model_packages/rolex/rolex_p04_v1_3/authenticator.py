"""
Authenticator for Rolex P04

Handles authentication logic using trained model.
Migrated from features/rolex/RolexP01.py
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS,AUTH_WEIGHTS_v2, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS, AUTH_CLASS_LABELS_v2)
from .architecture import create_rolex_model, GGNet_P04
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image
from raiki_sdk.preprocessing import resize_pwd
import cv2

class RolexP04Authenticator(AuthenticateBase):
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
        self.model = GGNet_P04(checkpoint=AUTH_WEIGHTS,device=device).to(self.device)
        self.model_v2 = create_rolex_model(num_classes=3,checkpoint=AUTH_WEIGHTS_v2, device=device).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_num_classes": AUTH_NUM_CLASSES,
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

        with torch.inference_mode():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            numpy_image_hwc = tensor_image.squeeze(0).permute(1, 2, 0).cpu().numpy()
            image_model_v1 = resize_pwd(numpy_image_hwc, output_size=224,
                                        color_padding=(85, 85, 85))
            pil_image = Image.fromarray(cv2.cvtColor(image_model_v1, cv2.COLOR_BGR2RGB))
            scource = self.transform(pil_image).unsqueeze(0).to(self.device)
            probs_v1 = self.model(scource.float(), flag="val")
            result = torch.softmax(probs_v1, dim=-1)
            probs_sm_v1, labels_p1 = self.get_prob(result)
            metadata = self.metadata

            # Check if the first model predicts 'real' and real_prob => Use model_auth_v2
            if labels_p1 == "real" and probs_sm_v1 >= 0.8:

                image_model_v2 = resize_pwd(numpy_image_hwc, output_size=224, color_padding=(85, 85, 85))
                pil_image_v2 = Image.fromarray(cv2.cvtColor(image_model_v2, cv2.COLOR_BGR2RGB))

                # Apply transforms and prepare input tensor
                with torch.no_grad():
                    input_tensor = self.transform(pil_image_v2).unsqueeze(0).to(self.device)
                    output = self.model_v2(input_tensor)
                    probs = torch.softmax(output, dim=-1)
                    max_prob, predicted_index = torch.max(probs, dim=1)
                    predicted_index = predicted_index.item()
                    max_prob = max_prob.item()

                if predicted_index == 0:  # "broken"
                    final_prob = probs_sm_v1 * 0.7 if probs_sm_v1 > 0.9 else probs_sm_v1 * 0.75
                    return AuthResult(probability=final_prob, label="broken", metadata=metadata)

                elif predicted_index == 1:  # "real"
                    return AuthResult(probability=probs_sm_v1, label="real", metadata=metadata)

                else:  # predicted_index == 2: "superfake"
                    final_prob = probs_sm_v1 * 0.8 if probs_sm_v1 > 0.9 else probs_sm_v1 * 0.85
                    return AuthResult(probability=final_prob, label="superfake", metadata=metadata)

            else:
                return AuthResult(probability=probs_sm_v1, label=labels_p1, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        _, index = torch.max(probs, 1)
        label = AUTH_CLASS_LABELS[index]
        # if index == 0:
        #     return min(0.5, 1 - probs[0][0].item()), label
        return 1 - probs[0][0].item(), label
        # index = torch.argmax(probs, -1)
        # return probs[1].item(), P04_CLASSES[index]

    def generate_heatmap(self) -> Image.Image:
        pass




