"""
Authenticator for Rolex P04

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS_1,AUTH_WEIGHTS_2, AUTH_MODEL_2_NUM_CLASSES,
    AUTH_MODEL_1_NAME, AUTH_MODEL_1_VERSION, AUTH_MODEL_2_NAME, AUTH_MODEL_2_VERSION,
    YOLO_MODEL_VERSION, AUTH_MODEL_1_CLASS_LABELS, AUTH_MODEL_2_CLASS_LABELS,
    AUTH_MODEL_2_NAME, AUTH_MODEL_2_VERSION, AUTH_MODEL_2_NUM_CLASSES,
    AUTH_MODEL_1_NUM_CLASSES, TARGET_SIZE)
from .architecture import create_rolex_model, MaxvitSmall
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image

class RolexP04Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P04
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: RolexP04Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model_1 = MaxvitSmall(
            num_classes=AUTH_MODEL_1_NUM_CLASSES,
            checkpoint=AUTH_WEIGHTS_1,
            device=self.device,
            model_name=AUTH_MODEL_1_NAME).to(self.device).eval()
        self.model_2 = create_rolex_model(
            num_classes=AUTH_MODEL_2_NUM_CLASSES,
            checkpoint=AUTH_WEIGHTS_2, 
            device=self.device).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_1_name": AUTH_MODEL_1_NAME,
            "auth_model_1_classes": AUTH_MODEL_1_CLASS_LABELS,
            "auth_model_1_version": AUTH_MODEL_1_VERSION,
            "auth_model_2_name": AUTH_MODEL_2_NAME,
            "auth_model_2_classes": AUTH_MODEL_2_CLASS_LABELS,
            "auth_model_2_version": AUTH_MODEL_2_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Rolex P04 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """

        with torch.inference_mode():
            tensor_image_1 = self.transform(image).unsqueeze(0).to(self.device)
            logits_1 = self.model_1(tensor_image_1)
            result_1 = torch.softmax(logits_1, dim=-1)
            # print(f"RolexP04: result_1: {result_1}")
            probs_sm_1, labels_p1 = self.get_prob(result_1)
            metadata = self.metadata
            # print(f"RolexP04: probs_v1: {probs_sm_1}, labels_p1: {labels_p1}")

            # Check if the first model predicts 'real' and real_prob => Use model_auth_v2
            if labels_p1 == "real" and probs_sm_1 >= 0.8:

                # Apply transforms and prepare input tensor
                with torch.no_grad():
                    tensor_image_2 = self.transform(image).unsqueeze(0).to(self.device)
                    output_2 = self.model_2(tensor_image_2)
                    probs_2 = torch.softmax(output_2, dim=-1)
                    max_prob, predicted_index = torch.max(probs_2, dim=1)
                    predicted_index = predicted_index.item()
                    max_prob = max_prob.item()

                    # print(f"RolexP04: probs_v2: {max_prob}, index_v2: {AUTH_MODEL_2_CLASS_LABELS[predicted_index]}")

                if predicted_index == 0:  # "broken"
                    final_prob = probs_sm_1 * 0.7 if probs_sm_1 > 0.9 else probs_sm_1 * 0.75
                    return AuthResult(probability=final_prob, label="broken", metadata=metadata)

                elif predicted_index == 1:  # "real"
                    return AuthResult(probability=probs_sm_1, label="real", metadata=metadata)

                else:  # predicted_index == 2: "superfake"
                    final_prob = probs_sm_1 * 0.8 if probs_sm_1 > 0.9 else probs_sm_1 * 0.85
                    return AuthResult(probability=final_prob, label="superfake", metadata=metadata)

            else:
                return AuthResult(probability=probs_sm_1, label=labels_p1, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        """
        Get probability and label from model output.
        Returns 1 - prob_fake (which equals prob_real) to match original GGNet logic.
        """
        index = torch.argmax(probs, dim=-1).item()
        label = AUTH_MODEL_1_CLASS_LABELS[index]
        # Return 1 - prob_fake (which equals prob_real) to match original GGNet logic
        # probs shape is (1, 2), so probs[0][0] is the probability of class 0 (fake)
        return 1 - probs[0][0].item(), label

    def generate_heatmap(self) -> Image.Image:
        pass




