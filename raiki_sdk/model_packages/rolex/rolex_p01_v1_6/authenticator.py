"""
Authenticator for Rolex P01

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS)
from .architecture import MaxvitSmall
from .preprocessor import get_transform
import torch
from typing import Tuple, Dict, Optional
from PIL import Image


class RolexP01Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P01
    """

    def __init__(self, device: str):
        self.device = device
        self.model = MaxvitSmall(
            num_classes=AUTH_NUM_CLASSES,
            checkpoint=AUTH_WEIGHTS,
            device=device
        ).to(self.device).eval()
        self.transform = get_transform()
        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def _authenticate_single(self, image: Image.Image) -> AuthResult:
        with torch.inference_mode():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            prob, label = self.get_prob(probs)
            return AuthResult(probability=prob, label=label, metadata=self.metadata.copy())

    def _authenticate_batch(self, images_dict: Dict[int, Image.Image]) -> Optional[Dict[int, AuthResult]]:
        if not images_dict:
            return None

        class_ids = []
        images_list = []
        for cls_id, img in images_dict.items():
            if img is not None and isinstance(img, Image.Image):
                class_ids.append(cls_id)
                images_list.append(img)

        if not images_list:
            return None

        with torch.inference_mode():
            batch_tensor = torch.stack([
                self.transform(img) for img in images_list
            ]).to(self.device)
            logits = self.model(batch_tensor)
            batch_probs = torch.softmax(logits, -1)

            results = {}
            for i, cls_id in enumerate(class_ids):
                prob, label = self.get_prob(batch_probs[i])
                results[cls_id] = AuthResult(
                    probability=prob,
                    label=label,
                    metadata=self.metadata.copy()
                )
        return results

    def combine_min_prob(self, results: Dict[int, AuthResult]) -> Optional[AuthResult]:
        if not results:
            return None

        real_probs = [result.probability for result in results.values()]
        min_prob = min(real_probs)

        for result in results.values():
            if result.probability == min_prob:
                final_label = result.label
                break

        return AuthResult(probability=min_prob, label=final_label, metadata=self.metadata.copy())

    def forward(self, image) -> Optional[AuthResult]:
        if image is None:
            return None

        # Multi-box case: dict {class_id: PIL Image}
        if isinstance(image, dict):
            if not image:
                return None
            individual_results = self._authenticate_batch(image)
            if not individual_results:
                return None
            return self.combine_min_prob(individual_results)

        # Single image case
        if not isinstance(image, Image.Image):
            return None

        return self._authenticate_single(image)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]
        if index in [0, 1, 2]:  # fake classes
            return min(0.5, 1.0 - probs[index].item()), label
        else:                   # real classes
            return torch.max(probs).item(), label

    def generate_heatmap(self) -> Image.Image:
        pass