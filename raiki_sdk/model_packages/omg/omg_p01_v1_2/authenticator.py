"""
Authenticator for Omega P01 v1.2

Handles authentication logic using trained model.
"""

import torch
from typing import Tuple, Dict, Optional
from PIL import Image

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_MODEL_NAME, AUTH_MODEL_VERSION, AUTH_NUM_CLASSES,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS
)
from .architecture import MaxvitSmall
from .preprocessor import get_transform


class OMGP01Authenticator(AuthenticateBase):
    """Authentication model for Omega P01 v1.2"""

    def __init__(self, device: str):
        """
        Initialize authenticator
        
        Args:
            device: Device to run model on (cuda/cpu)
        """
        print(f"Loading Omega P01 v1.2 model from {AUTH_WEIGHTS}")
        self.device = device
        self.model = MaxvitSmall(
            num_classes=AUTH_NUM_CLASSES,
            checkpoint=AUTH_WEIGHTS,
            device=self.device,
            model_name=AUTH_MODEL_NAME
        ).to(self.device).eval()
        self.transform = get_transform()
        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def _authenticate_single(self, image: Image.Image) -> AuthResult:
        """
        Authenticate a single image
        
        Args:
            image: PIL Image to authenticate
            
        Returns:
            AuthResult: Authentication result
        """
        with torch.inference_mode():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            probs = self.model.get_probs(tensor_image)
            prob, label = self.get_prob(probs)
            return AuthResult(probability=prob, label=label, metadata=self.metadata.copy())

    def _authenticate_batch(self, images_dict: Dict[int, Image.Image]) -> Optional[Dict[int, AuthResult]]:
        """
        Authenticate batch of images
        
        Args:
            images_dict: Dict mapping class_id to PIL Image
            
        Returns:
            Dict mapping class_id to AuthResult, or None if empty
        """
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

            batch_probs = self.model.get_probs(batch_tensor)

            results = {}
            for i, cls_id in enumerate(class_ids):
                probs = batch_probs[i:i+1]
                prob, label = self.get_prob(probs)
                results[cls_id] = AuthResult(
                    probability=prob,
                    label=label,
                    metadata=self.metadata.copy()
                )

        return results

    def combine_min_prob(self, results: Dict[int, AuthResult]) -> Optional[AuthResult]:
        """
        Combine results using minimum probability (conservative approach)
        
        Args:
            results: Dict mapping class_id to AuthResult
            
        Returns:
            AuthResult: Combined result, or None if empty
        """
        if not results:
            return None

        real_probs = [result.probability for result in results.values()]
        min_prob = min(real_probs)

        final_label = AUTH_CLASS_LABELS[1] if min_prob > 0.6 else AUTH_CLASS_LABELS[0]
        return AuthResult(probability=min_prob, label=final_label, metadata=self.metadata.copy())

    def forward(self, image) -> Optional[AuthResult]:
        """
        Authenticate image(s)
        
        Args:
            image: PIL Image or dict of {class_id: PIL Image}
            
        Returns:
            AuthResult: Authentication result, or None if invalid input
        """
        if image is None:
            return None

        # Multi image case
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
        """
        Extract probability and label from model output
        
        Args:
            probs: Model output probabilities [1, num_classes]
            
        Returns:
            Tuple of (real_probability, label_string)
        """
        probs_squeezed = probs.squeeze(0)
        index = torch.argmax(probs_squeezed, -1)
        # print(f"[DEBUG] probs: {probs_squeezed.tolist()}, index: {index.item()}")
        
        if index == 0:
            label = "fake"
            prob = min(0.55, 1.0 - probs_squeezed[index].item())
        else:
            label = "real"
            prob = max(0.65, probs_squeezed[index].item())
        
        # print(f"[DEBUG] result: prob={prob:.4f}, label={label}")
        return prob, label

    def generate_heatmap(self) -> Image.Image:
        """Generate heatmap visualization (not implemented)"""
        pass
