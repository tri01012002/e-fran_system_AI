"""
Authenticator for Omega P05 v1_3

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import Convnext
from .preprocessor import get_transform
import torch
from typing import Optional, Tuple, Dict
from PIL import Image

class OMGP05Authenticator(AuthenticateBase):
    """
    Authentication model for Omega P05 v1_3
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: OMGP05Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model = Convnext(
            num_classes=AUTH_NUM_CLASSES,
            model_name=AUTH_MODEL_NAME,
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
            # Store all class probabilities in metadata
            all_probs = self.get_all_probs(probs)
            result = AuthResult(probability=prob, label=label, metadata=self.metadata.copy())
            result.metadata['all_class_probs'] = all_probs
            return result
        
    def _authenticate_batch(self, images_dict: dict) -> dict:
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
                all_probs = self.get_all_probs(batch_probs[i])
                result = AuthResult(
                    probability=prob,
                    label=label,
                    metadata=self.metadata.copy()
                )
                result.metadata['all_class_probs'] = all_probs
                results[cls_id] = result
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

    def forward(self, image: Image.Image) -> Optional[AuthResult]:
        """
        Authenticate Omega P05 v1_3 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """
        if image is None:
            return None
        
        # Multi-box case: dict {class_id: PIL Image}
        if isinstance(image, dict):
            if not image:
                return None
            individual_results = self._authenticate_batch(image)
            print(f"Individual results: {individual_results}")
            if not individual_results:
                return None
            return self.combine_min_prob(individual_results)

        # Single image case
        if not isinstance(image, Image.Image):
            return None

        return self._authenticate_single(image)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        """ Get probability and label from model output """
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]
        if index in [0, 2]:  # fake classes
            return min(0.4, 1.0 - probs[index].item()), label
        else:                   # real classes
            return torch.max(probs).item(), label

    def get_all_probs(self, probs: torch.Tensor) -> Dict[str, float]:
        """
        Get probabilities for all classes
        
        Args:
            probs: Probability tensor from softmax
            
        Returns:
            Dict mapping class label to probability
        """
        all_probs = {}
        for i, class_label in enumerate(AUTH_CLASS_LABELS):
            all_probs[class_label] = probs[i].item()
        return all_probs

    def visualize_crop(
        self,
        image_crop,
        save_path: str = "cropped_image.jpg",
        show: bool = False
    ) -> Image.Image:
        """
        Visualize crop image(s) as a single PIL Image
        
        Args:
            image_crop: PIL Image or dict {class_id: PIL Image}
            save_path: Optional path to save the visualization
            show: Whether to call image.show()
            
        Returns:
            PIL Image visualization
        """
        # if image_crop is None:
        #     return None

        # if isinstance(image_crop, dict):
        #     panels = list(image_crop.values())
        #     panels = [img for img in panels if isinstance(img, Image.Image)]
        #     if not panels:
        #         return None

        #     # Resize all to same height
        #     target_h = max(img.height for img in panels)
        #     resized = []
        #     for img in panels:
        #         ratio = target_h / img.height
        #         new_w = int(img.width * ratio)
        #         resized.append(img.resize((new_w, target_h)))

        #     # Concatenate horizontally
        #     total_w = sum(img.width for img in resized)
        #     result = Image.new("RGB", (total_w, target_h))
        #     x_offset = 0
        #     for img in resized:
        #         result.paste(img, (x_offset, 0))
        #         x_offset += img.width

        # elif isinstance(image_crop, Image.Image):
        #     result = image_crop.copy()
        # else:
        #     return None

        # if save_path:
        #     result.save(save_path)

        # if show:
        #     result.show()

        # return result
        pass

    def generate_heatmap(self) -> Image.Image:
        pass



