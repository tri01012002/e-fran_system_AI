"""
Authenticator for Omega P02

Handles authentication logic using trained model.
"""

from raiki_sdk import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_CLASS_LABELS,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_NUM_CLASSES)
from .architecture import MaxvitSmall
from .preprocessor import get_transform
import torch
from typing import Tuple
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2

class OMGP02Authenticator(AuthenticateBase):
    """
    Authentication model for Omega P02
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
        self.model = MaxvitSmall(num_classes=AUTH_NUM_CLASSES,checkpoint=AUTH_WEIGHTS, device=device, model_name=AUTH_MODEL_NAME).to(self.device).eval()

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Omega P02 part

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
        
        # Grad-CAM
        heatmap = self.generate_heatmap(image)
        heatmap.save("heatmap.png")

        metadata = self.metadata

        return AuthResult(probability=probs, label=label, metadata=metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        index = torch.argmax(probs, -1)
        label = AUTH_CLASS_LABELS[index]
        # AUTH_CLASS_LABELS = ["type1_fake", "type1_old_real", "type1_real", "type1_white_fake", "type1_white_real", "type2_fake", "type2_real", "type3_fake", "type3_real"]

        if index == 0:  
            label = "fake"    
            return min(0.55, 1.0 - probs[index].item()), label
        label = "real"
        return max(0.65, probs[index].item()), label

    def generate_heatmap(self, image: Image.Image, bbox=None) -> Image.Image:
        """
        Generate Grad-CAM heatmap (crop object first to remove background)

        Args:
            image: PIL Image
            bbox: (x1, y1, x2, y2)

        Returns:
            Image.Image
        """

        original = np.array(image)

        # -------------------------
        # 1. Crop object
        # -------------------------
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            image_crop = image.crop((x1, y1, x2, y2))
        else:
            image_crop = image
            x1, y1 = 0, 0

        tensor_image = self.transform(image_crop).unsqueeze(0).to(self.device)

        features = []
        gradients = []

        def forward_hook(module, input, output):
            features.append(output.detach())

        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0].detach())

        target_layer = self.model.backbone.stages[-1].blocks[-1].conv

        handle_f = target_layer.register_forward_hook(forward_hook)
        handle_b = target_layer.register_full_backward_hook(backward_hook)

        logits = self.model(tensor_image)
        pred_class = logits.argmax(dim=1)

        self.model.zero_grad()
        logits[:, pred_class].backward()

        feature_map = features[0]
        gradient = gradients[0]

        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_map).sum(dim=1, keepdim=True)

        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        cam = cv2.resize(cam, (image_crop.size[0], image_crop.size[1]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        cam = 1 - cam  # Invert heatmap to highlight important regions

        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_TURBO)

        crop_np = np.array(image_crop)
        overlay_crop = cv2.addWeighted(crop_np, 0.6, heatmap, 0.4, 0)

        # -------------------------
        # 2. Paste back to original
        # -------------------------
        overlay = original.copy()
        overlay[y1:y1 + overlay_crop.shape[0], x1:x1 + overlay_crop.shape[1]] = overlay_crop

        handle_f.remove()
        handle_b.remove()

        return Image.fromarray(overlay)

