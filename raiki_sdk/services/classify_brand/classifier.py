"""
Classifier nhận ảnh đồng hồ đã crop, trả về brand omg/rolex.
"""

import logging
from typing import Tuple
from PIL import Image
import torch

from raiki_sdk.base import AuthenticateBase 

from .config import (
    CLASSIFY_WEIGHTS, CLASS_LABELS, MODEL_NAME, MODEL_VERSION,
    NUM_CLASSES, TARGET_SIZE
)
from .architecture import Maxvit
from .preprocessor import get_transform

_logger = logging.getLogger(__name__)


class WatchBrandClassifier(AuthenticateBase):
    def __init__(self, device: str):
        self.device = device
        self.model = Maxvit(
            num_classes=NUM_CLASSES,
            model_name=MODEL_NAME,
            checkpoint=CLASSIFY_WEIGHTS,
            device=device
        ).to(self.device).eval()
        self.transform = get_transform()
        self.metadata = {
            "model_name": MODEL_NAME,
            "class_labels": CLASS_LABELS,
            "model_version": MODEL_VERSION,
            "target_size": TARGET_SIZE,
        }

    def forward(self, image: Image.Image) -> Tuple[str, float]:
        with torch.no_grad():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            logits = self.model(tensor_image)
            probs = torch.softmax(logits, -1).squeeze(0)
            return self.get_prob(probs)

    def get_prob(self, probs: torch.Tensor) -> Tuple[str, float]:
        idx = torch.argmax(probs, -1).item()
        label = CLASS_LABELS[idx]
        prob_value = probs[idx].item()
        return label, prob_value

    def generate_heatmap(self):
        raise NotImplementedError