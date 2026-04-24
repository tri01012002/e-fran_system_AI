"""
Image Preprocessing for Rolex P03 - v2 (2-class)

Handles image transformations and preprocessing.
Uses bicubic interpolation consistent with v2 training pipeline.
"""

from torchvision import transforms
from .config import TARGET_SIZE


def get_transform():
    """
    Get image transformation pipeline for Rolex P03.

    Matches the val/inference transform used during training:
      Resize(224, 224, bicubic) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)

    Returns:
        transforms.Compose: Transform pipeline
    """
    bicubic = transforms.InterpolationMode.BICUBIC
    return transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=bicubic),
        transforms.CenterCrop(TARGET_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
