"""
Image Preprocessing for LV_P04 V1.5

Handles image transformations and preprocessing.
Uses raiki-sdk preprocessing utilities.
"""

from torchvision import transforms
from .config import TARGET_SIZE


def get_transform():
    """
    Get image transformation pipeline for LV_P04 V1.5

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
