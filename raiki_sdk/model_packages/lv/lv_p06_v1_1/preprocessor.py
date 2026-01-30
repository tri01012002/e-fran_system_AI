"""
Image Preprocessing for Louis Vuitton P06

Handles image transformations and preprocessing.
"""

from torchvision import transforms
from .config import TARGET_SIZE


def get_transform():
    """
    Get image transformation pipeline for Louis Vuitton P06

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        # transforms.CenterCrop(368), 
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
