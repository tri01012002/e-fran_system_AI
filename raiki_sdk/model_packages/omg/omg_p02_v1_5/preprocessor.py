"""
Image Preprocessing for Omega P02 v1.5

Handles image transformations and preprocessing.
Uses torchvision transforms matching the training pipeline from the notebook.
"""

from torchvision import transforms
from .config import TARGET_SIZE


def get_transform():
    """
    Get image transformation pipeline for Omega P02 v1.5

    Matches the val_transform from omg-efficientnetv2-p02.ipynb:
      - Resize to TARGET_SIZE x TARGET_SIZE (bicubic)
      - ToTensor
      - Normalize with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        transforms.Resize(
            (TARGET_SIZE, TARGET_SIZE),
            interpolation=transforms.InterpolationMode.BICUBIC
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
