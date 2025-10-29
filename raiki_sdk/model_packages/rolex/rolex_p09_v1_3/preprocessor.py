"""
Image Preprocessing for Rolex P09

Handles image transformations and preprocessing.
Uses raiki-sdk preprocessing utilities.
"""

from torchvision import transforms
from raiki_sdk.preprocessing.filters import LoGFilter, AdjustBrightnessContrast
def get_transform():
    """
    Get image transformation pipeline for Rolex P09

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    LoGFilter(sigma=2.0),
    AdjustBrightnessContrast(brightness_increase=10, contrast_factor=10),
    transforms.ToTensor(),
])

