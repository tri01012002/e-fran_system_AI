"""
Image Preprocessing for Rolex P08

Handles image transformations and preprocessing.
Uses raiki-sdk preprocessing utilities.
"""

from torchvision import transforms
from raiki_sdk.preprocessing.filters import CLAHE, UnsharpMask
def get_transform():
    """
    Get image transformation pipeline for Rolex P08

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
    transforms.ToTensor(),
    CLAHE(clip_limit=2.0, tile_grid_size=(8,8)),
    UnsharpMask(kernel_size=(5,5), sigma=1.0, amount=1.5),
])

