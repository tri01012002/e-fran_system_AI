"""
Image Preprocessing for Omega P04

Handles image transformations and preprocessing.
"""

from torchvision import transforms
from PIL import Image, ImageOps
from .config import TARGET_SIZE


class PadToSquare:
    """
    Pad image to square with specific color
    """
    def __init__(self, fill=(255, 255, 255)):
        self.fill = fill

    def __call__(self, img):
        w, h = img.size
        size = max(w, h)
        pad_w = size - w
        pad_h = size - h
        
        # Calculate padding: (left, top, right, bottom)
        # Matches cv2 logic: pad_h // 2, pad_h - pad_h // 2, pad_w // 2, pad_w - pad_w // 2
        padding = (
            pad_w // 2, 
            pad_h // 2, 
            pad_w - (pad_w // 2), 
            pad_h - (pad_h // 2)
        )
        return ImageOps.expand(img, padding, fill=self.fill)


def get_transform():
    """
    Get image transformation pipeline for Omega P04 inference.
    Matches the validation/inference transformation logic from training.

    Returns:
        Transform pipeline
    """

    return transforms.Compose([
        PadToSquare(),
        transforms.Resize((TARGET_SIZE, TARGET_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])



