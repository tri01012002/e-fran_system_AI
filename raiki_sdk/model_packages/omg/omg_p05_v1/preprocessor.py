"""
Image Preprocessing for Omega P05

Handles image transformations and preprocessing.
"""
from torchvision import transforms
from .config import TARGET_SIZE


def get_transform():
    """
    Get image transformation pipeline for Omega P06 V1

    Returns:
        Transform pipeline
    """
    return transforms.Compose([
    transforms.Resize((TARGET_SIZE, TARGET_SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

