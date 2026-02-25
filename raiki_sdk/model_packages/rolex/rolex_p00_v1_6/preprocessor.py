from torchvision import transforms
from .config import TARGET_SIZE

def get_transform():
    """
    Get image transformation pipeline for Rolex P00
    
    Returns:
        Transform pipeline
    """
    return transforms.Compose([
    transforms.Resize(size=(TARGET_SIZE, TARGET_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
