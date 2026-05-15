from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
from .config import TARGET_SIZE
class ResizePadSquare:
    def __init__(self, size, fill=255):
        self.size = size
        self.fill = fill  # 255 = trắng

    def __call__(self, img):
        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        img = img.resize((new_w, new_h), Image.BILINEAR)

        pad_w = self.size - new_w
        pad_h = self.size - new_h

        padding = (
            pad_w // 2,
            pad_h // 2,
            pad_w - pad_w // 2,
            pad_h - pad_h // 2
        )

        img = TF.pad(img, padding, fill=self.fill)

        return img

def get_transform():
    """
    Get image transformation pipeline for Rolex P00
    
    Returns:
        Transform pipeline
    """
    return transforms.Compose([
        ResizePadSquare(TARGET_SIZE, fill=255),  # padding trắng
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
