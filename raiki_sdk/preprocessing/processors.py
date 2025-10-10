import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def resize_pwd(img: np.ndarray, output_size: int = 224, color_padding: tuple = (85, 85, 85)) -> np.ndarray:
    """
    Resize the input image while preserving its aspect ratio, 
    then pad it to make it a square image of size output_size x output_size.

    Args:
        img (np.ndarray): Input image (H x W x 3)
        output_size (int): Desired output size (e.g., 224)
        color_padding (tuple): RGB color used for padding (default is gray)

    Returns:
        np.ndarray: Final square image of shape (output_size, output_size, 3)
    """
    h, w = img.shape[:2]

    # Compute the scaling factor to resize the longest side to output_size
    scale = output_size / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)

    # Resize the image with aspect ratio preserved
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create a square canvas with the desired padding color
    square_img = np.full((output_size, output_size, 3), color_padding, dtype=np.uint8)

    # Compute offsets to center the resized image on the square canvas
    x_offset = (output_size - new_w) // 2
    y_offset = (output_size - new_h) // 2

    # Place the resized image in the center of the canvas
    square_img[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_img

    return square_img

def resize_224(image: Image.Image) -> Image.Image:
    """
    Resize the input Image.Image to 224x224.
    """
    new_w  = 224
    new_h = 224
    image = image.resize((new_w, new_h), Image.LANCZOS)

    return image

def remove_white_padding(image: np.ndarray, threshold: int = 250) -> np.ndarray:
    """
    Remove white padding from the edges of an image.
    
    Args:
        image: Input image as numpy array
        threshold: Pixel value threshold to consider as "white" (default 250)
    
    Returns:
        Image with white padding removed
    """
    # Convert to grayscale for easier processing
    if len(image.shape) == 3:
        gray = np.mean(image, axis=2)
    else:
        gray = image
    
    # Find rows and columns that are NOT mostly white
    non_white_rows = np.where(np.min(gray, axis=1) < threshold)[0]
    non_white_cols = np.where(np.min(gray, axis=0) < threshold)[0]
    
    # If no non-white pixels found, return original image
    if len(non_white_rows) == 0 or len(non_white_cols) == 0:
        return image
    
    # Get the bounding box of non-white content
    top = non_white_rows[0]
    bottom = non_white_rows[-1] + 1
    left = non_white_cols[0]
    right = non_white_cols[-1] + 1
    
    # Crop the image to remove white padding
    return image[top:bottom, left:right]

def padding_crop_yolo(image: np.ndarray, position: list[int], p: int = 0, remove_padding: bool = False) -> np.ndarray:
    """
    Pad the image based on the position and return the padded image.

    Args:
        image: np.ndarray (cv2) BGR image
        position: list [x1, y1, x2, y2] in pixel (not normalized)
        p: padding ratio
        remove_padding: whether to remove white padding

    Returns:
        Padded image
    """

    x1, y1, x2, y2 = position
    if p<1: px,py = int((x2 - x1) * p), int((y2 - y1) * p)
    else: px,py = p,p  

    height, width, _ = image.shape
    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(width, x2 + px)
    y2 = min(height, y2 + py)
    cropped_img = image[y1:y2, x1:x2]
    if remove_padding:
        cropped_img = remove_white_padding(cropped_img)
    return cropped_img

class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def pil_to_tensor(img: Image.Image, device: str = 'cpu') -> torch.Tensor:
    """
    Convert a PIL Image to a tensor.
    """
    img = img.convert("RGB")
    img = AdaptiveResize(224)(img)
    img = ToTensor()(img).to(device=device)
    img = img.unsqueeze(0)
    return img