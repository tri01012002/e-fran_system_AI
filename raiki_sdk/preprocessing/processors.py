import numpy as np
from PIL import Image
import cv2
from torchvision import transforms
from torchvision.transforms import ToTensor
import torch
import random

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

def pad_to_square(image, n=10, randomize=True):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    w, h = image.size
    rotation_angle = random.randint(-n, n) if randomize else 0

    # Xoay ảnh với nền trắng luôn
    rotated_image = image.rotate(
        rotation_angle,
        expand=True,
        resample=Image.BICUBIC,
        fillcolor=(255, 255, 255)  # Không còn cần (0,0,0,0)
    )

    rotated_w, rotated_h = rotated_image.size
    size = max(rotated_w, rotated_h)

    # Tạo nền trắng vuông
    background = Image.new('RGB', (size, size), (255, 255, 255))

    x_offset = (size - rotated_w) // 2
    y_offset = (size - rotated_h) // 2

    if randomize:
        max_x_shift = max(0, size // 6 - rotated_w // 6)
        max_y_shift = max(0, size // 6 - rotated_h // 6)
        x_offset += random.randint(-max_x_shift, max_x_shift)
        y_offset += random.randint(-max_y_shift, max_y_shift)
        x_offset = max(0, min(x_offset, size - rotated_w))
        y_offset = max(0, min(y_offset, size - rotated_h))

    background.paste(rotated_image, (x_offset, y_offset))
    return background


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


def padding_crop_yolo(image: np.ndarray, position: list[int], p: int = 0, rotate_180: bool = False, remove_padding: bool = False) -> np.ndarray:
    """
    Pad the image based on the position and return the padded image.

    Args:
        image: np.ndarray (cv2) BGR image
        position: list [x1, y1, x2, y2] in pixel (not normalized)
        p: padding ratio
        rotate_180: whether to rotate the image 180 degrees
        remove_padding: whether to remove white padding

    Returns:
        Padded image
    """

    if len(position) != 4:
        raise ValueError("position must have 4 elements [x1, y1, x2, y2]")

    x1, y1, x2, y2 = position
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid coordinates: x1 < x2 and y1 < y2 required")
    
    if p<1: px,py = int((x2 - x1) * p), int((y2 - y1) * p)
    else: px,py = p,p  

    if len(image.shape) == 2:
        height, width = image.shape
        # Handle grayscale
    else:
        height, width, _ = image.shape

    x1 = max(0, x1 - px)
    y1 = max(0, y1 - py)
    x2 = min(width, x2 + px)
    y2 = min(height, y2 + py)
    
    # Ensure valid crop coordinates after padding
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid crop coordinates after padding: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
    
    cropped_img = image[y1:y2, x1:x2]
    if remove_padding:
        cropped_img = remove_white_padding(cropped_img)
    if rotate_180:
        cropped_img = cv2.rotate(cropped_img, cv2.ROTATE_180)
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


def normalize_position_to_square(position: list[int]) -> list[int]:
    """
    Normalize a YOLO position [x1, y1, x2, y2] so that the crop becomes a square.

    Logic:
        - Compute width and height.
        - Take the larger side minus the smaller side.
        - Add that difference to the smaller side (keep the top-left corner fixed).

    Args:
        position (list|tuple): [x1, y1, x2, y2]

    Returns:
        list: Normalized position [x1, y1, x2, y2] forming a square.
    """
    if not position or len(position) != 4:
        return position

    x1, y1, x2, y2 = position
    width = x2 - x1
    height = y2 - y1

    # Already square
    if width == height:
        return [x1, y1, x2, y2]

    if width > height:
        diff = width - height
        y2 = y2 + diff
    else:
        diff = height - width
        x2 = x2 + diff

    return [int(x1), int(y1), int(x2), int(y2)]

def crop_center(image: Image.Image) -> Image.Image:
    """
    Crop the center of the input image to produce a square image.
    The crop size is determined by the smaller dimension (width or height),
    and the crop is centered along both axes.
    Args:
        image (Image.Image): Input PIL image.
    Returns:
        Image.Image: Center-cropped square image.
    """

    width, height = image.size
    center_size = min(width, height)
    left = (width - center_size) // 2
    top = (height - center_size) // 2
    right = left + center_size
    bottom = top + center_size
    return image.crop((left, top, right, bottom))