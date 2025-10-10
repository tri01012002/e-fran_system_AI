import numpy as np
import cv2
from PIL import Image
import torch

class CLAHE(object):
    """Wrapper so you can use it in torchvision.transforms.Compose"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8,8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def apply_clahe(self, img: torch.Tensor,
                clip_limit: float = 2.0,
                tile_grid_size: tuple = (8, 8)) -> torch.Tensor:
        """
        Apply CLAHE on a torch.Tensor image.
        Args:
            img: Tensor[C,H,W], either float in [0,1] or uint8 in [0,255].
            clip_limit: contrast clip limit.
            tile_grid_size: size of grid for histogram eq.
        Returns:
            Tensor[C,H,W], float in [0,1].
        """
        # to H×W×C uint8
        if img.dtype.is_floating_point:
            np_img = (img * 255).byte().permute(1,2,0).cpu().numpy()
        elif img.dtype == torch.uint8:
            np_img = img.permute(1,2,0).cpu().numpy()
        else:
            raise TypeError(f"Unsupported dtype {img.dtype}")

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # split channels, apply CLAHE, merge back
        chans = cv2.split(np_img)
        clahe_chans = [clahe.apply(c) for c in chans]
        out_np = cv2.merge(clahe_chans)

        # back to C×H×W float [0,1]
        out_t = torch.from_numpy(out_np).permute(2,0,1).float().div(255)
        return out_t

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.apply_clahe(img, self.clip_limit, self.tile_grid_size)

class UnsharpMask(object):
    """Wrapper so you can use it in torchvision.transforms.Compose"""
    def __init__(self,
                 kernel_size=(5,5),
                 sigma=1.0,
                 amount=1.0,
                 threshold=0):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.amount = amount
        self.threshold = threshold

    def apply_unsharp_mask(self, img: torch.Tensor,
                       kernel_size: tuple = (5, 5),
                       sigma: float = 1.0,
                       amount: float = 1.0,
                       threshold: int = 0) -> torch.Tensor:
        """
        Unsharp masking: out = orig + amount * (orig - blurred).
        Args:
            img: Tensor[C,H,W], float in [0,1] or uint8 in [0,255].
            kernel_size: Gaussian blur kernel size.
            sigma: Gaussian blur sigma.
            amount: strength of the sharpening.
            threshold: minimal brightness change to sharpen.
        Returns:
            Tensor[C,H,W], float in [0,1].
        """
        # to HxWxC uint8
        if img.dtype.is_floating_point:
            np_img = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
        elif img.dtype == torch.uint8:
            np_img = img.permute(1, 2, 0).cpu().numpy()
        else:
            raise TypeError(f"Unsupported dtype {img.dtype}")

        blurred = cv2.GaussianBlur(np_img, kernel_size, sigma)
        # calculate mask
        mask = cv2.subtract(np_img, blurred)
        if threshold:
            # zero out small differences
            mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_TOZERO)[1]
        sharpened = cv2.addWeighted(np_img, 1.0, mask, amount, 0)

        # back to C×H×W float[0,1]
        out_t = torch.from_numpy(sharpened).permute(2, 0, 1).float().div(255)
        return out_t

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        return self.apply_unsharp_mask(
            img,
            kernel_size=self.kernel_size,
            sigma=self.sigma,
            amount=self.amount,
            threshold=self.threshold
        )

class LoGFilter:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img)

        # Generate LoG kernel
        size = int(6 * self.sigma + 1) if self.sigma >= 1 else 7
        if size % 2 == 0:
            size += 1

        x, y = np.meshgrid(np.arange(-size//2+1, size//2+1), np.arange(-size//2+1, size//2+1))
        kernel = -(1/(np.pi * self.sigma**4)) * (1 - ((x**2 + y**2) / (2 * self.sigma**2))) * np.exp(-(x**2 + y**2) / (2 * self.sigma**2))
        kernel = kernel / np.sum(np.abs(kernel))

        # Perform convolution using OpenCV filter2D
        result = cv2.filter2D(img_np, -1, kernel)

        # Convert back to PIL Image
        img_pil = Image.fromarray(cv2.convertScaleAbs(result))

        return img_pil

class AdjustBrightnessContrast:
    def __init__(self, brightness_increase=0, contrast_factor=1.0):
        self.brightness_increase = brightness_increase
        self.contrast_factor = contrast_factor

    def __call__(self, img):
        # Convert PIL image to numpy array
        img_np = np.array(img, dtype=np.float32)

        # Adjust brightness
        img_np += self.brightness_increase

        # Calculate mean for contrast adjustment
        mean = np.mean(img_np)

        # Adjust contrast
        img_np = (img_np - mean) * self.contrast_factor + mean

        # Clip values to ensure they remain valid
        img_np = np.clip(img_np, 0, 255)

        # Convert back to PIL Image
        img_pil = Image.fromarray(img_np.astype(np.uint8))

        return img_pil