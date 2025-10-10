"""
Raiki SDK - Counterfeit Detection Toolkit

A Python SDK providing base interfaces and utilities for counterfeit detection systems.
"""

__version__ = "0.1.0"

from raiki_sdk.base import (
    AuthenticateBase,
    YoloBase,
    AuthResult,
    YoloResult,
    YoloValidation,
    # RedFrameResult,
)

from raiki_sdk.preprocessing import (
    resize_pwd,
    padding_crop_yolo,
    remove_white_padding,
    pil_to_tensor,
    CLAHE,
    LoGFilter,
    AdjustBrightnessContrast,
)

# from raiki_sdk.services import RedFrameService

__all__ = [
    # Base classes
    "AuthenticateBase",
    "YoloBase",
    # Schemas
    "AuthResult",
    "YoloResult",
    "YoloValidation",
    # Preprocessing
    "resize_pwd",
    "padding_crop_yolo",
    "remove_white_padding",
    "pil_to_tensor",
    "CLAHE",
    "LoGFilter",
    "AdjustBrightnessContrast",
    # Services
    # "RedFrameService",
]

