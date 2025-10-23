"""
Raiki SDK - LuxAuthen Toolkit

A Python SDK providing base interfaces and utilities for counterfeit detection and product classification systems.
"""

__version__ = "0.1.1"

from raiki_sdk.base import (
    AuthenticateBase,
    YoloBase,
    AuthResult,
    ClassificationResultModel,
    ClassificationResultProb,
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

from raiki_sdk.core import registry
from raiki_sdk.config import FOLDER_CACHE_MODEL_HF, HF_TOKEN
from raiki_sdk.core.registry import (
    AuthFeature,
    ClassificationFeature
)
# from raiki_sdk.services import RedFrameService

def get_auth_feature(category: str, part: str, model_version: str, device: str) -> AuthFeature:
    return registry.get_auth_feature(category, part, model_version, device)

def get_classification_feature(category: str, part: str, model_version: str, device: str) -> ClassificationFeature:
    return registry.get_classification_feature(category, part, model_version, device)


def list_models() -> dict:
    """
    List all cached models (without loading them)
    
    Returns:
        dict: Dictionary of cached models with their metadata
    """
    return registry.list_models()

def list_available_models() -> list:
    """
    Scan and list all available model packages without loading them
    
    Returns:
        list: List of available models with their metadata
    """
    return registry.list_available_models()

__all__ = [
    # Base classes
    "AuthenticateBase",
    "FOLDER_CACHE_MODEL_HF",
    "HF_TOKEN",
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
    # Model Management
    "get_auth_feature",
    "get_classification_feature",
    "list_models",
    "list_available_models",
    "registry",
    # Services
    # "RedFrameService",
]

