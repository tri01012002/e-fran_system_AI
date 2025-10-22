"""
Raiki SDK - LuxAuthen Toolkit

A Python SDK providing base interfaces and utilities for counterfeit detection and product classification systems.
"""

__version__ = "0.1.1"

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

from raiki_sdk.core import registry

# from raiki_sdk.services import RedFrameService

def get_auth_model(category: str, part: str, model_version: str, device: str) -> dict:
    """
    Load specific model version on-demand
    
    Args:
        category: Brand category ("rolex", "lv", etc.)
        part: Part identifier ("P01", "P02", etc.)
        model_version: Version identifier ("v1", "v2", etc.)
    
    Returns:
        dict with 'detector' and 'authenticator' instances
    
    Example:
        >>> model = get_model("rolex", "P01", "v1")
        >>> detector = model['detector']
        >>> authenticator = model['authenticator']
    """
    return registry.get_model(category, part, model_version, device)

def get_model(category: str, part: str, model_version: str, device: str) -> dict:
    """
    Get model package - automatically determines if it's auth or classifier model
    
    Args:
        category: Brand category ("rolex", "lv", etc.)
        part: Part identifier ("P00" for classifier, "P01", "P02", etc. for auth)
        model_version: Version identifier ("v1_1", "v1_2", etc.)
    """
    return registry.get_model(category, part, model_version, device)

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
    "get_auth_model",
    "get_model",
    "list_models",
    "list_available_models",
    "registry",
    # Services
    # "RedFrameService",
]

