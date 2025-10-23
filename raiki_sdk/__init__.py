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
    IqaResult,
    RedFrameResult,
    ClassificatorBase,
    IqaBase,
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

from raiki_sdk.core.registry import (
    AuthFeature,
    ClassificationFeature
)
from raiki_sdk.services import get_redframe_service, LIQEServices, LIQECustomServices

def get_auth_feature(category: str, part: str, model_version: str, device: str) -> AuthFeature:
    """
    Load specific authentication model version on-demand.    
    
    Args:
        category: Brand category ("rolex", "lv", etc.)
        part: Part identifier ("P01", "P02", etc. for auth)
        model_version: Version identifier ("v1_1", "v1_2", etc.)
    
    Returns:
        AuthFeature: Authentication model feature
    
    Example:
        >>> model = get_auth_model("rolex", "P01", "v1_1", "cuda")
        >>> detector = model.detector
        >>> authenticator = model.authenticator
    """
    return registry.get_auth_feature(category, part, model_version, device)

def get_classification_feature(category: str, part: str, model_version: str, device: str) -> ClassificationFeature:
    """
    Load specific classifier model version on-demand.    
    
    Args:
        category: Brand category ("rolex", "lv", etc.)
        part: Part identifier ("P00" for classifier)
        model_version: Version identifier ("v1_1", "v1_2", etc.)
    
    Returns:
        ClassificationFeature: Classifier model feature
    
    Example:
        >>> model = get_classifier_model("rolex", "P00", "v1_1", "cuda")
        >>> classifier = model.classifier
    """
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
    "YoloBase",
    "ClassificatorBase",
    "IqaBase",
    # Schemas
    "AuthResult",
    "YoloResult",
    "YoloValidation",
    "ClassificationResultModel",
    "ClassificationResultProb",
    "IqaResult",
    "RedFrameResult",
    # Preprocessing
    "resize_pwd",
    "padding_crop_yolo",
    "remove_white_padding",
    "pil_to_tensor",
    "CLAHE",
    "LoGFilter",
    "AdjustBrightnessContrast",
    # Model Management
    "list_models",
    "list_available_models",
    "get_auth_feature",
    "get_classification_feature",
    # Services
    "get_redframe_service",
    "LIQEServices",
    "LIQECustomServices",
]

