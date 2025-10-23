"""Services for specialized detection tasks."""

from .redframe_service import RedFrameService
from .iqa_service import LIQEServices
import logging
from typing import Dict

_logger = logging.getLogger(__name__)

# Global cache for RedFrameService instances
_redframe_service_cache: Dict[str, RedFrameService] = {}

def get_redframe_service(
    category: str, 
    part: str, 
    class_labels: list = None,
    advanced_mode: bool = True,
    device: str = "cuda", 
    threshold: float = 0.1
) -> RedFrameService:
    """
    Get cached RedFrameService instance - creates once and reuses.
    
    This function implements singleton pattern per configuration to avoid
    reloading heavy models (CLIP, XGBoost) on every request.
    
    Args:
        category: Brand category ("rolex", "lv", etc.)
        part: Part identifier ("P01", "P02", etc.)
        class_labels: List of class labels for redframe model. 
                     Defaults to ["good", "wide", "angle", "close"]
        advanced_mode: Whether to use advanced IQA mode with LIQE custom model.
                      Default True (uses LIQECustomServices)
        device: Device to run on ("cuda", "mps", "cpu")
        threshold: IQA quality threshold (0-1). Default 0.1
    
    Returns:
        RedFrameService: Cached service instance
    
    Example:
        >>> from raiki_sdk.services import get_redframe_service
        >>> service = get_redframe_service("rolex", "P01", device="mps")
        >>> result = service.forward(image, frame, position, crop_func)
    """
    if class_labels is None:
        class_labels = ["good", "wide", "angle", "close"]
    
    # Create cache key based on configuration parameters
    cache_key = f"{category.lower()}_{part.lower()}_{advanced_mode}_{device}"
    
    if cache_key not in _redframe_service_cache:
        _logger.info(f"Creating new RedFrameService for {category}-{part} on {device} (advanced={advanced_mode})")
        _redframe_service_cache[cache_key] = RedFrameService(
            category=category,
            part=part,
            class_labels=class_labels,
            advanced_mode=advanced_mode,
            device=device,
            threshold=threshold
        )
    else:
        _logger.info(f"Reusing cached RedFrameService for {category}-{part}")
    
    return _redframe_service_cache[cache_key]

__all__ = ["get_redframe_service", "LIQEServices"]

