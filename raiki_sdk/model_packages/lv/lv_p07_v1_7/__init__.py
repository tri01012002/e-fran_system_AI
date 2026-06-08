"""
Louis Vuitton P07 V1.7 Model Package

This package provides detector and authenticator for Louis Vuitton P07 models version 1.7.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        LVP07Detector: Detector instance
    """
    from .detector import LVP07Detector
    return LVP07Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        LVP07Authenticator: Authenticator instance
    """
    from .authenticator import LVP07Authenticator
    return LVP07Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
