"""
Louis Vuitton P04 V1.3 Model Package

This package provides detector and authenticator for Louis Vuitton P04 models version 1.3.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP04Detector: Detector instance
    """
    from .detector import LVP04Detector
    return LVP04Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP04Authenticator: Authenticator instance
    """
    from .authenticator import LVP04Authenticator
    return LVP04Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
