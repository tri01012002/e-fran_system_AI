"""
Louis Vuitton P01 V1.6 Model Package

This package provides detector and authenticator for Louis Vuitton P01 models version 1.6.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP01Detector: Detector instance
    """
    from .detector import LVP01Detector
    return LVP01Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP01Authenticator: Authenticator instance
    """
    from .authenticator import LVP01Authenticator
    return LVP01Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
