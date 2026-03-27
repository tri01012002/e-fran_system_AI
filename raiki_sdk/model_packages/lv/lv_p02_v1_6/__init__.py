"""
Louis Vuitton P02 V1.5 Model Package

This package provides detector and authenticator for Louis Vuitton P02 models version 1.5.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP02Detector: Detector instance
    """
    from .detector import LVP02Detector
    return LVP02Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP02Authenticator: Authenticator instance
    """
    from .authenticator import LVP02Authenticator
    return LVP02Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
