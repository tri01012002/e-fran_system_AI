"""
Louis Vuitton P03 V1.2 Model Package

This package provides detector and authenticator for Louis Vuitton P03 models version 1.2.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP03Detector: Detector instance
    """
    from .detector import LVP03Detector
    return LVP03Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP03Authenticator: Authenticator instance
    """
    from .authenticator import LVP03Authenticator
    return LVP03Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
