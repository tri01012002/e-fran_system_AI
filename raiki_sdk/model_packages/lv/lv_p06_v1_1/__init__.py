"""
Louis Vuitton P06 V1.1 Model Package

This package provides detector and authenticator for Louis Vuitton P06 models version 1.1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP06Detector: Detector instance
    """
    from .detector import LVP06Detector
    return LVP06Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP06Authenticator: Authenticator instance
    """
    from .authenticator import LVP06Authenticator
    return LVP06Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
