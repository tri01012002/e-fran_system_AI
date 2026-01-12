"""
Rolex P00 V1.3 Model Package

This package provides detector and authenticator for Rolex P00 models version 1.3.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        RolexP00Detector: Detector instance
    """
    from .detector import RolexP00Detector
    return RolexP00Detector(device)

def get_main_model(device: str):
    """
    Factory function to create classifier instance

    Returns:
        RolexP00Classificator: Classifier instance
    """
    from .classifier import RolexP00Classificator
    return RolexP00Classificator(device)

__all__ = ['get_detector', 'get_main_model']
