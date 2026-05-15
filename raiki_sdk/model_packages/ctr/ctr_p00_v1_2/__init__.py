"""
CTR P00 V1.2 Model Package

This package provides detector and authenticator for CTR P00 models version 1.2.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        CTRP00Detector: Detector instance
    """
    from .detector import CTRP00Detector
    return CTRP00Detector(device)

def get_main_model(device: str):
    """
    Factory function to create classifier instance

    Returns:
        CTRP00Classificator: Classifier instance
    """
    from .classifier import CTRP00Classificator
    return CTRP00Classificator(device)

__all__ = ['get_detector', 'get_main_model']
