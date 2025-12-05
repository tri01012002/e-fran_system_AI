"""
Louis Vuitton P00 V1.1 Model Package

This package provides detector and authenticator for Louis Vuitton P00 models version 1.1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP00Detector: Detector instance
    """
    from .detector import LVP00Detector
    return LVP00Detector(device)

def get_main_model(device: str):
    """
    Factory function to create classifier instance

    Returns:
        LVP00Classificator: Classifier instance
    """
    from .classifier import LVP00Classificator
    return LVP00Classificator(device)

__all__ = ['get_detector', 'get_main_model']
