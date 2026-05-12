"""
OMG P00 V1.5 Model Package

This package provides detector and authenticator for OMG P00 models version 1.5.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        OMGP00Detector: Detector instance
    """
    from .detector import OMP00Detector
    return OMP00Detector(device)

def get_main_model(device: str):
    """
    Factory function to create classifier instance

    Returns:
        OMGP00Classificator: Classifier instance
    """
    from .classifier import OMP00Classificator
    return OMP00Classificator(device)

__all__ = ['get_detector', 'get_main_model']
