"""
Omega P01 V1 Model Package

This package provides detector and authenticator for Omega P01 models version 1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        CTRP05Detector: Detector instance
    """
    from .detector import CTRP05Detector
    return CTRP05Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        CTRP05Authenticator: Authenticator instance
    """
    from .authenticator import CTRP05Authenticator
    return CTRP05Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
