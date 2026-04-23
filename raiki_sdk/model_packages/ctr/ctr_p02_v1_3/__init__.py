"""
CTR P02 V1 Model Package

This package provides detector and authenticator for CTR P02 models version 1_1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        CTRP02Detector: Detector instance
    """
    from .detector import CTRP02Detector
    return CTRP02Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        CTRP02Authenticator: Authenticator instance
    """
    from .authenticator import CTRP02Authenticator
    return CTRP02Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
