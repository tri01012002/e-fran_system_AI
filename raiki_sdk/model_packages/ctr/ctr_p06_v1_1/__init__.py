"""
Omega P01 V1 Model Package

This package provides detector and authenticator for Omega P01 models version 1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        CTRP06Detector: Detector instance
    """
    from .detector import CTRP06Detector
    return CTRP06Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        CTRP06Authenticator: Authenticator instance
    """
    from .authenticator import CTRP06Authenticator
    return CTRP06Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
