"""
Omega P01 V1 Model Package

This package provides detector and authenticator for Omega P01 models version 1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        CTRP03Detector: Detector instance
    """
    from .detector import CTRP03Detector
    return CTRP03Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        CTRP03Authenticator: Authenticator instance
    """
    from .authenticator import CTRP03Authenticator
    return CTRP03Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
