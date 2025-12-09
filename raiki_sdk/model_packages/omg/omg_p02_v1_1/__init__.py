"""
Omega P02 V1.1 Model Package

This package provides detector and authenticator for Omega P02 models version 1.1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        OMGP02Detector: Detector instance
    """
    from .detector import OMGP02Detector
    return OMGP02Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        OMGP02Authenticator: Authenticator instance
    """
    from .authenticator import OMGP02Authenticator
    return OMGP02Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
