"""
Omega P05 V1.2 Model Package

This package provides detector and authenticator for Omega P05 models version 1.2.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        OMGP05Detector: Detector instance
    """
    from .detector import OMGP05Detector
    return OMGP05Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        OMGP05Authenticator: Authenticator instance
    """
    from .authenticator import OMGP05Authenticator
    return OMGP05Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
