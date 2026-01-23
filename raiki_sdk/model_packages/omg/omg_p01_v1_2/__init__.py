"""
Omega P01 V1 Model Package

This package provides detector and authenticator for Omega P01 models version 1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        OMGP01Detector: Detector instance
    """
    from .detector import OMGP01Detector
    return OMGP01Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        OMGP01Authenticator: Authenticator instance
    """
    from .authenticator import OMGP01Authenticator
    return OMGP01Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
