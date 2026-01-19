"""
Omega P03 V1.3 Model Package

This package provides detector and authenticator for Omega P03 models version 1.3.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        OMGP03Detector: Detector instance
    """
    from .detector import OMGP03Detector
    return OMGP03Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        OMGP03Authenticator: Authenticator instance
    """
    from .authenticator import OMGP03Authenticator
    return OMGP03Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
