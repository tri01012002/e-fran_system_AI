"""
Omega P06 v1.2 Model Package

This package provides detector and authenticator for Omega P06 models version 1.2.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        OMGP06Detector: Detector instance
    """
    from .detector import OMGP06Detector
    return OMGP06Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        OMGP06Authenticator: Authenticator instance
    """
    from .authenticator import OMGP06Authenticator
    return OMGP06Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
