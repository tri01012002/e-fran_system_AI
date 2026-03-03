"""
Rolex P01 Model Package

This package provides detector and authenticator for Rolex P01 models version 1.4.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP01Detector: Detector instance
    """
    from .detector import RolexP01Detector
    return RolexP01Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP01Authenticator: Authenticator instance
    """
    from .authenticator import RolexP01Authenticator
    return RolexP01Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
