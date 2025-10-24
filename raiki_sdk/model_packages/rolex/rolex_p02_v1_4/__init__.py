"""
Rolex P02 V1.4 Model Package

This package provides detector and authenticator for Rolex P02 models version 1.4.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP02Detector: Detector instance
    """
    from .detector import RolexP02Detector
    return RolexP02Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP02Authenticator: Authenticator instance
    """
    from .authenticator import RolexP02Authenticator
    return RolexP02Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
