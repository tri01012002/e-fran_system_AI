"""
Rolex P04 Model Package

This package provides detector and authenticator for Rolex P04 models version 1.4.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP04Detector: Detector instance
    """
    from .detector import RolexP04Detector
    return RolexP04Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP04Authenticator: Authenticator instance
    """
    from .authenticator import RolexP04Authenticator
    return RolexP04Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
