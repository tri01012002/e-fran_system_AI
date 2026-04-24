"""
Rolex P08 Model Package

This package provides detector and authenticator for Rolex P08 models version 1.4.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP08Detector: Detector instance
    """
    from .detector import RolexP08Detector
    return RolexP08Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP08Authenticator: Authenticator instance
    """
    from .authenticator import RolexP08Authenticator
    return RolexP08Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
