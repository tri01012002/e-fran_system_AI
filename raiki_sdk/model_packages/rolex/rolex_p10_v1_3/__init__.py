"""
Rolex P10 Model Package

This package provides detector and authenticator for Rolex P10 models version 1.3.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP10Detector: Detector instance
    """
    from .detector import RolexP10Detector
    return RolexP10Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP10Authenticator: Authenticator instance
    """
    from .authenticator import RolexP10Authenticator
    return RolexP10Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
