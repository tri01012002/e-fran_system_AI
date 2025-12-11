"""
Rolex P05 V1.3 Model Package

This package provides detector and authenticator for Rolex P05 models version 1.3.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP05Detector: Detector instance
    """
    from .detector import RolexP05Detector
    return RolexP05Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP05Authenticator: Authenticator instance
    """
    from .authenticator import RolexP05Authenticator
    return RolexP05Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
