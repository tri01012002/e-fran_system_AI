"""
Rolex P09 Model Package

This package provides detector and authenticator for Rolex P09 models version 1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP09Detector: Detector instance
    """
    from .detector import RolexP09Detector
    return RolexP09Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP09Authenticator: Authenticator instance
    """
    from .authenticator import RolexP09Authenticator
    return RolexP09Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
