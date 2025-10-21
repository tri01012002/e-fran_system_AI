"""
Louis Vuitton P01 V1 Model Package

This package provides detector and authenticator for LV P01 models version 1.
"""

def get_detector():
    """
    Factory function to create detector instance
    
    Returns:
        LVP01V1Detector: Detector instance
    """
    from .detector import LVP01V1Detector
    return LVP01V1Detector()

def get_authenticator():
    """
    Factory function to create authenticator instance
    
    Returns:
        LVP01V1Authenticator: Authenticator instance
    """
    from .authenticator import LVP01V1Authenticator
    return LVP01V1Authenticator()

__all__ = ['get_detector', 'get_authenticator']
