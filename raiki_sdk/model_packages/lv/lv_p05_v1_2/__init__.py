"""
Louis Vuitton P05 V1.1 Model Package

This package provides detector and authenticator for Louis Vuitton P05 models version 1.1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        LVP05Detector: Detector instance
    """
    from .detector import LVP05Detector
    return LVP05Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        LVP05Authenticator: Authenticator instance
    """
    from .authenticator import LVP05Authenticator
    return LVP05Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
