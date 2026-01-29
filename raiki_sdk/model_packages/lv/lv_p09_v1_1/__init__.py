"""
Louis Vuitton P09 V1.1 Model Package

This package provides detector and authenticator for Louis Vuitton P09 models version 1.1.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        LVP09Detector: Detector instance
    """
    from .detector import LVP09Detector
    return LVP09Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        LVP09Authenticator: Authenticator instance
    """
    from .authenticator import LVP09Authenticator
    return LVP09Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
