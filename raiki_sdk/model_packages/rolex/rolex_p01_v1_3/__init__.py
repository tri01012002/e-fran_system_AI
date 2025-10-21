"""
Rolex P01 V2 Model Package

This package provides detector and authenticator for Rolex P01 models version 2.
"""

def get_detector(device: str):
    """
    Factory function to create detector instance
    
    Returns:
        RolexP01Detector: Detector instance
    """
    from raiki_sdk.model_packages.rolex.rolex_p01_v1_3.detector import RolexP01Detector
    return RolexP01Detector(device)

def get_authenticator(device: str):
    """
    Factory function to create authenticator instance
        
    Returns:
        RolexP01Authenticator: Authenticator instance
    """
    from raiki_sdk.model_packages.rolex.rolex_p01_v1_3.authenticator import RolexP01Authenticator
    return RolexP01Authenticator(device)

__all__ = ['get_detector', 'get_authenticator']
