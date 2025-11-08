"""
Louis Vuitton P04 Case 4 V1.4 Model Package

This package provides detector and authenticator for Louis Vuitton P04 Case 4 models version 1.4.
"""

# Import classes at module level for proper pickling support

def get_detector(device: str):
    """
    Factory function to create detector instance

    Returns:
        LVP04Case4Detector: Detector instance
    """
    from .detector import LVP04Case4Detector
    return LVP04Case4Detector(device)

def get_main_model(device: str):
    """
    Factory function to create authenticator instance

    Returns:
        LVP04Case4Authenticator: Authenticator instance
    """
    from .authenticator import LVP04Case4Authenticator
    return LVP04Case4Authenticator(device)

__all__ = ['get_detector', 'get_main_model']
