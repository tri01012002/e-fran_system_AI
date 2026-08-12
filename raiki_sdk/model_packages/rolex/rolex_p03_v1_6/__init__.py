"""
Rolex P03 Model Package - v2 (2-class: fake / real)

This package provides detector and authenticator for Rolex P03 v2.
"""


def get_detector(device: str):
    """
    Factory function to create detector instance.

    Returns:
        RolexP03Detector: Detector instance
    """
    from .detector import RolexP03Detector
    return RolexP03Detector(device)


def get_main_model(device: str):
    """
    Factory function to create authenticator instance.

    Returns:
        RolexP03Authenticator: Authenticator instance (2-class)
    """
    from .authenticator import RolexP03Authenticator
    return RolexP03Authenticator(device)


__all__ = ['get_detector', 'get_main_model']
