"""
Watch Brand Classify Package

Cung cấp factory để tạo model classify omega/rolex cho đồng hồ đã crop.
"""

def get_main_model(device: str):
    """
    Factory function để tạo instance classifier chính.

    Returns:
        WatchBrandClassifier: classifier instance
    """
    from .classifier import WatchBrandClassifier
    return WatchBrandClassifier(device)


__all__ = ["get_main_model"]