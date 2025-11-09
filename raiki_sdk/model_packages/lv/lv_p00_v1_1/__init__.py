

def get_detector(device: str):
    from .detector import LVP00Detector
    return LVP00Detector(device)

def get_main_model(device: str):
    from .authenticator import LVP00Classificator
    return LVP00Classificator(device)

__all__ = ['get_detector', 'get_main_model']
