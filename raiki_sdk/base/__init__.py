"""Base abstract classes and schemas for authentication and detection."""

from .schemas import (
    AuthResult,
    YoloResult,
    YoloValidation,
    ClassificationResultModel,
    ClassificationResultProb,
    IqaResult,
    RedFrameResult,
    DetectionResult,
    WatchBrandClassifyResult
)
from .authenticate_base import AuthenticateBase
from .detector_base import DetectorBase
from .classificator_base import ClassificatorBase
from .iqa_base import IqaBase

__all__ = [
    "AuthenticateBase",
    "DetectorBase",
    "IqaBase",
    "ClassificationResultModel",
    "ClassificationResultProb",
    "ClassificatorBase",
    "YoloResult",
    "YoloValidation",
    "IqaResult",
    "RedFrameValidation",
    "RedFrameResult",
    "AuthResult",
    "DetectionResult",
    "WatchBrandClassifyResult"
]

