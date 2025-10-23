"""Base abstract classes and schemas for authentication and detection."""

from .schemas import (
    AuthResult,
    YoloResult,
    YoloValidation,
    ClassificationResultModel,
    ClassificationResultProb,
    IqaResult,
    RedFrameResult,
)
from .authenticate_base import AuthenticateBase
from .yolo_base import YoloBase
from .classificator_base import ClassificatorBase
from .iqa_base import IqaBase

__all__ = [
    "AuthenticateBase",
    "YoloBase",
    "IqaBase",
    "ClassificationResultModel",
    "ClassificationResultProb",
    "ClassificatorBase",
    "YoloResult",
    "YoloValidation",
    "IqaResult",
    "RedFrameValidation",
    "RedFrameResult",
    "AuthResult"
]

