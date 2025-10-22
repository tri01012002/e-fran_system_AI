"""Base abstract classes and schemas for authentication and detection."""

from raiki_sdk.base.schemas import (
    AuthResult,
    YoloResult,
    YoloValidation,
    ClassificationResultModel,
    ClassificationResultProb
)
from raiki_sdk.base.authenticate_base import AuthenticateBase
from raiki_sdk.base.yolo_base import YoloBase
from raiki_sdk.base.classificator_base import ClassificatorBase

__all__ = [
    "AuthenticateBase",
    "YoloBase",
    "AuthResult",
    "ClassificationResultModel",
    "ClassificationResultProb",
    "ClassificatorBase",
    "YoloResult",
    "YoloValidation",
]

