from pydantic import BaseModel
from typing import Any
from PIL import Image

class AuthResult(BaseModel):
    probability: float
    label: str
    metadata: dict = {}

class YoloResult(BaseModel):
    results: Any

class YoloValidation(BaseModel):
    is_valid: bool
    position: list[int] | None = None
    class_id: int | None = None