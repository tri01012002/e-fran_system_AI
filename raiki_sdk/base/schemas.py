from pydantic import BaseModel, ConfigDict
from typing import Any, Optional
from PIL import Image
from ultralytics.engine.results import Results as YoloResult

class AuthResult(BaseModel):
    probability: float
    label: str
    metadata: dict = {}

class AuthResultProb(BaseModel):
    real_prob: float
    label: str
    metadata: dict = {}
    # is_authentic: bool
    # confidence: float

class ClassificationResultModel(BaseModel):
    probabilities: dict[str, float]
    top_label: str
    metadata: dict = {}

class ClassificationResultProb(BaseModel):
    topk_probs: list[dict[str, Any]]
    metadata: dict = {}

# class YoloResult(BaseModel):
#     results: Any

class YoloValidation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_valid: bool
    position: list[int] | None = None
    class_id: int | None = None
    image_crop: Optional[Image.Image] = None