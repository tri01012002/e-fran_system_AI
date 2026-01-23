from pydantic import BaseModel, ConfigDict, Field
from typing import Any, Optional, Literal
from dataclasses import dataclass    
from PIL import Image
from ultralytics.engine.results import Results as YoloResult


class AuthResult(BaseModel):
    probability: float
    label: str
    metadata: dict = Field(default_factory=dict)

class AuthResultProb(BaseModel):
    real_prob: float
    label: str
    metadata: dict = Field(default_factory=dict)
    # is_authentic: bool
    # confidence: float

class ClassificationResultModel(BaseModel):
    probabilities: dict[str, float]
    top_label: str
    metadata: dict = Field(default_factory=dict)

class ClassificationResultProb(BaseModel):
    topk_probs: list[dict[str, Any]]
    metadata: dict = Field(default_factory=dict)

class YoloValidation(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    is_valid: bool
    position: list[int] | dict | None = None 
    class_id: int | None = None
    image_crop: Optional[Image.Image | dict] = None

class IqaResult(BaseModel):
    iqa_score: float
    iqa_threshold: float | None = None
    distortion_type: str | None = None
    image_quality_validate: bool

class RedFrameResult(BaseModel):
    result: bool
    iqa_result: Optional[IqaResult] = None
    # image_quality_validate: Literal["pass", "fail"] | None = None
    # iqa_score: float | None = None
    # distortion_type: str | None = None
    error_type: Literal[
        "angle_error", 
        "wide_error", 
        "out_of_frame_error", 
        "no_frame_error",
        "detect_error",
        "close_error"
    ] | str | None = None
    
@dataclass
class DetectionResult:
    category: str
    confidence: float
    cropped_image: Optional[Image.Image] = None
    bbox: Optional[tuple[int, int, int, int]] = None
@dataclass
class WatchBrandClassifyResult:
    category: str
    confidence: float