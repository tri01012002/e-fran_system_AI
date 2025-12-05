from abc import ABC, abstractmethod
from PIL import Image
from .schemas import ClassificationResultProb
from typing import Any

class ClassificatorBase(ABC):
    @abstractmethod
    def forward(self, image: Image.Image, topk: int = 3) -> Any:
        pass

    @abstractmethod
    def get_topk(self, result_model: Any, topk: int = 3) -> ClassificationResultProb:
        pass

    @abstractmethod
    def generate_heatmap(self) -> Image.Image:
        pass