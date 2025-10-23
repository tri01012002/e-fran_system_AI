from abc import ABC, abstractmethod
from PIL import Image
from .schemas import ClassificationResultProb, ClassificationResultModel

class ClassificatorBase(ABC):
    @abstractmethod
    def forward(self, image: Image.Image, topk: int = 3) -> ClassificationResultModel:
        pass

    @abstractmethod
    def get_prob(self, result_model: ClassificationResultModel = None, topk: int = 3) -> ClassificationResultProb:
        pass

    @abstractmethod
    def generate_heatmap(self) -> Image.Image:
        pass