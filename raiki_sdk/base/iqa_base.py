from abc import ABC, abstractmethod
from PIL import Image
from .schemas import IqaResult

class IqaBase(ABC):
    @abstractmethod
    def iqa_detect(self, image: Image.Image) -> IqaResult:
        pass