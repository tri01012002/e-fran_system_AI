from abc import ABC, abstractmethod
from typing import Any, Tuple
from PIL import Image
from .schemas import AuthResult

class AuthenticateBase(ABC):
    @abstractmethod
    def forward(self, image: Image.Image) -> AuthResult:
        pass

    @abstractmethod
    def get_prob(self, output: Any) -> Tuple[float, str]:
        pass

    @abstractmethod
    def generate_heatmap(self) -> Image.Image:
        pass