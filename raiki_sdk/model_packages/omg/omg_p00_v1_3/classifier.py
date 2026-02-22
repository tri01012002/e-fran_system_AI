from raiki_sdk.base import ClassificatorBase, ClassificationResultProb
from .config import (
    CLASSIFIER_WEIGHTS, CLASSIFIER_NUM_CLASSES,
    CLASSIFIER_MODEL_NAME, CLASSIFIER_MODEL_VERSION,
    YOLO_MODEL_VERSION, CLASSIFIER_EMBEDDING_SIZE, TARGET_SIZE, EMBEDDING_STORAGE_PKL)
from .preprocessor import get_transform
import torch
from PIL import Image
from ._load_model import _load_weights, _load_database
from .architecture import FaceRecognizerBagLV
from typing import Any

class OMP00Classificator(ClassificatorBase):
    def __init__(self, device: str):
        self.device = device
        self.model = _load_weights(
            num_classes=CLASSIFIER_NUM_CLASSES,
            embedding_size=CLASSIFIER_EMBEDDING_SIZE,
            pretrained=False,
            model_name='vit_base_patch16_dinov3.lvd1689m',
            img_size=TARGET_SIZE,
            checkpoint_path=CLASSIFIER_WEIGHTS,
            device=self.device
        )
        self.transform = get_transform()
        self.vector_storage_manager = _load_database(
            load_path=EMBEDDING_STORAGE_PKL,
            device=self.device
        )
        self.metadata = {
            "auth_model_name": CLASSIFIER_MODEL_NAME,
            "auth_model_version": CLASSIFIER_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }
        self.faceRecognizer = FaceRecognizerBagLV(
            model=self.model,
            transform=self.transform,
            vector_db_file=EMBEDDING_STORAGE_PKL,
            device=self.device
        )
        self.probs: list[tuple[str, float]] = None
        self.topk: list[dict[str, float | str]] = None
        self.labels: list[str] = None

    def forward(self, image: Image.Image) -> list[tuple[str, float]]:
        # cv2.imwrite("image_predict.jpg", self.image_predict)
    
        # Turn on no-gradient mode
        with torch.no_grad():
            self.probs = self.faceRecognizer.recognize_pil_double(image, top_k=10, N_closest = 3)
            return self.probs

    def get_topk(self, result_model: Any, topk: int) -> ClassificationResultProb:
        topks_nor = []
        set_label = set()

        with torch.no_grad():
            for name, sim in result_model:
                name = name.replace("_rotate", "").replace("_gray","")
                name =  name.split("-")[0]
                if "_" in name:
                    set_label.add(name.split("_")[0])
                    set_label.add(name.split("_")[1])
                    topks_nor.append({"name": name.split("_")[0], "prob": sim})
                    topks_nor.append({"name": name.split("_")[1], "prob": sim})
                else: 
                    topks_nor.append({"name": name, "prob": sim})
                    set_label.add(name)

        topks_nor = topks_nor[:min(len(topks_nor),topk)]

        return ClassificationResultProb(topk_probs=topks_nor, metadata=self.metadata)

    def generate_heatmap(self) -> Image.Image:
        pass
        




