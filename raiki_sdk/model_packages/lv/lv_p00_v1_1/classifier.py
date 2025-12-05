from raiki_sdk.base import ClassificatorBase, ClassificationResultProb
from .config import (
    AUTH_WEIGHTS, NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, EMBEDDING_SIZE, TARGET_SIZE, VECTOR_STORAGE_PKL)
import torch.nn.functional as F
from .preprocessor import get_transform
import torch
from PIL import Image
from .architecture import MViTArcFaceModel
from .vector_storage import VectorStorageManager
from typing import Any

class LVP00Classificator(ClassificatorBase):
    def __init__(self, device: str):
        self.device = device
        self.model = self._load_model()
        self.transform = get_transform()
        self.vector_storage_manager = VectorStorageManager(
            model = self.model,
            transform = self.transform,
            vector_db_file = VECTOR_STORAGE_PKL,
            device = device
        )

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }
        

    def _load_model(self) -> MViTArcFaceModel:
        model = MViTArcFaceModel(
            num_classes=NUM_CLASSES,
            embedding_size=EMBEDDING_SIZE,
            pretrained=False,
            model_name='maxvit_small_tf_384.in1k',
            img_size=TARGET_SIZE
        )
        model.load_state_dict(torch.load(AUTH_WEIGHTS, map_location=self.device))
        model.eval()
        return model

    def forward(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vector = self.model(img_tensor)
        vector = vector.squeeze(0)
        return F.normalize(vector, p=2, dim=0)
    
    def normalize_output(self, results: list, topk: int) -> ClassificationResultProb:
        topk_probs = []
        set_label = set()

        # Hard code: M41418 +<->+ M41428 
        for name, sim in results:
            # Merge multi version
            name =  name.split("_")[0]
            if name not in set_label:
                topk_probs.append({
                    "name": name,
                    "prob": sim
                })
                set_label.add(name)

            if name == "M41418" and "M41428" not in set_label:
                topk_probs.append({
                    "name": "M41428",
                    "prob": sim
                })
                set_label.add("M41428")

            if name == "M41428" and "M41418" not in set_label:
                topk_probs.append({
                    "name": "M41418",
                    "prob": sim
                })             
                set_label.add("M41418")

        return ClassificationResultProb(
            topk_probs=topk_probs[:topk],
            metadata= self.metadata
        )


    def get_topk(self, result_model: Any, topk: int =3) -> ClassificationResultProb:
        results = self.vector_storage_manager.search_vector(result_model, topk)
        return self.normalize_output(results, topk=topk)

    def generate_heatmap(self) -> Image.Image:
        pass
        




