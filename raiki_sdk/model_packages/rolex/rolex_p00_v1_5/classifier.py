from raiki_sdk.base import ClassificatorBase, ClassificationResultProb
from .config import (
    CLASSIFIER_WEIGHTS, CLASSIFIER_NUM_CLASSES,
    CLASSIFIER_MODEL_NAME, CLASSIFIER_MODEL_VERSION,
    YOLO_MODEL_VERSION, CLASSIFIER_EMBEDDING_SIZE, TARGET_SIZE, EMBEDDING_STORAGE_PKL)
from .preprocessor import get_transform
import torch
from PIL import Image
from ._load_model import _load_weights, _load_database
from .architecture import FaceRecognizerBrand
from typing import Any
from .labels import ROLEX_CODE_NAME_MAP

class RolexP00Classificator(ClassificatorBase):
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
        
        self.metadata = {
            "auth_model_name": CLASSIFIER_MODEL_NAME,
            "auth_model_version": CLASSIFIER_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }
        self.faceRecognizer = FaceRecognizerBrand(
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

    def _add_detail_name(self, result: list):
        set_label = set()
        result_new = []
        for item in result:
            name = item['name']
            base_name = name.split("-")[0]

            if name in ROLEX_CODE_NAME_MAP:
                item['detail_name'] = ROLEX_CODE_NAME_MAP[name]
                item['name'] = base_name
            elif base_name in ROLEX_CODE_NAME_MAP:
                item['detail_name'] = ROLEX_CODE_NAME_MAP[base_name]
                item['name'] = base_name
            if 'detail_name' not in item:
                continue

            key = item['name'] + "-" + item['detail_name']

            if key in set_label:
                continue

            set_label.add(key)
            result_new.append(item)

        return result_new

    def get_topk(self, result_model: Any, topk: int) -> ClassificationResultProb:
        topks_nor = []
        # set_label = set()

        with torch.no_grad():
            for name, sim in result_model:
                name = name.replace("_rotate", "").replace("_gray","")
                # name =  name.split("-")[0]
                

                # if name in set_label:
                #     continue
                # set_label.add(name)

                if "_" in name:
                    topks_nor.append({"name": name.split("_")[0], "prob": sim})
                    topks_nor.append({"name": name.split("_")[1], "prob": sim})
                else: 
                    topks_nor.append({"name": name, "prob": sim})

        topks_nor = self._add_detail_name(topks_nor)
        topks_nor = topks_nor[:min(len(topks_nor),topk)]
        for item in topks_nor:
            if item["name"] == "16800":
                item["name"] = '16800/168000'

            # REVUE_AI-410    
            if item["name"] == "268621":
                item["name"] = '169623'            

        return ClassificationResultProb(topk_probs=topks_nor, metadata=self.metadata)

    def generate_heatmap(self) -> Image.Image:
        pass
        




