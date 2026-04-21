"""
Authenticator for Louis Vuitton P07

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS)
from .architecture import Network_Wrapper
from .preprocessor import get_transform
from raiki_sdk.utils import map_generate, highlight_im
import torch
import torchvision.models as models
from typing import Tuple
from PIL import Image
import numpy as np

class LVP07Authenticator(AuthenticateBase):
    """
    Authentication model for Louis Vuitton P07
    """

    def __init__(self, device: str):
        self.device = device
        self.model = self._load_model(AUTH_WEIGHTS)
        self.transform = get_transform()
        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def _load_model(self, weights_path: str) -> Network_Wrapper:
        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        base_model = models.resnet50(weights=None)
        children = list(base_model.children())

        net_layers = [
            list(children[0:4]),
            list(children[4].children()),
            list(children[5].children()),
            list(children[6].children()),
            list(children[7].children()),
            [],
        ]

        net = Network_Wrapper(net_layers, AUTH_NUM_CLASSES)

        first_value = next(iter(checkpoint.values()))
        state_dict = checkpoint if isinstance(first_value, torch.Tensor) else (
            checkpoint.get('model') or checkpoint.get('state_dict') or checkpoint.get('model_state_dict')
        )

        net.load_state_dict(state_dict)
        net.to(self.device)
        net.eval()
        return net

    def forward(self, image) -> AuthResult:
        """
        Authenticate Louis Vuitton P07 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """
        metadata = self.metadata

        with torch.inference_mode():
            scource = self.transform(image).unsqueeze(0).to(self.device)

            output1, output2, output3, output_concat, map1, map2, map3 = self.model(scource)

            p1 = self.model.state_dict()['classifier3.1.weight']
            p2 = self.model.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output3, p1, p2)

            p1 = self.model.state_dict()['classifier2.1.weight']
            p2 = self.model.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output2, p1, p2)

            p1 = self.model.state_dict()['classifier1.1.weight']
            p2 = self.model.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output1, p1, p2)

            inputs_ATT = highlight_im(scource, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model(inputs_ATT)

            outputs_com2 = output1 + output2 + output3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT
            main_output = outputs_com

            pred1 = torch.argmax(output1, dim=1).cpu().numpy()[0]
            pred2 = torch.argmax(output2, dim=1).cpu().numpy()[0]
            pred3 = torch.argmax(output3, dim=1).cpu().numpy()[0]
            pred_concat = torch.argmax(output_concat, dim=1).cpu().numpy()[0]
            pred_main = torch.argmax(main_output, dim=1).cpu().numpy()[0]

            print(f"pred1: {int(pred1)}, pred2: {int(pred2)}, pred3: {int(pred3)}, pred_concat: {int(pred_concat)}, pred_main: {int(pred_main)}")

            prob1 = torch.softmax(output1, dim=1).cpu().numpy()[0]
            prob2 = torch.softmax(output2, dim=1).cpu().numpy()[0]
            prob3 = torch.softmax(output3, dim=1).cpu().numpy()[0]
            prob_concat = torch.softmax(output_concat, dim=1).cpu().numpy()[0]
            prob_main = torch.softmax(main_output, dim=1).cpu().numpy()[0]

            print(f"prob1: {float(np.max(prob1))}, prob2: {float(np.max(prob2))}, prob3: {float(np.max(prob3))}, prob_concat: {float(np.max(prob_concat))}, prob_main: {float(np.max(prob_main))}")

            pred_list = [
                (int(pred1), float(np.max(prob1))),
                (int(pred2), float(np.max(prob2))),
                (int(pred3), float(np.max(prob3))),
                (int(pred_concat), float(np.max(prob_concat))),
                (int(pred_main), float(np.max(prob_main))),
            ]

            new_pred_list = [
                (p, c) for p, c in pred_list if p == int(pred_main)
            ]

            average_conf = sum(c for _, c in new_pred_list) / len(new_pred_list)

            # index 0 = fake, index 1 = real
            if AUTH_CLASS_LABELS[int(pred_main)] == 'real':
                return AuthResult(probability=average_conf, label='real', metadata=metadata)
            else:
                return AuthResult(probability=1 - average_conf, label='fake', metadata=metadata)

    def get_prob(self, probs) -> Tuple[float, str]:
        pass

    def generate_heatmap(self) -> Image.Image:
        pass