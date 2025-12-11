"""
Authenticator for Louis Vuitton P04 V1.3

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_1_VERSION, YOLO_MODEL_2_VERSION, AUTH_CLASS_LABELS)
from .architecture import Network_Wrapper
from tresnet_utils import TResnetL_V2
from .preprocessor import get_transform
from raiki_sdk.utils import map_generate, highlight_im
import torch
from typing import Tuple
from PIL import Image
import numpy as np

class LVP04Authenticator(AuthenticateBase):
    """
    Authentication model for Louis Vuitton P04 V1.3
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: LVP04Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device

        self.model = self._load_model(AUTH_WEIGHTS)

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_1_version": YOLO_MODEL_1_VERSION,
            "yolo_model_2_version": YOLO_MODEL_2_VERSION
        }

    def _load_model(self, weights_path: str) -> Network_Wrapper:
        """Load model weights into architecture"""

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Initialize the base TResNet-L model
        model_params = {'num_classes': AUTH_NUM_CLASSES}
        base_model = TResnetL_V2(model_params)

        # Extract layers for Network_Wrapper
        net_layers = list(base_model.children())
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())

        # Create the network wrapper
        net = Network_Wrapper(net_layers, AUTH_NUM_CLASSES)

        # Load the trained weights
        net.load_state_dict(checkpoint['model'])

        # Move model to device and set to evaluation mode
        net.to(self.device)
        net.eval()

        return net

    def forward(self, image) -> AuthResult:
        """
        Authenticate Louis Vuitton P04 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """
        metadata = self.metadata

        with torch.inference_mode():
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)

            output1, output2, output3, output_concat, map1, map2, map3 = self.model(tensor_image)

            # Generate attention maps using the same logic as training/testing
            p1 = self.model.state_dict()['classifier3.1.weight']
            p2 = self.model.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output3, p1, p2)

            p1 = self.model.state_dict()['classifier2.1.weight']
            p2 = self.model.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output2, p1, p2)

            p1 = self.model.state_dict()['classifier1.1.weight']
            p2 = self.model.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output1, p1, p2)

            # Apply attention to input image
            inputs_ATT = highlight_im(tensor_image, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model(inputs_ATT)

            # Combine predictions (same as in test function)
            outputs_com2 = output1 + output2 + output3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            # Use combined prediction as the main prediction
            main_output = outputs_com

            # Get predections from all classifiers
            pred1 = torch.argmax(output1, dim=1).cpu().numpy()[0]
            pred2 = torch.argmax(output2, dim=1).cpu().numpy()[0]
            pred3 = torch.argmax(output3, dim=1).cpu().numpy()[0]
            pred_concat = torch.argmax(output_concat, dim=1).cpu().numpy()[0]
            pred_main = torch.argmax(main_output, dim=1).cpu().numpy()[0]

            # print(f"pred1: {int(pred1)}, pred2: {int(pred2)}, pred3: {int(pred3)}, pred_concat: {int(pred_concat)}, pred_main: {int(pred_main)}")

            # Apply softmax to get probabilities
            prob1 = torch.softmax(output1, dim=1).cpu().numpy()[0]
            prob2 = torch.softmax(output2, dim=1).cpu().numpy()[0]
            prob3 = torch.softmax(output3, dim=1).cpu().numpy()[0]
            prob_concat = torch.softmax(output_concat, dim=1).cpu().numpy()[0]
            prob_main = torch.softmax(main_output, dim=1).cpu().numpy()[0]

            # print(f"prob1: {float(np.max(prob1))}, prob2: {float(np.max(prob2))}, prob3: {float(np.max(prob3))}, prob_concat: {float(np.max(prob_concat))}, prob_main: {float(np.max(prob_main))}")

            # Create pair of prediction list of tuple. e.g: [(pred1_cls, conf_pred1),...]
            pred_list = [(int(pred1), float(np.max(prob1))),
                         (int(pred2), float(np.max(prob2))),
                         (int(pred3), float(np.max(prob3))),
                         (int(pred_concat), float(np.max(prob_concat))),
                         (int(pred_main), float(np.max(prob_main)))]

            # Create new pair of prediction list of tuple if all pred equal pred_main

            new_pred_list = []
            for pred in range(len(pred_list)):
                if pred_list[pred][0] == int(pred_main):
                    new_pred_list.append((pred_list[pred][0], pred_list[pred][1]))

            # Calculate average confidence
            average_conf = sum(pred[1] for pred in new_pred_list) / len(new_pred_list)

            if int(pred_main) in [0,1]:
                return AuthResult(probability=1-average_conf, label=AUTH_CLASS_LABELS[int(pred_main)], metadata=metadata)

            return AuthResult(probability=average_conf, label=AUTH_CLASS_LABELS[int(pred_main)], metadata=metadata)

    def get_prob(self, probs) -> Tuple[float, str]:
        """
        Process the probability

        Args:
            probs: torch.Tensor

        Returns:
            Tuple[float, str]: Probability and label
        """
        pass

    def generate_heatmap(self) -> Image.Image:
        pass




