"""
Authenticator for Louis Vuitton P04

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, DECODER1_WEIGHTS, DECODER2_WEIGHTS, DECODER3_WEIGHTS, AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME, AUTH_MODEL_VERSION,
    YOLO_MODEL_1_VERSION, YOLO_MODEL_2_VERSION, AUTH_CLASS_LABELS)
from .architecture import Network_Wrapper, Anti_Noise_Decoder
from .preprocessor import get_transform
from raiki_sdk.utils import map_generate, highlight_im
from tresnet_utils import TResnetL_V2
import torch
from typing import Tuple
from PIL import Image
import numpy as np

class LVP04Authenticator(AuthenticateBase):
    """
    Authentication model for Louis Vuitton P04
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

        self.model, self.decoder1, self.decoder2, self.decoder3 = self._load_model(AUTH_WEIGHTS, DECODER1_WEIGHTS, DECODER2_WEIGHTS, DECODER3_WEIGHTS)

        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_1_version": YOLO_MODEL_1_VERSION,
            "yolo_model_2_version": YOLO_MODEL_2_VERSION
        }

    def _load_model(self, weights_path: str, decoder1_weights_path: str, decoder2_weights_path: str, decoder3_weights_path: str) -> Tuple[Network_Wrapper, Anti_Noise_Decoder, Anti_Noise_Decoder, Anti_Noise_Decoder]:
        """Load model weights into architecture"""

        # Initialize the base TResNet-L model
        model_params = {'num_classes': AUTH_NUM_CLASSES}
        base_model = TResnetL_V2(model_params)

        # Extract layers for Network_Wrapper
        net_layers = list(base_model.children())
        classifier = net_layers[1:3]
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())

        # Create the network wrapper
        net = Network_Wrapper(net_layers, AUTH_NUM_CLASSES, classifier)

        decoder1 = Anti_Noise_Decoder(1, 512)
        decoder2 = Anti_Noise_Decoder(2, 1024)
        decoder3 = Anti_Noise_Decoder(4, 2048)

        # Load the trained weights
        net.load_state_dict(torch.load(weights_path, map_location=self.device, weights_only=False))
        decoder1.load_state_dict(torch.load(decoder1_weights_path, map_location=self.device, weights_only=False))
        decoder2.load_state_dict(torch.load(decoder2_weights_path, map_location=self.device, weights_only=False))
        decoder3.load_state_dict(torch.load(decoder3_weights_path, map_location=self.device, weights_only=False))

        # Move model to device and set to evaluation mode
        net.to(self.device)
        decoder1.to(self.device)
        decoder2.to(self.device)
        decoder3.to(self.device)
        net.eval()
        decoder1.eval()
        decoder2.eval()
        decoder3.eval()

        return net, decoder1, decoder2, decoder3


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

            output_1, output_2, output_3, output_main, map1, map2, map3 = self.model(tensor_image)

            outputs = output_1 + output_2 + output_3 + output_main

            # Get predictions from all branches (for detailed output)
            pred_1 = torch.argmax(output_1, dim=1).cpu().numpy()[0]
            pred_2 = torch.argmax(output_2, dim=1).cpu().numpy()[0]
            pred_3 = torch.argmax(output_3, dim=1).cpu().numpy()[0]
            pred_main = torch.argmax(output_main, dim=1).cpu().numpy()[0]
            pred_ensemble = torch.argmax(outputs, dim=1).cpu().numpy()[0]
            
            # Apply softmax to get probabilities for all branches
            prob_1 = torch.softmax(output_1, dim=1).cpu().numpy()[0]
            prob_2 = torch.softmax(output_2, dim=1).cpu().numpy()[0]
            prob_3 = torch.softmax(output_3, dim=1).cpu().numpy()[0]
            prob_main = torch.softmax(output_main, dim=1).cpu().numpy()[0]
            prob_ensemble = torch.softmax(outputs, dim=1).cpu().numpy()[0]


            # Create pair of prediction list of tuple. e.g: [(pred1_cls, conf_pred1),...]
            pred_list = [(int(pred_1), float(np.max(prob_1))),
                         (int(pred_2), float(np.max(prob_2))),
                         (int(pred_3), float(np.max(prob_3))),
                         (int(pred_main), float(np.max(prob_main))),
                         (int(pred_ensemble), float(np.max(prob_ensemble)))]

            new_pred_list = []
            for pred in range(len(pred_list)):
                if pred_list[pred][0] == int(pred_ensemble):
                    new_pred_list.append((pred_list[pred][0], pred_list[pred][1]))

            # Calculate average confidence
            average_conf = sum(pred[1] for pred in new_pred_list) / len(new_pred_list)

            if int(pred_ensemble) in [0,1]:
                return AuthResult(probability=1-average_conf, label=AUTH_CLASS_LABELS[int(pred_ensemble)], metadata=metadata)

            return AuthResult(probability=average_conf, label=AUTH_CLASS_LABELS[int(pred_ensemble)], metadata=metadata)

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




