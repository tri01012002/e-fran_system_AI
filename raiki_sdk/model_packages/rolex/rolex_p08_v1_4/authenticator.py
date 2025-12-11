"""
Authenticator for Rolex P08

Handles authentication logic using trained model.
"""

from raiki_sdk.base import AuthenticateBase, AuthResult
from .config import (
    AUTH_WEIGHTS, AUTH_MODEL_NAME, AUTH_MODEL_VERSION, AUTH_NUM_CLASSES,
    YOLO_MODEL_VERSION, AUTH_CLASS_LABELS, FILTER_CLASS_LABELS, FILTER_WEIGHTS, FILTER_NUM_CLASSES)
from .architecture import Network_Wrapper
from .preprocessor import get_transform
from tresnet_utils import TResnetL_V2
import torch
from typing import Tuple
from PIL import Image
from raiki_sdk.utils import map_generate, highlight_im
import numpy as np

class RolexP08Authenticator(AuthenticateBase):
    """
    Authentication model for Rolex P08
    """

    def __init__(self, device: str):
        """
        Initialize authenticator

        Args:
            model: RolexP08Architecture instance
            transform: Image transformation pipeline
            device: Device to run model on (cuda/cpu)
        """
        self.device = device
        self.model_auth = self._load_model(AUTH_WEIGHTS, AUTH_NUM_CLASSES)
        self.model_filter = self._load_model(FILTER_WEIGHTS, FILTER_NUM_CLASSES)
        self.transform = get_transform()

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "filter_class_labels": FILTER_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION
        }

    def _load_model(self, weights_path: str, num_classes: int) -> Network_Wrapper:
        """Load model weights into architecture"""

        checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)

        # Initialize the base TResNet-L model
        model_params = {'num_classes': num_classes}
        base_model = TResnetL_V2(model_params)

        # Extract layers for Network_Wrapper
        net_layers = list(base_model.children())
        net_layers = net_layers[0]
        net_layers = list(net_layers.children())

        # Create the network wrapper
        net = Network_Wrapper(net_layers, num_classes)

        # Load the trained weights
        net.load_state_dict(checkpoint['model'])

        # Move model to device and set to evaluation mode
        net.to(self.device)
        net.eval()

        return net

    def forward_filter(self, image: Image.Image):
        with torch.inference_mode():
            scource = self.transform(image).unsqueeze(0).to(self.device)

            # Main forward
            output1, output2, output3, output_concat, map1, map2, map3 = self.model_filter(scource)

            # Attention maps (same as your training/testing logic)
            p1 = self.model_filter.state_dict()['classifier3.1.weight']
            p2 = self.model_filter.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output3, p1, p2)

            p1 = self.model_filter.state_dict()['classifier2.1.weight']
            p2 = self.model_filter.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output2, p1, p2)

            p1 = self.model_filter.state_dict()['classifier1.1.weight']
            p2 = self.model_filter.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output1, p1, p2)

            # Apply attention to input image
            inputs_ATT = highlight_im(scource, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model_filter(inputs_ATT)

            # Combine predictions (same as in test function)
            outputs_com2 = output1 + output2 + output3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            main_output = outputs_com

            # Final probs / label from main_output
            probs = torch.softmax(main_output, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = FILTER_CLASS_LABELS[idx]
            print(f"Filter: conf: {conf}, label: {label}, idx: {idx}")

        return conf, label, idx

    def forward_auth(self, image: Image.Image):
        with torch.inference_mode():
            scource = self.transform(image).unsqueeze(0).to(self.device)

            # Main forward
            output1, output2, output3, output_concat, map1, map2, map3 = self.model_auth(scource)

            # Attention maps (same as your training/testing logic)
            p1 = self.model_auth.state_dict()['classifier3.1.weight']
            p2 = self.model_auth.state_dict()['classifier3.4.weight']
            att_map_3 = map_generate(map3, output3, p1, p2)

            p1 = self.model_auth.state_dict()['classifier2.1.weight']
            p2 = self.model_auth.state_dict()['classifier2.4.weight']
            att_map_2 = map_generate(map2, output2, p1, p2)

            p1 = self.model_auth.state_dict()['classifier1.1.weight']
            p2 = self.model_auth.state_dict()['classifier1.4.weight']
            att_map_1 = map_generate(map1, output1, p1, p2)

            # Apply attention to input image
            inputs_ATT = highlight_im(scource, att_map_1, att_map_2, att_map_3)
            output_1_ATT, output_2_ATT, output_3_ATT, output_concat_ATT, _, _, _ = self.model_auth(inputs_ATT)

            # Combine predictions (same as in test function)
            outputs_com2 = output1 + output2 + output3 + output_concat
            outputs_com = outputs_com2 + output_1_ATT + output_2_ATT + output_3_ATT + output_concat_ATT

            main_output = outputs_com

            # Final probs / label from main_output
            probs = torch.softmax(main_output, dim=1).cpu().numpy()[0]
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            label = AUTH_CLASS_LABELS[idx]
            print(f"Auth: conf: {conf}, label: {label}, idx: {idx}")

            if idx in [0, 2]:
                return min(max(1 - conf, 0.01), 0.55), label, idx
        return min(max(conf, 0.65), 0.99), label, idx

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate Rolex P08 part

        Args:
            image: Image.Image

        Returns:
            AuthResult: probabilities, label, metadata
        """

        filter_conf, filter_label, filter_idx = self.forward_filter(image)

        if filter_label == "three_dots":
            return AuthResult(probability=min(0.55, 1 - filter_conf), label="fake", metadata=self.metadata)

        auth_conf, auth_label, auth_idx = self.forward_auth(image)
        return AuthResult(probability=auth_conf, label=auth_label, metadata=self.metadata)

    def get_prob(self, probs: torch.Tensor) -> Tuple[float, str]:
        pass

    def generate_heatmap(self) -> Image.Image:
        pass




