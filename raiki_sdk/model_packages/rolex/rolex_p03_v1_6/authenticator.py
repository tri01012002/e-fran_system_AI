"""
Authenticator for Rolex P03.

2-class classification:
    0 = fake
    1 = real
"""

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import torch
from PIL import Image

from config import  (
    AUTH_WEIGHTS,
    AUTH_NUM_CLASSES,
    AUTH_MODEL_NAME,
    AUTH_MODEL_VERSION,
    YOLO_MODEL_VERSION,
    AUTH_CLASS_LABELS,
)

from architecture import load_rolex_model
from preprocessor import get_transform


# =========================================================
# AUTH RESULT
# =========================================================

@dataclass
class AuthResult:
    probability: float
    label: str
    metadata: Dict[str, Any]


# =========================================================
# AUTHENTICATOR
# =========================================================

class RolexP03Authenticator:

    def __init__(self, device: str):

        self.device = torch.device(device)

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------

        self.model = load_rolex_model(
            checkpoint_path=AUTH_WEIGHTS,
            device=self.device,
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # -------------------------------------------------
        # Image preprocessing
        # -------------------------------------------------

        self.transform = get_transform()

        # -------------------------------------------------
        # Metadata
        # -------------------------------------------------

        self.metadata = {
            "auth_model_name": AUTH_MODEL_NAME,
            "auth_class_labels": AUTH_CLASS_LABELS,
            "auth_model_version": AUTH_MODEL_VERSION,
            "yolo_model_version": YOLO_MODEL_VERSION,
        }

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(self, image: Image.Image) -> AuthResult:
        """
        Authenticate a cropped Rolex P03 image.

        Args:
            image:
                PIL Image in RGB format.

        Returns:
            AuthResult:
                probability = probability of REAL
                label = fake / real
                metadata = model information
        """

        # -------------------------------------------------
        # Ensure RGB
        # -------------------------------------------------

        if image.mode != "RGB":
            image = image.convert("RGB")

        # -------------------------------------------------
        # Preprocess
        # -------------------------------------------------

        tensor_image = self.transform(image)

        # Add batch dimension
        tensor_image = tensor_image.unsqueeze(0)

        # Move to device
        tensor_image = tensor_image.to(self.device)

        # -------------------------------------------------
        # Model inference
        # -------------------------------------------------

        with torch.no_grad():

            outputs = self.model(tensor_image)

            # Network_Wrapper returns 7 outputs:
            #
            # outputs[0] -> classifier output
            # outputs[1] -> classifier output
            # outputs[2] -> classifier output
            # outputs[3] -> FINAL classifier output
            # outputs[4] -> feature map
            # outputs[5] -> feature map
            # outputs[6] -> feature map

            logits = outputs[3]

            # -------------------------------------------------
            # Softmax
            # -------------------------------------------------

            probs = torch.softmax(
                logits,
                dim=1
            ).squeeze(0)

        # -------------------------------------------------
        # Get prediction
        # -------------------------------------------------

        real_prob, label = self.get_prob(probs)

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return AuthResult(
            probability=real_prob,
            label=label,
            metadata=self.metadata,
        )

    # =====================================================
    # GET PROBABILITY
    # =====================================================

    def get_prob(
        self,
        probs: torch.Tensor
    ) -> Tuple[float, str]:
        """
        Convert model probabilities into label and
        REAL probability.

        Class mapping:

            0 = fake
            1 = real
        """

        # Get predicted class
        predicted_class = torch.argmax(
            probs,
            dim=0
        ).item()

        # -------------------------------------------------
        # Validate class index
        # -------------------------------------------------

        if predicted_class < 0 or predicted_class >= AUTH_NUM_CLASSES:
            raise ValueError(
                f"Invalid predicted class: {predicted_class}"
            )

        # -------------------------------------------------
        # Get label
        # -------------------------------------------------

        label = AUTH_CLASS_LABELS[predicted_class]

        # -------------------------------------------------
        # REAL probability
        #
        # Class 0 = fake
        # Class 1 = real
        #
        # Therefore real probability is always probs[1].
        # -------------------------------------------------

        real_prob = probs[1].item()

        return real_prob, label

    # =====================================================
    # HEATMAP
    # =====================================================

    def generate_heatmap(self, image: Image.Image):
        """
        Heatmap is not implemented yet.
        """

        raise NotImplementedError(
            "Heatmap generation is not implemented yet."
        )