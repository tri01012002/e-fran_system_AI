import torch
import timm
import torch.nn as nn


class Maxvit(nn.Module):
    """
    Rolex P05 V1.5 Authenticator Architecture
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = 'maxvit_small_tf_384.in1k',
        pretrained: bool = False,
        drop_rate: float = 0.4,
        checkpoint: str = None,
        device: str = 'cpu',
    ):
        super().__init__()

        # Shared backbone
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0, 
        )
        in_features = self.backbone.num_features

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(drop_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.75),

            nn.Linear(512, num_classes),
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return self.classifier(features)
