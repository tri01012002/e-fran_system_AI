import timm
from torch import nn
import torch
from .config import AUTH_MODEL_NAME
class ConvNext(nn.Module):
    def __init__(self, num_classes=2, model_name=AUTH_MODEL_NAME, 
                 pretrained=False, checkpoint=None, device='cpu'):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3
        )

        in_features = self.backbone.num_features
        self.backbone.reset_classifier(0)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


