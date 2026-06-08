import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import timm


def create_rolex_model(num_classes=3, checkpoint=None, device='cpu'):
    model = models.efficientnet_v2_s(weights=None)

    # Đóng băng tất cả parameters
    for p in model.parameters():
        p.requires_grad = False

    num_ftrs = model.classifier[1].in_features

    model.classifier = torch.nn.Sequential(
        nn.Dropout(0.4, inplace=True),
        nn.Linear(num_ftrs, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.3, inplace=True),
        nn.Linear(512, 256),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(256),
        nn.Dropout(0.2, inplace=True),
        nn.Linear(256, num_classes),
    )
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


class MaxvitSmall(nn.Module):
    """
    MaxViT Small model for Rolex P04 authentication model 1
    """
    def __init__(self, num_classes=2, model_name='maxvit_small_tf_224.in1k',
                 pretrained=False, checkpoint=None, device='cpu'):
        super(MaxvitSmall, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)

        in_features = self.backbone.num_features

        self.backbone.reset_classifier(num_classes=0)

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)

    def forward(self, x):
        features = self.backbone(x)  # (B, in_features)
        output = self.classifier(features)
        return output
