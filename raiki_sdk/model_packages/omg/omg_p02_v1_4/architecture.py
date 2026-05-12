import torch
import timm
import torch.nn as nn


class WatchClassifier(nn.Module):
    """
    convnextv2_base.fcmae_ft_in22k_in1k_384
    backbone output: 1024-dim feature vector (after global avg pool)
    """
    def __init__(self, num_classes=2,
                 model_name='convnextv2_base.fcmae_ft_in22k_in1k_384',
                 pretrained=True, drop_rate=0.3,
                 checkpoint=None, device='cpu'):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, in_chans=3, num_classes=0
        )
        in_features = self.backbone.num_features  # 1024 for convnextv2_base

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, num_classes),
        )

        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)

    def forward(self, x):
        return self.classifier(self.backbone(x))
