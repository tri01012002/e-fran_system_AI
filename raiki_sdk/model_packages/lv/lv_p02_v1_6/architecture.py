import torch
import timm
import torch.nn as nn

class MaxvitSmall(nn.Module):
    def __init__(self, num_classes=2, model_name='maxvit_small_tf_512.in1k',
                 pretrained=False, checkpoint=None, device='cpu'):
        super(MaxvitSmall, self).__init__()

        self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)

        feature_dim = self.backbone.num_features

        self.backbone.reset_classifier(num_classes=0)

        self.fc = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, num_classes)
        )

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device)
            self.load_state_dict(state_dict)

    def forward(self, x):
        features = self.backbone(x)  # (B, in_features)
        output = self.fc(features)
        return output