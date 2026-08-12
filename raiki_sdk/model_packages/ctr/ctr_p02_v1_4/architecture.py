# import torch
# import timm
# import torch.nn as nn


# class MaxvitSmall(nn.Module):
#     def __init__(self, num_classes=2, model_name='maxvit_small_tf_512.in1k',
#                  pretrained=False, checkpoint=None, device='cpu'):
#         super(MaxvitSmall, self).__init__()

#         self.backbone = timm.create_model(model_name, pretrained=pretrained, in_chans=3)

#         feature_dim = self.backbone.num_features

#         self.backbone.reset_classifier(num_classes=0)

#         self.fc = nn.Sequential(
#             nn.LayerNorm(feature_dim),
#             nn.Linear(feature_dim, num_classes)
#         )

#         if checkpoint is not None:
#             state_dict = torch.load(checkpoint, map_location=device)
#             self.load_state_dict(state_dict)

#     def forward(self, x):
#         features = self.backbone(x)  # (B, in_features)
#         output = self.fc(features)
#         return output



import torch
import timm
import torch.nn as nn

class MaxvitSmall(nn.Module):
    def __init__(
        self,
        num_classes=2,
        model_name="vit_base_patch16_dinov3.lvd1689m",
        dropout=0.3,
        checkpoint=None,
        device="cpu"
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )

        dim = self.backbone.num_features

        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        if checkpoint:
            state_dict = torch.load(
                checkpoint,
                map_location=device
            )
            self.load_state_dict(state_dict)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)