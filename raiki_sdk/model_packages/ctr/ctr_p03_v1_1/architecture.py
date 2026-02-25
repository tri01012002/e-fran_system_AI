import timm
from torch import nn
import torch
from .config import AUTH_MODEL_NAME
class MultiStageDINO(nn.Module):
    def __init__(self, num_classes, checkpoint=None, model_name=AUTH_MODEL_NAME, device='cuda'):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=(checkpoint is None), num_classes=0)
        dim = self.backbone.num_features

        self.type_head = nn.Linear(dim, 2)
        self.fine_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        if checkpoint is not None:
            self.load_state_dict(torch.load(checkpoint, map_location=device))

    def forward(self, x):
        feat = self.backbone(x)
        return self.type_head(feat), self.fine_head(feat)


# ================= LOSS =====================

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

