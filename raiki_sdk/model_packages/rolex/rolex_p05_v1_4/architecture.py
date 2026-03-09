import torch
import timm
import torch.nn as nn


class Maxvit(nn.Module):
    """
    Multi-task MaxViT screw-face classifier.

    1 shared backbone + 2 heads:
      - 'main' : 4-class  (fake=0 / real=1 / old=2 / new=3)  <- output chinh khi inference
      - 'auth' : 2-class  (fake=0 vs non-fake=1)             <- auxiliary task

    Training loss:
        L = CE(main, y4) + lambda_auth * CE(auth, y2)
      voi y2 = 0 neu class==fake, nguoc lai 1.
    """

    def __init__(
        self,
        num_classes: int = 4,
        model_name: str = 'maxvit_small_tf_512.in1k',
        pretrained: bool = False,
        drop_rate: float = 0.4,
        checkpoint: str = None,
        device: str = 'cpu',
    ):
        super().__init__()

        # Backbone chia se
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,  # bo head mac dinh
        )
        in_features = self.backbone.num_features

        # Head chinh: 4 class (fake / real / old / new)
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(drop_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.75),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),

            nn.Linear(256, num_classes),
        )

        # Head phu: binary fake(0) vs real/old/new(1)
        self.head_auth = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(drop_rate * 0.5),
            nn.Linear(256, 2),
        )

        # Khoi tao weight cho 2 head
        for head in [self.classifier, self.head_auth]:
            for m in head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        if checkpoint is not None:
            state_dict = torch.load(checkpoint, map_location=device, weights_only=True)
            self.load_state_dict(state_dict)

    def forward(self, x: torch.Tensor) -> dict:
        features = self.backbone(x)
        return {
            'main': self.classifier(features),  # 4-class logits
            'auth': self.head_auth(features),   # 2-class logits
        }
