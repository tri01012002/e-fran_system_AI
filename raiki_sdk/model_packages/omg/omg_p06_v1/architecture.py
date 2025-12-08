import torch
import timm
import torch.nn as nn

class Maxvit(nn.Module):
    def __init__(self, num_classes=2, model_name='maxvit_small_tf_512.in1k',
                 pretrained=False, checkpoint=None, device='cpu'):
        super(Maxvit, self).__init__()

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
            map_location = torch.device(device) if isinstance(device, str) else device
            state_dict = torch.load(checkpoint, map_location=map_location, weights_only=False)
            self.load_state_dict(state_dict)

    def forward(self, x):
        features = self.backbone(x) 
        output = self.classifier(features)
        return output
