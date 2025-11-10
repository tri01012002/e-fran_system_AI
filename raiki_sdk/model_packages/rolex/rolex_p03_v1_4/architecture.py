import torch
from torch import nn
from torchvision import models


def create_rolex_model(num_classes=2, pretrained=False, checkpoint = None, device ='cpu'):
    if pretrained:
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    else:
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


def load_checkpoint(model, checkpoint_path, device='cpu'):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_batch(model, images, device='cpu'):
    model.eval()
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds_idx = torch.argmax(logits, 1)
        confidence = probs.max(dim=1)[0]

    return preds_idx.cpu().numpy(), probs.cpu().numpy(), confidence.cpu().numpy()