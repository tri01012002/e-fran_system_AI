"""
Model Architecture for Rolex P03 - v2 (2-class: fake / real)

EfficientNet V2-S backbone with custom classification head.
Architecture is kept identical to v1_4; only num_classes changes (3 -> 2).
"""

import torch
from torch import nn
from torchvision import models


def create_rolex_model(num_classes=2, pretrained=False, checkpoint=None, device='cpu'):
    """
    Build EfficientNet V2-S model with custom head for Rolex P03 v2.

    Args:
        num_classes (int): Number of output classes. Default 2 (fake / real).
        pretrained (bool): Load ImageNet-1k pre-trained backbone weights.
        checkpoint (str | None): Path to a .pt state-dict file to load.
        device (str): Target device ('cpu' or 'cuda').

    Returns:
        nn.Module: Model instance (eval mode if checkpoint provided).
    """
    if pretrained:
        model = models.efficientnet_v2_s(
            weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        )
    else:
        model = models.efficientnet_v2_s(weights=None)

    # Freeze backbone by default (unfreeze during fine-tuning if needed)
    for p in model.parameters():
        p.requires_grad = False

    num_ftrs = model.classifier[1].in_features

    # Custom head — identical structure to v1_4
    model.classifier = nn.Sequential(
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
    """Load state-dict from *checkpoint_path* into *model* and switch to eval."""
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_batch(model, images, device='cpu'):
    """
    Run inference on a pre-processed image batch.

    Returns:
        preds_idx   (np.ndarray): Predicted class indices.
        probs       (np.ndarray): Softmax probabilities [N, num_classes].
        confidence  (np.ndarray): Max probability per sample.
    """
    model.eval()
    with torch.no_grad():
        images = images.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds_idx = torch.argmax(logits, dim=1)
        confidence = probs.max(dim=1)[0]

    return preds_idx.cpu().numpy(), probs.cpu().numpy(), confidence.cpu().numpy()
