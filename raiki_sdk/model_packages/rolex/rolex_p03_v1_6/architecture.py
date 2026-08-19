"""
Model Architecture for Rolex P03 - v2 (2-class: fake / real)

EfficientNet-B2 backbone with custom classification head.
Architecture follows the training configuration used to create best_model.pth.
"""

import torch
from torch import nn
from torchvision import models


def create_rolex_model(num_classes=2, pretrained=False, checkpoint=None, device='cpu'):
    """
    Build the EfficientNet-B2 model used in the training notebook.

    The training code uses the default EfficientNet-B2 classifier head:
        model.classifier[1] = nn.Linear(in_features, 2)
    so the architecture must match the saved checkpoint exactly.
    """
    if pretrained:
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
    else:
        model = models.efficientnet_b2(weights=None)

    for p in model.parameters():
        p.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if checkpoint is not None:
        ckpt = torch.load(checkpoint, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        else:
            state_dict = ckpt

        if all(k.startswith("model.") for k in state_dict.keys()):
            state_dict = {k[len("model."):]: v for k, v in state_dict.items()}

        model.load_state_dict(state_dict, strict=False)

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
