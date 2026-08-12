import torch
from torchvision import models

from models import Network_Wrapper


def create_rolex_model(num_classes=2):
    """
    Create Rolex P03 model with the same architecture used during training.
    """

    # ResNet50 backbone
    net = models.resnet50(weights=None)

    # Same layer groups as training
    group0 = [
        net.conv1,
        net.bn1,
        net.relu,
        net.maxpool
    ]

    group1 = list(net.layer1.children())
    group2 = list(net.layer2.children())
    group3 = list(net.layer3.children())
    group4 = list(net.layer4.children())
    group5 = []

    net_layers = [
        group0,
        group1,
        group2,
        group3,
        group4,
        group5
    ]

    # Create model
    model = Network_Wrapper(
        net_layers,
        num_classes=num_classes
    )

    return model
def load_rolex_model(checkpoint_path, device="cpu"):
    """
    Load trained Rolex P03 model from checkpoint.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model = create_rolex_model(
        num_classes=2
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.to(device)
    model.eval()

    return model