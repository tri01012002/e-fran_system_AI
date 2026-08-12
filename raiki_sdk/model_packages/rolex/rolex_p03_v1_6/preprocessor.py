from torchvision import transforms


TARGET_SIZE = (384, 384)


def get_transform():
    """
    Preprocessing used for Rolex P03 inference.
    """

    return transforms.Compose([
        transforms.Resize(
            TARGET_SIZE,
            interpolation=transforms.InterpolationMode.BICUBIC
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            (0.5, 0.5, 0.5),
            (0.5, 0.5, 0.5)
        ),
    ])