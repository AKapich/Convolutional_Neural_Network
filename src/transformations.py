from torchvision import transforms


def resize_transform(resize_dim: tuple[int, int] = (224, 224)) -> transforms:
    return transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor()])


def simple_transform() -> transforms:
    return transforms.Compose([transforms.ToTensor()])


def normalized_simple_transform() -> transforms:
    return transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4789, 0.4723, 0.4305], std=[0.1998, 0.1965, 0.1997]
            ),
        ]
    )


def pretrained_transform() -> transforms:
    """
    Transformations to use in pretrained vgg16 and googlenet.
    """
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4789, 0.4723, 0.4305], std=[0.1998, 0.1965, 0.1997]
            ),
        ]
    )


# Data Augmentation Transforms


def random_rotation_transform(degrees: int = 15) -> transforms:
    return transforms.Compose(
        [transforms.RandomRotation(degrees=degrees), transforms.ToTensor()]
    )


def horizontal_flip_transform(p: float = 0.5) -> transforms:
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(p=p), transforms.ToTensor()]
    )


def color_jitter_transform(
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> transforms:

    return transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=brightness, contrast=contrast, saturation=saturation, hue=hue
            ),
            transforms.ToTensor(),
        ]
    )


def combined_augmentation_transform(
    degrees: int = 15,
    p: float = 0.5,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1,
) -> transforms:
    return transforms.Compose(
        [
            transforms.RandomRotation(degrees=degrees),
            transforms.RandomHorizontalFlip(p=p),
            transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
                hue=hue,
            ),
            transforms.ToTensor(),
        ]
    )
