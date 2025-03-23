from torchvision import transforms


def resize_transform(resize_dim: tuple[int, int] = (224, 224)) -> transforms:
    return transforms.Compose([transforms.Resize(resize_dim), transforms.ToTensor()])


def simple_transform() -> transforms:
    return transforms.Compose([transforms.ToTensor()])


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
