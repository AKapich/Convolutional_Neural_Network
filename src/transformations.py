from torchvision import transforms
import torch
from PIL import Image
from math import sqrt


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


class RandomRotation(torch.nn.Module):
    def __init__(self, degrees=15):
        super().__init__()
        self.degrees = degrees
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
        if isinstance(image, torch.Tensor):
            image = self.to_pil(image)

        width, height = image.size
        diagonal = sqrt(width**2 + height**2)

        # Apply edge padding
        padding_w = int((diagonal - width) / 2) + 1
        padding_h = int((diagonal - height) / 2) + 1
        padding_transform = transforms.Pad(
            padding=(padding_w, padding_h, padding_w, padding_h), padding_mode="edge"
        )
        padded_image = padding_transform(image)

        # Apply random rotation to the padded image
        angle = torch.empty(1).uniform_(-self.degrees, self.degrees).item()
        rotated_image = padded_image.rotate(angle, resample=Image.BICUBIC, expand=False)

        # Crop back to original size from the center
        center_w, center_h = rotated_image.size[0] // 2, rotated_image.size[1] // 2
        left = center_w - width // 2
        top = center_h - height // 2
        right = left + width
        bottom = top + height

        cropped_image = rotated_image.crop((left, top, right, bottom))

        return self.to_tensor(cropped_image)


def random_rotation_transform(degrees: int = 15) -> transforms.Compose:
    return transforms.Compose([RandomRotation(degrees=degrees)])


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
