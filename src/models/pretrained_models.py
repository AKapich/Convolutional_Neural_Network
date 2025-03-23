import torch.nn as nn
from torchvision.models import vgg16, resnet18, ResNet18_Weights
from torch import Tensor


class VGG16Pretrained(nn.Module):
    def __init__(self, num_classes=10, device="cpu"):
        super(VGG16Pretrained, self).__init__()
        self.device = device
        self.model = vgg16(pretrained=True)

        for param in self.model.features.parameters():
            param.requires_grad = False
        num_in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_in_features, num_classes)
        self.model = self.model.to(device)

    def forward(self, x):
        return self.model(x)


class ResNetPretrained(nn.Module):
    """
    Original ResNet is tailored for 224x224 images (CIFAR10 images ought to be rescaled)
    """

    def __init__(
        self, num_classes: int = 10, device: str = "cpu", fine_tune: bool = True
    ) -> None:
        super(ResNetPretrained, self).__init__()
        self.device = device
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        if fine_tune:
            for layer in [self.model.layer4, self.model.fc]:
                for param in layer.parameters():
                    param.requires_grad = True

        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_in_features, num_classes)
        self.model = self.model.to(device)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)


class ResNet32x32(nn.Module):
    pass


class ResNet_ResidualBlock:
    pass
