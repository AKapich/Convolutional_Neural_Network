import torch.nn as nn
import torchvision.models as models

class VGG16Pretrained(nn.Module):
    def __init__(self, num_classes=10, device='cuda'):
        super(VGG16Pretrained, self).__init__()
        self.device = device
        self.model = models.vgg16(pretrained=True)
        
        for param in self.model.features.parameters():
            param.requires_grad = False
        num_in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_in_features, num_classes)
        self.model = self.model.to(device)
    
    def forward(self, x):
        return self.model(x)

