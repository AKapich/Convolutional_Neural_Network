import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class VGG16(nn.Module):
    def __init__(self, num_classes=10, optimizer_type='adam', lr=0.001,
                 optimizer_params=None, device='cuda'):
        super(VGG16, self).__init__()
        self.device = device
        self.model = models.vgg16(pretrained=True)
        for param in self.model.features.parameters():
            param.requires_grad = False
            
        num_in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_in_features, num_classes)
        self.model = self.model.to(device)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        
        if optimizer_params is None:
            optimizer_params = {}
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, **optimizer_params)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(trainable_params, lr=lr, **optimizer_params)
        else:
            raise ValueError("Unsupported optimizer type: {}".format(optimizer_type))
    
    def forward(self, x):
        return self.model(x)

class GoogleNet(nn.Module):
    def __init__(self, num_classes=10, optimizer_type='adam', lr=0.001,
                 optimizer_params=None, device='cuda'):
        super(GoogleNet, self).__init__()
        self.device = device
        self.model = models.googlenet(pretrained=True, aux_logits=True)
        for param in self.model.parameters():
            param.requires_grad = False
            
        num_in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_in_features, num_classes)
        self.model = self.model.to(device)
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        if optimizer_params is None:
            optimizer_params = {}
            
        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(trainable_params, lr=lr, **optimizer_params)
        elif optimizer_type.lower() == 'sgd':
            self.optimizer = optim.SGD(trainable_params, lr=lr, **optimizer_params)
        else:
            raise ValueError("Unsupported optimizer type: {}".format(optimizer_type))
    
    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs
