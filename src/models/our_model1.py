import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, channels, filters_br1, filters_red_br2, filters_br2, filters_red_br3, filters_br3, filters_br4, squeeze=16, batch_norm_mom=0.1, se_enabled=True):
        super(Inception, self).__init__()
        self.se_enabled = se_enabled

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, filters_br1, kernel_size=1),
            nn.BatchNorm2d(filters_br1, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, filters_red_br2, kernel_size=1),
            nn.BatchNorm2d(filters_red_br2, momentum=batch_norm_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_red_br2, filters_br2, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_br2, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, filters_red_br3, kernel_size=1),
            nn.BatchNorm2d(filters_red_br3, momentum=batch_norm_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_red_br3, filters_br3, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_br3, momentum=batch_norm_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(filters_br3, filters_br3, kernel_size=3, padding=1),
            nn.BatchNorm2d(filters_br3, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(channels, filters_br4, kernel_size=1),
            nn.BatchNorm2d(filters_br4, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.se = SqueezeExcitationBlock(filters_br1 + filters_br2 + filters_br3 + filters_br4, squeeze=squeeze)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        x = torch.cat([b1, b2, b3, b4], dim=1)
        if self.se:
            return self.se(x)
        return torch.cat([b1, b2, b3, b4], dim=1)

class OurModel(nn.Module):
    def __init__(self, dropout=0.4, se_squeeze=16, batch_norm_mom=0.1, se_enabled=True, aux_enabled=True):
        super(OurModel, self).__init__()
        self.aux_enabled = aux_enabled
        self.se_enabled = se_enabled
        
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(48, momentum=batch_norm_mom),
        #     nn.ReLU(inplace=True)
        # )

        # self.inception1 = Inception(48, 24, 24, 48, 16, 48, 16, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        # # output channels = 24 + 48 + 48 + 16 = 136

        # self.inception2 = Inception(136, 48, 32, 64, 24, 64, 32, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        # # output channels = 48 + 64 + 64 + 32 = 208

        # self.aux = AuxiliaryClassifier(208, 10, dropout=dropout, batch_norm_mom=batch_norm_mom)

        # self.inception3 = Inception(208, 64, 48, 96, 32, 96, 48, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        # # output channels = 64 + 96 + 96 + 48 = 304

        # self.inception4 = Inception(304, 96, 64, 128, 48, 128, 64, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        # # output channels = 96 + 128 + 128 + 64 = 416
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(416, 10)
        # self.maxpool = nn.MaxPool2d(2, 2)
        # self.dropout = nn.Dropout(dropout)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batch_norm_mom),
            nn.ReLU(inplace=True))

        self.inception1 = Inception(64, filters_br1=16, filters_red_br2=12, filters_br2=32, filters_red_br3=12, filters_br3=32, filters_br4=16,
            squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.inception2 = Inception(96, filters_br1=48, filters_red_br2=12, filters_br2=72, filters_red_br3=12, filters_br3=72, filters_br4=12,
            squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.aux = AuxiliaryClassifier(204, 10, dropout=dropout, batch_norm_mom=batch_norm_mom)
        
        self.inception3 = Inception(204, filters_br1=96, filters_red_br2=12, filters_br2=144,filters_red_br3=12, filters_br3=144, filters_br4=36,
            squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.inception4 = Inception(420,filters_br1=252, filters_red_br2=12, filters_br2=252, filters_red_br3=12, filters_br3=252, filters_br4=132,
            squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(888, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.maxpool(x)
        x = self.inception2(x)
        x = self.maxpool(x)

        if self.aux_enabled:
            aux_out = self.aux(x)

        x = self.inception3(x)
        x = self.inception4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)

        if self.aux_enabled:
            return x, aux_out
        else:
            return x

class AuxiliaryClassifier(nn.Module):
    def __init__(self, channels, classes, dropout=0.5, batch_norm_mom=0.1):
        super(AuxiliaryClassifier, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(channels, 128, kernel_size=1)
        self.batch_norm = nn.BatchNorm2d(128, momentum=batch_norm_mom)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, classes)

    def forward(self, x):
        x = self.avg_pool(x)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
class SqueezeExcitationBlock(nn.Module):
    def __init__(self, channels, squeeze=16):
        super(SqueezeExcitationBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // squeeze)
        self.fc2 = nn.Linear(channels // squeeze, channels)

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        y = F.adaptive_avg_pool2d(x, 1)
        y = y.view(batch_size, channels)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        y = y.view(batch_size, channels, 1, 1)
        return x * y.expand_as(x)

class OurModelSmall(OurModel):
    def __init__(self, dropout=0.4, se_squeeze=16, batch_norm_mom=0.1, se_enabled=True, aux_enabled=True):
        super(OurModel, self).__init__()
        
        self.se_enabled = se_enabled
        self.aux_enabled = aux_enabled

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.inception1 = Inception(32, 16, 16, 16, 8, 8, 8, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        self.inception2 = Inception(48, 32, 16, 32, 8, 16, 16, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.aux = AuxiliaryClassifier(96, 10, dropout=dropout, batch_norm_mom=batch_norm_mom)

        self.inception3 = Inception(96, 32, 24, 48, 12, 24, 16, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        self.inception4 = Inception(120, 40, 28, 56, 14, 32, 20, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(148, 10)
        
        
class OurModelBig(OurModel):
    def __init__(self, dropout=0.4, se_squeeze=16, batch_norm_mom=0.1, se_enabled=True, aux_enabled=True):
        super(OurModel, self).__init__()
        self.se_enabled = se_enabled
        self.aux_enabled = aux_enabled
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=batch_norm_mom),
            nn.ReLU(inplace=True)
        )

        self.inception1 = Inception(64, 32, 32, 64, 32, 64, 32, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        self.inception2 = Inception(192, 64, 64, 128, 64, 128, 64, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)

        self.aux = AuxiliaryClassifier(384, 10, dropout=dropout, batch_norm_mom=batch_norm_mom)


        self.inception3 = Inception(384, 128, 128, 256, 128, 256, 128, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom, se_enabled=se_enabled)
        self.inception4 = Inception(768, 192, 192, 384, 192, 384, 192, squeeze=se_squeeze, batch_norm_mom=batch_norm_mom,se_enabled=se_enabled)

        self.dropout = nn.Dropout(dropout)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1152, 10)