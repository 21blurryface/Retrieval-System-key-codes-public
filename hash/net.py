import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
alexnet_model = models.alexnet(pretrained=True)

class SE(nn.Module):

    def __init__(self, in_chnls, ratio):
        super(SE, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = self.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)

class ReIDSE(nn.Module):
    def __init__(self, bits):
        super(ReIDSE, self).__init__()
        self.bits = bits
        self.features = nn.Sequential(*list(alexnet_model.features.children()))
        self.remain = nn.Sequential(*list(alexnet_model.classifier.children())[:-1])
        self.Linear1 = nn.Linear(4096, self.bits)
        self.sigmoid = nn.Sigmoid()
        #self.Linear2 = nn.Linear(self.bits, 10)
        self.Linear2 = nn.Linear(self.bits, 751)
        #se block
        self.se = SE(256, 16)

    def forward(self, x):
        x = self.features(x)
        coefficient = self.se(x)
        x = coefficient*x

        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.remain(x)
        x = self.Linear1(x)
        features = self.sigmoid(x)
        result = self.Linear2(features)
        return features, result

