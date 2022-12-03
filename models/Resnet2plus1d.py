import torch 
import torch.nn as nn
import json as js
import urllib as ulib
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo
)
import torch.nn.functional as F
import torchvision.models as models
import math


class r2plus1d_18(nn.Module):
    def __init__(self, pretrained=True, num_classes=500):
        super(r2plus1d_18, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = models.video.r2plus1d_18(pretrained=self.pretrained)
        # delete the last fc layer
        modules = list(model.children())[:-1]
        # print(modules)
        self.r2plus1d_18 = nn.Sequential(*modules)
        self.fc1 = nn.Linear(model.fc.in_features, self.num_classes)

    def forward(self, x):
        out = self.r2plus1d_18(x)
        # print(out.shape)
        # Flatten the layer to fc
        out = out.flatten(1)
        out = self.fc1(out)

        return out


