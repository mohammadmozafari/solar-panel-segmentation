import torch
from torch import nn

from .base import ResnetBase


class Classifier(ResnetBase):
    """A ResNet34 Model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """

    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__(imagenet_base=imagenet_base)

        # resnet34
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 2),
        )
        
        # swin_v2_t
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Linear(768, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        x = self.pretrained(x)
        x = self.avgpool(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x
