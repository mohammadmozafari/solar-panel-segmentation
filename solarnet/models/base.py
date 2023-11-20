import torch
from torch import nn
from torchvision.models import resnet34, resnet50, resnet101
from torchvision.models import swin_v2_t


class ResnetBase(nn.Module):
    """ResNet pretrained on Imagenet. This serves as the
    base for the classifier, and subsequently the segmentation model

    Attributes:
        imagenet_base: boolean, default: True
            Whether or not to load weights pretrained on imagenet
    """
    def __init__(self, imagenet_base: bool = True) -> None:
        super().__init__()

        # resnet 34, 50, 101
        # resnet 100
        # base = resnet34(pretrained=imagenet_base).float()
        base = resnet50(pretrained=imagenet_base).float()
        # resnet = resnet101(pretrained=imagenet_base).float()
        self.pretrained = nn.Sequential(*list(base.children())[:-2])
        
        # vision transformers
        # base = swin_v2_t(pretrained=imagenet_base).float()
        # self.pretrained = nn.Sequential(*list(base.children())[:-3])

    def forward(self, x):
        # Since this is just a base, forward() shouldn't directly
        # be called on it.
        raise NotImplementedError
