import torch
from torch import nn
import torch.nn.functional as F

class DinoClassifier(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.linear_head = nn.Linear(384, 1)

    def forward(self, x):
        x = self.dinov2_vits14(x)
        x = self.linear_head(x)
        x = F.sigmoid(x)
        return x
