import torch
from torch import nn
import torch.nn.functional as F

# class DinoClassifier(nn.Module):

#     def __init__(self, mode='small') -> None:
#         super().__init__()
        
#         if mode == 'small':
#             self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
#             self.linear_head = nn.Linear(384, 1)
        
#         elif mode == 'base':
#             self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
#             self.linear_head = nn.Linear(768, 1)

#     def forward(self, x):
#         x = self.dinov2(x)
#         x = self.linear_head(x)
#         x = F.sigmoid(x)
#         return x

class DinoClassifier(nn.Module):

    def __init__(self, layers=1, mode='small') -> None:
        super().__init__()
        
        self.layers = layers
        if mode == 'small':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            self.linear_head = nn.Linear((1 + layers) * 384, 2)
        
        elif mode == 'base':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            self.linear_head = nn.Linear((1 + layers) * 768, 2)

    def forward(self, x):
        
        if self.layers == 1:
            x = self.dinov2.forward_features(x)
            cls_token = x["x_norm_clstoken"]
            patch_tokens = x["x_norm_patchtokens"]
            # fmt: off
            linear_input = torch.cat([
                cls_token,
                patch_tokens.mean(dim=1),
            ], dim=1)
            # fmt: on
        elif self.layers == 4:
            x = self.dinov2.get_intermediate_layers(x, n=4, return_class_token=True)
            # fmt: off
            linear_input = torch.cat([
                x[0][1],
                x[1][1],
                x[2][1],
                x[3][1],
                x[3][0].mean(dim=1),
            ], dim=1)
            # fmt: on
        else:
            assert False, f"Unsupported number of layers: {self.layers}"
        return self.linear_head(linear_input)