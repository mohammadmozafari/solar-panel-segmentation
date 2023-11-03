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

    def __init__(self, layers=1, mode='small', stacked_layers=1) -> None:
        super().__init__()
        
        self.layers = layers
        self.stacked_layers = stacked_layers
        
        if mode == 'small':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 384, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 384, 100)
                self.linear_head2 = nn.Linear(100, 2)
        
        elif mode == 'base':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 768, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 768, 200)
                self.linear_head2 = nn.Linear(200, 2)
            
        elif mode == 'giant':
            self.dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')
            if stacked_layers == 1:
                self.linear_head = nn.Linear((1 + layers) * 1536, 2)
            elif stacked_layers == 2:
                self.linear_head = nn.Linear((1 + layers) * 1536, 300)
                self.linear_head2 = nn.Linear(300, 2)

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
        
        x = self.linear_head(linear_input)
        if self.stacked_layers == 2: x = F.relu(self.linear_head2(x))
        
        return x