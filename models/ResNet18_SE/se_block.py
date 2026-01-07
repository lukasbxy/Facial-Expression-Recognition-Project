'''
Implementation:
from models.attention.se_block import SEBlock 
...
in class BasicBlock __init__:
self.se = SEBlock(out_channels)

in class BasicBlock forward:
BEFORE shortcut:
out = self.se(out)

'''


import torch 
import torch.nn as nn
import torch.nn.functional as F 

class SEBlock(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super().__init__()
        num_channels_reduced = max(4, num_channels // reduction_ratio)
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias = True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias = True)

    def forward(self, input_tensor):

        batch_size, num_channels , H, W = input_tensor.size()
        squeeze_tensor = F.adaptive_avg_pool2d(input_tensor,1).view(batch_size, num_channels)


        fc_out_1 = F.relu(self.fc1(squeeze_tensor))
        fc_out_2 = torch.sigmoid(self.fc2(fc_out_1))


        output_tensor = torch.mul(input_tensor, fc_out_2.view(batch_size,num_channels,1,1))
        return output_tensor
