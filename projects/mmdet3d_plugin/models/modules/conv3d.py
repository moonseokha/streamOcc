import torch
from torch import nn
# from ..builder import BACKBONES, build_backbone
from mmdet.models import BACKBONES
# from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmcv.cnn import xavier_init, constant_init
from .utils import init_weights
import pdb

@BACKBONES.register_module()
class conv3d_block(nn.Module):
    
    def __init__(self,C1,C2):
        super(conv3d_block, self).__init__()
        self.conv3d1 = nn.Conv3d(C1,C2,1)
        self.bn1 = nn.BatchNorm3d(C2, eps=1e-05, momentum=0.1, affine=True)
        self.conv3d2 = nn.Conv3d(C2,C2,3,1,1)
        self.bn2 = nn.BatchNorm3d(C2, eps=1e-05, momentum=0.1, affine=True)
        self.conv3d3 = nn.Conv3d(C2,C1,1)
        self.bn3 = nn.BatchNorm3d(C1, eps=1e-05, momentum=0.1, affine=True)
        self.relu = nn.ReLU()

        # self.apply(init_weights)
        
        
    def forward(self,x):
        out = self.conv3d1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv3d2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3d3(out)
        out = self.bn3(out)
        out = out + x
        out = self.relu(out)
        return out

@BACKBONES.register_module
class myConv3d(nn.Module):

    def __init__(self,C1,C2,layers):
        super(myConv3d, self).__init__()
        self.layers = layers
        self.blocks = nn.ModuleList(
            [
                conv3d_block(C1,C2) for _ in range(self.layers)
            ]
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        # x = self.block3(x)
        return x
