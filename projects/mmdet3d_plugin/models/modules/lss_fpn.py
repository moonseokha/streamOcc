# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn
from mmcv.cnn import build_norm_layer
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .resnet import ConvModule
from mmdet.models import NECKS


@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False):
        super().__init__()
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x,None
    

@NECKS.register_module()
class LSSFPN3D_small(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_layers=3,
                 size=None,
                 with_cp=False):
        super().__init__()
        if size is None:
            self.up1 =  nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.size = None
        else:
            self.size = size

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        old_x_8, old_x_16, old_x_32 = x_8, x_16, x_32
        if self.size is not None:
            x_8 = F.interpolate(x_8, size=self.size,
                            mode='trilinear', align_corners=True)
            x_16 = F.interpolate(x_16, size=self.size,
                                    mode='trilinear', align_corners=True)
            x_32 = F.interpolate(x_32, size=self.size,
                                    mode='trilinear', align_corners=True)
        else:
            x_16 = self.up1(x_16)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x,(old_x_8, old_x_16, old_x_32)
    
@NECKS.register_module()
class LSSFPN3D_small_v2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 input_layers=3,
                 size=None,
                 with_cp=False):
        super().__init__()
        if size is None:
            self.up1 =  nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.size = None
        else:
            self.size = size

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16 = feats
        old_x_8, old_x_16 = x_8, x_16
        if self.size is not None:
            x_8 = F.interpolate(x_8, size=self.size,
                            mode='trilinear', align_corners=True)
            x_16 = F.interpolate(x_16, size=self.size,
                                    mode='trilinear', align_corners=True)
        else:
            x_16 = self.up1(x_16)
        x = torch.cat([x_8, x_16], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x,(old_x_8, old_x_16)
        
@NECKS.register_module()
class LSSFPN3D_COTR(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 reverse=False,
                 size=(16, 50, 50)):
        super().__init__()
        self.reverse = reverse
        self.size = size
        if not reverse:
            self.up1 =  nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
            self.up2 =  nn.Upsample(
                scale_factor=4, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        x_8, x_16, x_32 = feats
        old_x_8 = x_8
        old_x_16 = x_16
        # print(f"x_8.shape:{x_8.shape}")
        # print(f"x_16.shape:{x_16.shape}")
        # print(f"x_32.shape:{x_32.shape}")
        if not self.reverse:
            x_16 = self.up1(x_16)
            x_32 = self.up2(x_32)
        else:
            x_8 = F.interpolate(x_8, size=self.size,
                                 mode='trilinear', align_corners=True)
            x_16 = F.interpolate(x_16, size=self.size,
                                 mode='trilinear', align_corners=True)
            # x_32 = F.interpolate(x_32, size=(z, h, w),
            #                      mode='trilinear', align_corners=True)
        
        if x_32.shape[-3:] != x_8.shape[-3:]:
            x_32 = F.interpolate(x_32, size=x_8.shape[-3:], mode='trilinear')
        # print(f"x_8.shape:{x_8.shape}")
        # print(f"x_16.shape:{x_16.shape}")
        # print(f"x_32.shape:{x_32.shape}")
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        if self.reverse:
            return x, (old_x_8, old_x_16, x_32)
        return x, x
    
    
    
@NECKS.register_module()    
class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x,None