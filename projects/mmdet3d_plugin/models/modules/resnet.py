# Copyright (c) Phigent Robotics. All rights reserved.

import torch.utils.checkpoint as checkpoint
from torch import nn
import torch
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
import torch.nn.functional as F
import pdb


@BACKBONES.register_module()
class CustomResNet(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
            block_type='Basic',
    ):
        super(CustomResNet, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        if block_type == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    Bottleneck(
                        curr_numC,
                        num_channels[i] // 4,
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    Bottleneck(curr_numC, curr_numC // 4, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        elif block_type == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [
                    BasicBlock(
                        curr_numC,
                        num_channels[i],
                        stride=stride[i],
                        downsample=nn.Conv2d(curr_numC, num_channels[i], 3,
                                             stride[i], 1),
                        norm_cfg=norm_cfg)
                ]
                curr_numC = num_channels[i]
                layer.extend([
                    BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                    for _ in range(num_layer[i] - 1)
                ])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        return feats


class BasicBlock3D(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None):
        super(BasicBlock3D, self).__init__()
        self.conv1 = ConvModule(
            channels_in,
            channels_out,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.conv2 = ConvModule(
            channels_out,
            channels_out,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=None)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.downsample is not None:
            identity = self.downsample(x)
        else:
            identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + identity
        return self.relu(x)


# @BACKBONES.register_module()
# class CustomResNet3D(nn.Module):

#     def __init__(
#             self,
#             numC_input,
#             num_layer=[2, 2, 2],
#             num_channels=None,
#             stride=[2, 2, 2],
#             # backbone_output_ids=None,
#             with_cp=False,
#     ):
#         super(CustomResNet3D, self).__init__()
#         # build backbone
#         assert len(num_layer) == len(stride)
#         num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
#             if num_channels is None else num_channels
#         # self.backbone_output_ids = range(len(num_layer)) \
#         #     if backbone_output_ids is None else backbone_output_ids
#         layers = []
#         curr_numC = numC_input
#         for i in range(len(num_layer)):
#             layer = [
#                 BasicBlock3D(
#                     curr_numC,
#                     num_channels[i],
#                     stride=stride[i],
#                     downsample=ConvModule(
#                         curr_numC,
#                         num_channels[i],
#                         kernel_size=3,
#                         stride=stride[i],
#                         padding=1,
#                         bias=False,
#                         conv_cfg=dict(type='Conv3d'),
#                         norm_cfg=dict(type='BN3d', ),
#                         act_cfg=None))
#             ]
#             curr_numC = num_channels[i]
#             layer.extend([
#                 BasicBlock3D(curr_numC, curr_numC)
#                 for _ in range(num_layer[i] - 1)
#             ])
#             layers.append(nn.Sequential(*layer))
#         self.layers = nn.Sequential(*layers)

#         self.with_cp = with_cp

#     def forward(self, x):
#         # feats = []
#         x_tmp = x
#         for lid, layer in enumerate(self.layers):
#             if self.with_cp:
#                 x_tmp = checkpoint.checkpoint(layer, x_tmp)
#             else:
#                 x_tmp = layer(x_tmp)
#             # if lid in self.backbone_output_ids:
#             #     feats.append(x_tmp)
#         return x_tmp
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        # Channel Attention
        x = self.channel_attention(x)
        # Spatial Attention
        x = self.spatial_attention(x)
        return x

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # Global Max Pooling
        
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Global Average Pooling
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        # Global Max Pooling
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        # Combine with Sigmoid
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise Average and Max Pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # Concatenate and Convolve
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

@BACKBONES.register_module()
class BottleneckConv3DWithCBAM(nn.Module):
    def __init__(self, channels, internal_channels):
        super(BottleneckConv3DWithCBAM, self).__init__()
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)
        
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)
        
        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)
        
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # Apply CBAM
        out = self.cbam(out)
        
        out += identity
        out = self.relu(out)
        return [out]
    
    
@BACKBONES.register_module()
class BottleneckConv3D(nn.Module):
    def __init__(self, channels, internal_channels):
        super(BottleneckConv3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)
        
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)
        
        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out += identity
        out = self.relu(out)
        return [out]


class OptimizedCBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(OptimizedCBAM3D, self).__init__()
        self.channel_attention = OptimizedChannelAttention3D(channels, reduction)
        self.spatial_attention = OptimizedSpatialAttention3D(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class OptimizedChannelAttention3D(nn.Module):
    def __init__(self, channels, reduction=16):
        super(OptimizedChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # Global Max Pooling
        
        # Reduction 비율 조정
        self.fc1 = nn.Conv3d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv3d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # 평균 풀링과 최대 풀링
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        
        # 두 결과를 더하여 Attention 가중치 생성
        out = avg_out + max_out
        return x * self.sigmoid(out)

class OptimizedSpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=3):
        super(OptimizedSpatialAttention3D, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise Pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Shape: [B, 1, D, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Shape: [B, 1, D, H, W]
        
        # Concatenate Average and Max Pooling Results
        out = torch.cat([avg_out, max_out], dim=1)  # Shape: [B, 2, D, H, W]
        
        # Reduce D dimension by averaging
        out = torch.mean(out, dim=2)  # Shape: [B, 2, H, W]
        
        # Apply Conv2D
        out = self.conv(out)  # Shape: [B, 1, H, W]
        return x * self.sigmoid(out).unsqueeze(2)  # Restore shape to [B, C, D, H, W]



@BACKBONES.register_module()
class BottleneckConv3DWithOptimizedCBAM(nn.Module):
    def __init__(self, channels, internal_channels):
        super(BottleneckConv3DWithOptimizedCBAM, self).__init__()
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)
        
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)
        
        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)
        
        self.cbam = OptimizedCBAM3D(channels)  # 최적화된 CBAM3D 적용
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        # Bottleneck 연산
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # CBAM3D 적용
        out = self.cbam(out)
        
        # Residual Connection
        out += identity
        out = self.relu(out)
        return [out]
    
@BACKBONES.register_module()
class BottleneckConv3DWithTriPerspectiveCBAM(nn.Module):
    def __init__(self, channels, internal_channels):
        super(BottleneckConv3DWithTriPerspectiveCBAM, self).__init__()
        self.conv1 = nn.Conv3d(channels, internal_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm3d(internal_channels)
        
        self.conv2 = nn.Conv3d(internal_channels, internal_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(internal_channels)
        
        self.conv3 = nn.Conv3d(internal_channels, channels, kernel_size=1)
        self.bn3 = nn.BatchNorm3d(channels)
        
        self.cbam = TriPerspectiveOptimizedCBAM3D(channels)  # 최적화된 CBAM3D 적용
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x
        
        # Bottleneck 연산
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        # CBAM3D 적용
        out = self.cbam(out)
        
        # Residual Connection
        out += identity
        out = self.relu(out)
        return [out]
    
    
class TriPerspectiveOptimizedCBAM3D(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super(TriPerspectiveOptimizedCBAM3D, self).__init__()
        self.channel_attention = OptimizedChannelAttention3D(channels, reduction)
        self.tri_perspective_spatial_attention = TriPerspectiveSpatialAttention3D(channels, kernel_size)
    
    def forward(self, x):
        # Step 1: Channel Attention
        x = self.channel_attention(x)

        # Step 2: Tri-perspective Spatial Attention
        x = self.tri_perspective_spatial_attention(x)

        return x

class TriPerspectiveSpatialAttention3D(nn.Module):
    def __init__(self, channels, kernel_size=3, embedding_dim=128):
        super(TriPerspectiveSpatialAttention3D, self).__init__()
        # Separate convolution layers for each perspective
        self.xy_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.yz_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.xz_conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        
        # Weight embedding for data-dependent weighting
        self.xy_embed = nn.Linear(channels, embedding_dim)
        self.yz_embed = nn.Linear(channels, embedding_dim)
        self.xz_embed = nn.Linear(channels, embedding_dim)
        self.fusion = nn.Linear(3 * embedding_dim, 3)  # Outputs weights for XY, YZ, XZ

        # Activation functions
        self.sigmoid = nn.Sigmoid()
        
    def compute_attention(self, view, conv):
        avg_out = torch.mean(view, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(view, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        return self.sigmoid(conv(combined))  # [B, 1, H, W]
        
    def forward(self, x):
        # Input: [B, C, D, H, W]
        B, C, D, H, W = x.shape

        # Step 1: Extract XY, YZ, XZ views
        xy_view = x.mean(dim=2)  # Collapse D -> [B, C, H, W]
        yz_view = x.mean(dim=4).permute(0, 1, 3, 2)  # Collapse W -> [B, C, D, H] -> [B, C, H, D]
        xz_view = x.mean(dim=3).permute(0, 1, 3, 2)  # Collapse H -> [B, C, D, W] -> [B, C, W, D]
    
        # Step 2: Channel-wise pooling for each perspective
        xy_attention = self.compute_attention(xy_view, self.xy_conv)  # [B, 1, H, W]
        yz_attention = self.compute_attention(yz_view, self.yz_conv)  # [B, 1, H, D]
        xz_attention = self.compute_attention(xz_view, self.xz_conv)  # [B, 1, W, D]

        # Step 3: Compute data-dependent weights
        xy_mean = xy_view.mean(dim=(-2, -1))  # [B, C]
        yz_mean = yz_view.mean(dim=(-2, -1))  # [B, C]
        xz_mean = xz_view.mean(dim=(-2, -1))  # [B, C]

        xy_embed = self.xy_embed(xy_mean)  # [B, embedding_dim]
        yz_embed = self.yz_embed(yz_mean)  # [B, embedding_dim]
        xz_embed = self.xz_embed(xz_mean)  # [B, embedding_dim]

        combined_embed = torch.cat([xy_embed, yz_embed, xz_embed], dim=-1)  # [B, 3 * embedding_dim]
        weights = self.sigmoid(self.fusion(combined_embed))  # [B, 3]

        # Step 4: Apply weights to attentions
        weighted_xy = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1) * xy_attention  # [B, 1, H, W]
        weighted_yz = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1) * yz_attention  # [B, 1, H, D]
        weighted_xz = weights[:, 2:3].unsqueeze(-1).unsqueeze(-1) * xz_attention  # [B, 1, W, D]

        # Step 5: Expand dimensions to match input shape
        weighted_xy = weighted_xy.unsqueeze(2)  # [B, 1, 1, H, W]
        weighted_yz = weighted_yz.permute(0, 1, 3, 2).unsqueeze(4)  # [B, 1, D, H, 1]
        weighted_xz = weighted_xz.permute(0, 1, 3, 2).unsqueeze(3)  # [B, 1, D, 1, W]

        # Step 6: Combine attentions
        combined_attention = (weighted_xy + weighted_yz + weighted_xz) / 3  # [B, 1, D, H, W]

        # Step 7: Apply attention to input
        return x * combined_attention  # [B, C, D, H, W]

@BACKBONES.register_module()
class CustomResNet3D(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            kernel_size=3,
            padding=1,
            with_cp=False,
            size = None,
            return_input_feat = False
    ):
        super(CustomResNet3D, self).__init__()
        # build backbone
        self.return_input_feat = return_input_feat
        assert len(num_layer) == len(stride)
        self.size = size
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        for i in range(len(num_layer)):
            layer = [
                BasicBlock3D(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    downsample=ConvModule(
                        curr_numC,
                        num_channels[i],
                        kernel_size=kernel_size,
                        stride=stride[i],
                        padding=padding,
                        bias=False,
                        conv_cfg=dict(type='Conv3d'),
                        norm_cfg=dict(type='BN3d', ),
                        act_cfg=None))
            ]
            curr_numC = num_channels[i]
            layer.extend([
                BasicBlock3D(curr_numC, curr_numC)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp

    def forward(self, x, checksize=False):
        feats = []
        b, c, w, h, z = x.shape
        if checksize and self.size is not None:
            if self.size[1] != h or self.size[2] != w:
                x = F.interpolate(x, size=self.size, mode='trilinear', align_corners=True)
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
        if self.return_input_feat:
            feats = [x] + feats
        return feats