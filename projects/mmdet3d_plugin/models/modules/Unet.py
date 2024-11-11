import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
import time
import copy
import numpy as np
import pdb
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence


@PLUGIN_LAYERS.register_module()
class Unet_DAP(nn.Module):
    def __init__(self,
                 input_dimensions=None,
                 hidden_dimensions=None,
                 out_scale="1_1",
                 voxel_transformer=None,
                 ):

        super(Unet_DAP,
              self).__init__()
        '''
        SSCNet architecture
        :param N: number of classes to be predicted (i.e. 12 for NYUv2)
        '''

        super().__init__()
        self.out_scale=out_scale
        
        f = input_dimensions
        hidden_f = hidden_dimensions

        # self.pool = nn.MaxPool2d(2)  # [F=2; S=2; P=0; D=1]

        self.Encoder_block1 = nn.Sequential(
        nn.Conv3d(f, f, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(f, f, kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
        )

        self.Encoder_block2 = nn.Sequential(
        nn.MaxPool3d(2),
        nn.Conv3d(f,int(f*2), kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f*2, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(int(f*2),int(f*2), kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f*2, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
        )

        self.Encoder_block3 = nn.Sequential(
        nn.MaxPool3d(2),
        nn.Conv3d(int(f*2),int(f*4), kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f*4, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True),
        nn.Conv3d(int(f*4),int(f*4), kernel_size=3, padding=1, stride=1),
        nn.BatchNorm3d(f*4, eps=1e-05, momentum=0.1, affine=True),
        nn.ReLU(inplace=True)
        )  # [128,50,50,4]
        
        self.Encoder_block4 = nn.Sequential(
          nn.Conv3d(hidden_f-f,hidden_f, kernel_size=3, padding=1, stride=1),
          nn.BatchNorm3d(hidden_f, eps=1e-05, momentum=0.1, affine=True),
          nn.ReLU(inplace=True),
        #   nn.Conv3d(hidden_f,hidden_f, kernel_size=3, padding=1, stride=1),
        #   nn.BatchNorm3d(hidden_f, eps=1e-05, momentum=0.1, affine=True),
        #   nn.ReLU()
        )
        
        
        # if self.out_scale=="1_4" or self.out_scale=="1_2" or self.out_scale=="1_1":
        #   # self.deconv1_8          = nn.ConvTranspose2d(int(f/8), int(f/8), kernel_size=6, padding=2, stride=2)
        #   # self.conv1_4            = nn.Conv2d(int(f*2) + int(f/8), int(f*2), kernel_size=3, padding=1, stride=1)
        #   self.conv_out_scale_1_4 = nn.Conv2d(half_f, int(f/4), kernel_size=3, padding=1, stride=1)
        #   self.deconv_1_4__1_1    = nn.ConvTranspose2d(int(f/4), int(f/4), kernel_size=4, padding=0, stride=4)

        # # Treatment output 1:2
        # if self.out_scale=="1_2" or self.out_scale=="1_1":
        #   self.deconv1_4          = nn.ConvTranspose2d(int(f/4),int(f/4), kernel_size=6, padding=2, stride=2)
        #   self.conv1_2            = nn.Conv2d(half_f + int(f/4), half_f, kernel_size=3, padding=1, stride=1)
        #   self.conv_out_scale_1_2 = nn.Conv2d(half_f, int(f/2), kernel_size=3, padding=1, stride=1)

        # # Treatment output 1:1
        # if self.out_scale=="1_1":
        #   self.deconv1_2          = nn.ConvTranspose2d(int(f/2),int(f/2), kernel_size=6, padding=2, stride=2)
        #   self.conv1_1            = nn.Conv2d(int(f/2) + int(f/4) + half_f, f, kernel_size=3, padding=1, stride=1)
        # self.voxel_transformer = build_transformer_layer_sequence(voxel_transformer)
        
        # Treatment output 1:2
        # if self.out_scale=="1_2" or self.out_scale=="1_1":
        #   self.conv1_2            = nn.Conv3d(hidden_f + int(f*2), hidden_f, kernel_size=3, padding=1, stride=1)

        # # Treatment output 1:1
        # if self.out_scale=="1_1":
        #   self.conv1_1            = nn.Conv3d(f + hidden_f, hidden_f, kernel_size=3, padding=1, stride=1)
        
        self.voxel_transformer = build_transformer_layer_sequence(voxel_transformer)
        
        
    def forward(self, vox_feat,prev_vox_feat=None,prev_metas=None,metas=None,img_feats=None,):
        B,H,W,D,C = vox_feat.shape
        vox_feat = vox_feat.permute(0,4,1,2,3)
        # Encoder block
        skip_1_1 = self.Encoder_block1(vox_feat)
        skip_1_2 = self.Encoder_block2(skip_1_1)
        skip_1_4 = self.Encoder_block3(skip_1_2) # [B,C*2,H/4,W/4]
        
        b,c,h,w,d = skip_1_4.shape
        cat_feature  = torch.cat((
            F.interpolate(skip_1_4, size=(h,w,d*4), mode='trilinear', align_corners=False),
            F.interpolate(skip_1_2, size=(h,w,d*4), mode='trilinear', align_corners=False),
            F.interpolate(skip_1_1, size=(h,w,d*4), mode='trilinear', align_corners=False),
            # F.interpolate(vox_feat.clone(), size=(h,w,d*4), mode='trilinear', align_corners=False)
        ),dim=1)
        voxel_feat = self.Encoder_block4(cat_feature).permute(0,2,3,4,1).contiguous()
        voxel_feat,occ_list = self.voxel_transformer(voxel_feat,prev_vox_feat,prev_metas,metas,img_feats=img_feats)
        voxel_feature = voxel_feat.clone().detach()
        
        
        
        # voxel_feat = voxel_feat.permute(0,4,1,2,3)
        # # # Decoder 
        # b,c,h,w,d = voxel_feat.shape

        # h ,w = skip_1_2.shape[2:4]

        # out = torch.cat((F.interpolate(voxel_feat,size=(h,w,d),mode='trilinear',align_corners=False), F.interpolate(skip_1_2,size=(h,w,d),mode='trilinear',align_corners=False)), 1)
        # out = F.relu(self.conv1_2(out)) # torch.Size([1, 48, 128, 128])

        # h,w = skip_1_1.shape[2:4]
        # out = torch.cat((F.interpolate(out,size=(h,w,d),mode='trilinear',align_corners=False), F.interpolate(skip_1_1,size=(h,w,d),mode='trilinear',align_corners=False)), 1)
        # out_scale_1_1__2D = F.relu(self.conv1_1(out)) # [bs, c, h, w, d]

        # out_scale_1_1__2D = out_scale_1_1__2D.permute(0,2,3,4,1).contiguous()
        return voxel_feat,voxel_feature,occ_list
