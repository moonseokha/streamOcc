# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License-NC.
# To view a copy of this license, visit
# https://github.com/NVlabs/VoxFormer/blob/main/LICENSE

# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

import numpy as np
import torch
import cv2 as cv
import mmcv
import copy
import warnings
import torch.nn as nn
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE,FEEDFORWARD_NETWORK)
from mmcv.cnn.bricks.transformer import TransformerLayerSequence,build_positional_encoding,FFN
from mmcv.runner import force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
# from projects.mmdet3d_plugin.models.utils.bricks import run_time
# from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer
import pdb
from ..blocks import DAF
from mmcv.cnn.bricks.registry import (
    PLUGIN_LAYERS)

ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class VoxFormer(TransformerLayerSequence):

    """
    Attention with both self and cross
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False,positional_encoding=None,dataset_type='nuscenes',use_mask_token=False,use_downsample=False,
                 down_layer=None,up_layer=None,Interaction_Net:dict=None, **kwargs):

        super(VoxFormer, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.use_mask_token = use_mask_token
        if use_mask_token:
            self.mask_embed = nn.Embedding(1, positional_encoding['num_feats']*2)
        self.embed_dim = positional_encoding['num_feats']*2
        self.use_downsample=use_downsample
        if Interaction_Net is not None:
            self.Interaction_Net = build_from_cfg(Interaction_Net, PLUGIN_LAYERS)
                        
        
    @staticmethod
    def get_reference_points(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
        """Get the reference points used in DCA and DSA.
        Args:
            H, W: spatial shape of bev.
            Z: hight of pillar.
            D: sample D points uniformly from each pillar.
            device (obj:`device`): The device where
                reference_points should be.
        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """

        # reference points in 3D space, used in spatial cross-attention (SCA)
        if dim == '3d':
            zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
                                device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
            xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
                                device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
            ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
                                device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
            ref_3d = torch.stack((xs, ys, zs), -1)
            ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
            ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
            return ref_3d

        # reference points on 2D bev plane, used in temporal self-attention (TSA).
        elif dim == '2d':
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_2d = torch.stack((ref_x, ref_y), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d
        elif dim == '3dCustom':
            ref_y, ref_x, ref_z = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
                torch.linspace(
                    0.5, Z - 0.5, Z, dtype=dtype, device=device)
            )
            ref_y = ref_y.reshape(-1)[None] / H
            ref_x = ref_x.reshape(-1)[None] / W
            ref_z = ref_z.reshape(-1)[None] / Z
            ref_2d = torch.stack((ref_x, ref_y, ref_z), -1)
            ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)
            return ref_2d 

    @force_fp32(apply_to=('voxel_feat',))
    def forward(self,
                voxel_feat=None,
                prev_bev=None,
                prev_metas=None,
                metas=None,
                img_feats=None,):
        """Forward function for `TransformerEncoder`.
        Args:
            voxel_feat (Tensor): Input Voxel Feature with shape
                `(bs, y, x ,z embed_dims)`.
          
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        B, Y, X, Z, C = voxel_feat.shape
        if self.use_mask_token:
            mask_token =  self.mask_embed.weight.view(1,1,1,1,C).expand(B,Y,X,Z,C).to(voxel_feat.dtype)
            voxel_feat = torch.where(voxel_feat == 0, mask_token, voxel_feat)
        spatial_shapes = torch.tensor([Y, X, Z], device=voxel_feat.device).to(dtype=torch.long)
        voxel_feat = voxel_feat.view(B,Y*X*Z,C).permute(1,0,2)  # (num_query, bs, embed_dims)
        output = voxel_feat
        voxel_pos = self.positional_encoding(torch.zeros((B,Y*Z,X), device=voxel_feat.device).to(voxel_feat.dtype)).to(voxel_feat.dtype) # [B, dim, 128*4, 128]
        voxel_pos = voxel_pos.flatten(2).permute(2, 0, 1) # [128*4*128, B, dim]
        intermediate = []
        ref_3d = self.get_reference_points(
            Y, X, Z, dim='3dCustom', bs=voxel_feat.size(1), device=voxel_feat.device, dtype=voxel_feat.dtype)
        # bs, len_bev, num_bev_level, _ = ref_3d.shape
        voxel_feat = voxel_feat.permute(1, 0, 2)
        bs, len_bev, _ = voxel_feat.shape

        voxel_pos = voxel_pos.permute(1, 0, 2) # [B,num_query, dim]
             
        for lid, layer in enumerate(self.layers):
            output,occ = layer(
                voxel_feat,
                bev_pos=voxel_pos,
                ref_3d=ref_3d,
                spatial_shapes=spatial_shapes)
            voxel_feat = output  # (bs, num_query, embed_dims)
            if self.return_intermediate:
                intermediate.append(output)
            # pdb.set_trace()
        if self.return_intermediate:
            return torch.stack(intermediate)
        output = output.view(B, Y, X, Z, C) # (bs, y, x, z, embed_dims)  
        return output


@TRANSFORMER_LAYER.register_module()
class VoxelFormerLayer3D(MyCustomBaseTransformerLayer):
    """Implements encoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 grid_size = [100.0,100.0,8.0],
                 grid_config = None,
                 img_to_voxel = False,
                 embed_dims=256,
                 num_groups=8,
                 num_cams=6,
                 num_levels=4,
                 num_points=6,
                 **kwargs):
        super(VoxelFormerLayer3D, self).__init__(
            attn_cfgs=attn_cfgs,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.num_points = num_points
        
        if grid_config is not None:
            self.bev_origin = [grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]]
            self.bev_resolution = [(grid_config['x'][1]-grid_config['x'][0])/grid_size[0], (grid_config['y'][1]-grid_config['y'][0])/grid_size[1], (grid_config['z'][1]-grid_config['z'][0])/grid_size[2]]  
            self.grid_size = grid_size
            
        self.img_to_voxel = img_to_voxel
        
        # if img_to_voxel:
        #     self.attention_weights = nn.Linear(
        #             embed_dims, num_groups * num_cams * num_levels * num_points
        #         )
        #     self.sampling_offsets = nn.Linear(
        #         embed_dims,  num_points * 3)
        #     self.out_proj = nn.Linear(embed_dims, embed_dims)
        #     self.proj_drop = nn.Dropout(0.1)
        # if ffn_occ_cfgs is not None:
        #     self.ffn_occ = build_from_cfg(ffn_occ_cfgs, FEEDFORWARD_NETWORK)
            
    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                mask=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward function for `TransformerEncoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        occ = None
        identity = query
        bs,_,_ = query.shape
        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':
                query = self.attentions[attn_index](
                    query,
                    value = query,
                    identity = identity if self.pre_norm else None,
                    query_pos=bev_pos,
                    # attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    reference_points=ref_3d,
                    spatial_shapes=spatial_shapes,
                    level_start_index=torch.tensor([0], device=query.device),
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1
            elif layer == 'cross':
                if self.Interaction_Net.without_occ:
                    query = self.Interaction_Net(
                        instance_feature,
                        query.reshape(bs,self.bev_resolution[0],self.bev_resolution[1],self.bev_resolution[2],self.embed_dims),
                        agent_pos,
                        agent_pos_embed,
                        indices,
                        metas,
                        img_feats,
                    )
                else:
                    query, occ = self.Interaction_Net(
                        instance_feature,
                        query.reshape(bs,self.bev_resolution[0],self.bev_resolution[1],self.bev_resolution[2],self.embed_dims),
                        agent_pos,
                        agent_pos_embed,
                        indices,
                        metas,
                        img_feats,
                    )
                query = query.reshape(bs,-1, self.embed_dims)
            #     if self.use_occ_loss and (norm_index == 1):
            #         occ = self.ffn_occ(query.clone())
            #         bs = query.shape[0]
            #         if self.img_to_voxel:
            #             occupancy_scores = occ.sigmoid()
            #             occupied_indices = occupancy_scores.topk(int(self.top_k_ratio*occupancy_scores.shape[1]),dim=1)[1]
            #             vox_z = occupied_indices % self.grid_size[2]
            #             vox_y = (occupied_indices / self.grid_size[2]) % self.grid_size[1]
            #             vox_x = (occupied_indices / self.grid_size[2] / self.grid_size[1])
            #             vox_pos = torch.stack([vox_x,vox_y,vox_z],dim=2).squeeze() #[bs, num_points, 3]
            #             occupied_query = torch.gather(query,1,occupied_indices.repeat(1,1,query.shape[2]))
            #             sampling_offsets = self.sampling_offsets(occupied_query)
            #             sampling_offsets = sampling_offsets.view(
            #                 bs, -1, self.num_points, 3)
            #             attention_weights = self.attention_weights(occupied_query).view(
            #                 bs, -1,  self.num_points*self.num_cams*self.num_levels * self.num_groups)
            #             attention_weights = attention_weights.softmax(-2).view(bs, -1, self.num_points, self.num_cams, self.num_levels, self.num_groups)
            #             offset_normalizer = torch.tensor([self.grid_size[1], self.grid_size[0],self.grid_size[2]]).to(query.device)

            #             if vox_pos.dim() == 2:
            #                 vox_pos = vox_pos.unsqueeze(0)
            #             sampling_locations = vox_pos[:, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :]
            #             bev_resolution = torch.tensor([self.bev_resolution[1], self.bev_resolution[0], self.bev_resolution[2]]).to(query.device)
            #             bev_origin = torch.tensor([self.bev_origin[1], self.bev_origin[0], self.bev_origin[2]]).to(query.device)
            #             key_points = sampling_locations * bev_resolution[None, None, None, :] + bev_origin[None, None, None, :]
            #             points_2d = (
            #                 self.project_points(
            #                     key_points, #[bs, num_anchor, num_pts, 3]
            #                     metas["projection_mat"],
            #                     metas.get("image_wh"),
            #                 )
            #                 .permute(0, 2, 3, 1, 4)
            #                 .reshape(bs, -1, self.num_points, self.num_cams, 2)
            #             )
            #             features = DAF(*img_feats, points_2d, attention_weights).reshape(
            #                 bs, -1, self.embed_dims
            #             )   
            #             features = self.proj_drop(self.out_proj(features))
            #             occupied_query = occupied_query + features
            #             query = torch.scatter(query,1,occupied_indices.repeat(1,1,query.shape[2]),occupied_query)
            #     # if self.use_occ_loss:
            #     #     occ = self.ffn_occ(query.clone())
                
            # # elif layer == 'cross_attn':
            #     occupancy_scores = query.sigmoid()
            #     occupied_indices = occupancy_scores.topk(self.top_k_ratio,dim=1)[1]
            #     points_2d = (
            #     self.project_points(
            #         key_points,
            #         metas["projection_mat"],
            #         metas.get("image_wh"),
            #     )
            #     .permute(0, 2, 3, 1, 4)
            #     .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            #     )   
            #     weights = (
            #         weights.permute(0, 1, 4, 2, 3, 5)
            #         .contiguous()
            #         .reshape(
            #             bs,
            #             num_anchor,
            #             self.num_pts,
            #             self.num_cams,
            #             self.num_levels,
            #             self.num_groups,
            #         )
            #     )

            #     features = DAF(*feature_maps, points_2d, weights).reshape(
            #         bs, num_anchor, self.embed_dims
            #     )
                
                    
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query,occ
    
    # @staticmethod
    # def project_points(key_points, projection_mat, image_wh=None):
    #     bs, num_anchor, num_pts = key_points.shape[:3]
    #     pts_extend = torch.cat(
    #         [key_points, torch.ones_like(key_points[..., :1])], dim=-1
    #     )
    #     points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
    #     points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
    #     if image_wh is not None:
    #         points_2d = points_2d / image_wh[:, :, None, None]
    #     return points_2d
                