# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    FEEDFORWARD_NETWORK,
)
import pdb

try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = [
    "DeformableFeatureAggregation",
    "DenseDepthNet",
    "AsymmetricFFN",
    "SimpleFFN",
    "OccFFN",
]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        attn_cfgs: dict = None,
        kps_generator: dict = None,
        # voxel_kps_generator: dict = None,
        temporal_fusion_module=None,
        use_voxel_feature: bool = False,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        DA3D_fixed_point = False,
        residual_mode="add",
        use_DFA3D = False,
        DFA3D_cfg: dict = None,
    ):
        super(DeformableFeatureAggregation, self).__init__()
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )
        self.use_voxel_feature = use_voxel_feature
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_DFA3D = use_DFA3D
        if use_DFA3D:
            self.DFA3D = build_from_cfg(DFA3D_cfg, ATTENTION)
        self.use_temporal_anchor_embed = use_temporal_anchor_embed
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.DA3D_fixed_point = DA3D_fixed_point
        if use_deformable_func or use_DFA3D:
            self.proj_drop = nn.Dropout(proj_drop)
        if 'fix_scale' in kps_generator: 
            self.fix_points_num = len(kps_generator['fix_scale'])
        else:  
            self.fix_points_num = 1
        self.learnable_pts_num = kps_generator['num_learnable_pts']
        kps_generator["embed_dims"] = embed_dims
        
        self.num_pts = kps_generator['num_learnable_pts']
        if use_deformable_func:
            self.num_pts+=self.fix_points_num
        if attn_cfgs is not None and use_voxel_feature:
            # self.kps_generator_voxel = build_from_cfg(voxel_kps_generator, PLUGIN_LAYERS)
            self.DA3D = build_from_cfg(attn_cfgs, ATTENTION)  # 3D deformable attention
            if DA3D_fixed_point:
                self.voxel_weights_fc = Linear(embed_dims, num_groups * self.num_pts)
            else:
                self.voxel_weights_fc = Linear(embed_dims, num_groups * self.num_pts)
            if 'learnable_pts_all_num' not in kps_generator:
                if use_deformable_func:
                    kps_generator.update({'learnable_pts_all_num':(num_groups+1)*self.num_pts-self.fix_points_num})
                else:
                    kps_generator.update({'learnable_pts_all_num':(num_groups)*self.num_pts-self.fix_points_num})
        # else:
        if use_deformable_func or use_DFA3D:
            if use_DFA3D:
                kps_generator.update({'learnable_pts_all_num':(num_groups)*self.num_pts-self.fix_points_num})
            if use_camera_embed:
                self.camera_encoder = Sequential(
                    *linear_relu_ln(embed_dims, 1, 2, 12)
                )
                self.weights_fc = Linear(
                    embed_dims, num_groups * num_levels * self.num_pts
                )
            else:
                self.camera_encoder = None
                self.weights_fc = Linear(
                    embed_dims, num_groups * num_cams * num_levels * self.num_pts
                )
            self.output_proj = Linear(embed_dims, embed_dims)

        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)

        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(
                temporal_fusion_module, PLUGIN_LAYERS
            )
        else:
            self.temp_module = None

      
        
    def init_weight(self):
        if self.use_voxel_feature:
            constant_init(self.voxel_weights_fc, val=0.0, bias=0.0)
        if self.use_deformable_func:
            constant_init(self.weights_fc, val=0.0, bias=0.0)
            xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: torch.Tensor, #   [B, - ,C ]or [ N_cams, H*W*level, B, C]
        voxel_feature_maps: torch.Tensor, 
        metas: dict,
        depth = None, #[N_cams, H*W*level, B, D]
        spatial_shapes = None,
        level_start_index = None,
        **kwargs: dict,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)
        if self.use_deformable_func or self.use_DFA3D:
            weights = self._get_weights(instance_feature, anchor_embed, metas)

        if self.use_voxel_feature:
            bs, V_y, V_x, V_z, D = voxel_feature_maps.shape
            spatial_shapes=torch.tensor(
                    [[V_y,V_x, V_z]], device=voxel_feature_maps.device)
            # key_points_voxel = key_points
            
            # fixed_key_points = key_points[:,:,:self.fix_points_num].clone()
            # if self.use_deformable_func:
            #     learnable_key_points_voxel = key_points[:,:,self.num_pts:]
            # else:
            #     learnable_key_points_voxel = key_points[:,:,self.fix_points_num:]
        
            # learnable_key_points_voxel = learnable_key_points_voxel.reshape(bs, -1 ,self.num_groups, self.num_pts-self.fix_points_num, 3)
            # fixed_key_points_voxel = fixed_key_points.unsqueeze(2).repeat(1,1,self.num_groups,1,1)
            # if self.DA3D_fixed_point:
            #     key_points_voxel = torch.cat([fixed_key_points_voxel, learnable_key_points_voxel], dim=3)
            # else:
            #     key_points_voxel = learnable_key_points_voxel
            # key_points_voxel = key_points_voxel.unsqueeze(3).unsqueeze(3).contiguous()
            
            weights_voxel = self._get_weights_voxel(
                instance_feature, anchor_embed, metas
            )
            weights_voxel = weights_voxel.permute(0,1,3,2).unsqueeze(3).unsqueeze(3)
            if self.use_deformable_func:
                key_points = key_points[:,:,:self.num_pts]
                key_points_voxel = key_points[:,:,self.num_pts:].reshape(bs, -1 ,self.num_groups, self.num_pts, 3)
            else:
                key_points_voxel = key_points.reshape(bs, -1 ,self.num_groups, self.num_pts, 3)
            key_points_voxel = key_points_voxel.unsqueeze(3).unsqueeze(3).contiguous()
        if self.use_DFA3D:
            points_3d = (
                self.project_points(
                    key_points, #[bs, num_anchor, num_pts, 3]
                    metas["projection_mat"],
                    metas.get("image_wh"),
                    return_3d=True,
                    depth_max = 60,
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_groups, self.num_pts, self.num_cams, 3)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5) # [bs, num_anchor, num_pts, num_cams, num_levels, num_groups]
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )
            features = self.DFA3D(value = feature_maps,
                                  value_dpt_dist = depth,
                                  sampling_locations = points_3d,
                                  spatial_shapes = spatial_shapes,
                                  level_start_index = level_start_index,
                                  attention_weights = weights,
                                  )
        if self.use_deformable_func:
            points_2d = (
                self.project_points(key_points,metas["projection_mat"],metas.get("image_wh"),)
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5) # [bs, num_anchor, num_pts, num_cams, num_levels, num_groups]
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )

            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )

        if self.use_voxel_feature:
            output_voxel = self.DA3D(voxel_feature=voxel_feature_maps.reshape(bs, V_y*V_x*V_z, D),
                                                    sampling_offsets=key_points_voxel,
                                                    attention_weights=weights_voxel,
                                                    spatial_shapes=spatial_shapes,
                                                    )
        if self.use_deformable_func or self.use_DFA3D:
            output = self.proj_drop(self.output_proj(features))
        if self.use_DFA3D:
            if self.residual_mode == "add":
                output = output + instance_feature
            elif self.residual_mode == "cat":
                output = torch.cat([output, instance_feature], dim=-1)
        else:
            if self.residual_mode == "add":
                if self.use_deformable_func and self.use_voxel_feature :
                    output = output + instance_feature + output_voxel
                if self.use_deformable_func and not self.use_voxel_feature:
                    output = output + instance_feature
                if not self.use_deformable_func and self.use_voxel_feature:
                    output = instance_feature + output_voxel
            elif self.residual_mode == "cat":
                if self.use_deformable_func and self.use_voxel_feature:
                    output = torch.cat([output, instance_feature, output_voxel], dim=-1)
                if self.use_deformable_func and not self.use_voxel_feature:
                    output = torch.cat([output, instance_feature], dim=-1)
                if not self.use_deformable_func and self.use_voxel_feature:
                    output = torch.cat([instance_feature, output_voxel], dim=-1)
        return output

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(
                    bs, self.num_cams, -1
                )
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_pts, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights
    def _get_weights_voxel(self, voxel_instance_feature, anchor_embed, metas=None):
        bs, num_anchor = voxel_instance_feature.shape[:2]
        feature = voxel_instance_feature + anchor_embed

        weights = (
            self.voxel_weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
        )
        # if self.training and self.attn_drop > 0:
        #     mask = torch.rand(
        #         bs, num_anchor, self.num_pts, 1
        #     )
        #     mask = mask.to(device=weights.device, dtype=weights.dtype)
        #     weights = ((mask > self.attn_drop) * weights) / (
        #         1 - self.attn_drop
        #     )
 
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None,return_3d=False,depth_max=60):
        bs, num_anchor, num_pts = key_points.shape[:3]
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        if return_3d:
            points_3d = torch.cat([points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5),  torch.clamp(points_2d[..., 2:3], min=1e-5) ], dim=-1)
            if image_wh is not None:
                points_3d[...,:2] = points_3d[...,:2] / image_wh[:, :, None, None]
                points_3d[...,-1] = points_3d[...,-1] / depth_max
            return points_3d
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d
                
    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1
        points_2d = points_2d.flatten(end_dim=1)

        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(
                    fm.flatten(end_dim=1), points_2d
                )
            )
        features = torch.stack(features, dim=1)
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(
            0, 4, 1, 2, 5, 3
        )  # bs, num_anchor, num_cams, num_levels, num_pts, embed_dims

        return features

    def multi_view_level_fusion(
        self,
        features: torch.Tensor,
        weights: torch.Tensor,
    ):
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(
            bs, num_anchor, self.num_pts, self.embed_dims
        )
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)
            fg_mask = torch.logical_and(
                gt > 0.0, torch.logical_not(torch.isnan(pred))
            )
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = (
                    error
                    / max(1.0, len(gt) * len(depth_preds))
                    * self.loss_weight
                )
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, (
            "num_fcs should be no less " f"than 2. got {num_fcs}."
        )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = (
            build_dropout(dropout_layer)
            if dropout_layer
            else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)



@FEEDFORWARD_NETWORK.register_module()
class SimpleFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        num_fcs=1,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(SimpleFFN, self).__init__(init_cfg)

        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        self.layers=Sequential(
                Linear(in_channels, embed_dims),
                self.activate,
                nn.Dropout(ffn_drop)
                )
            
        # self.dropout_layer = (
        #     build_dropout(dropout_layer)
        #     if dropout_layer
        #     else torch.nn.Identity()
        # )
        # self.add_identity = add_identity
        # self.identity_fc = Linear(self.in_channels, embed_dims)
            

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        # if not self.add_identity:
        #     return self.dropout_layer(out)
        # if identity is None:
        #     identity = x
        # identity = self.identity_fc(identity)
        return out

@FEEDFORWARD_NETWORK.register_module()
class OccFFN(BaseModule):
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        out_dims=1,
        feedforward_channels=64,
        num_fcs=3,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(OccFFN, self).__init__(init_cfg)
        # assert num_fcs >= 2, (
        #     "num_fcs should be no less " f"than 2. got {num_fcs}."
        # )
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.out_dims = out_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
    
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, out_dims))
        # layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        # self.dropout_layer = (
        #     build_dropout(dropout_layer)
        #     if dropout_layer
        #     else torch.nn.Identity()
        # )
        # self.add_identity = add_identity
        # if self.add_identity:
        #     self.identity_fc = (
        #         torch.nn.Identity()
        #         if in_channels == embed_dims
        #         else Linear(self.in_channels, embed_dims)
        #     )

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        # if not self.add_identity:
        # #     return self.dropout_layer(out)
        # if identity is None:
        #     identity = x
        # identity = self.identity_fc(identity)
        # return identity + self.dropout_layer(out)
        return out
