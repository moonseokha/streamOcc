# Copyright (c) Horizon Robotics. All rights reserved.
from inspect import signature

import torch
import torch.nn as nn
from torch.nn.init import normal_
import torch.nn.functional as F
import numpy as np
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence,build_positional_encoding
from mmdet.models import (
    DETECTORS,
    BaseDetector,
    build_backbone,
    build_head,
    build_neck,
)
from mmcv.cnn import build_conv_layer

from .grid_mask import GridMask
from .loss_utils import multiscale_supervision, geo_scal_loss, sem_scal_loss
from mmdet.models import LOSSES
from mmdet.models.backbones.resnet import ResNet
try:
    from ..ops import feature_maps_format
    DAF_VALID = True
except:
    DAF_VALID = False
import pdb
__all__ = ["Sparse4D","Sparse4D_BEVDepth"]

X, Y, Z, W, L, H, SIN, COS, YAW= 0, 1, 2, 3, 4, 5, 6, 7, 6
def box3d_to_corners(box3d):
    # Define constants
    
    boxes = box3d.clone().detach().unsqueeze(0)
    boxes[..., 3:6] = boxes[..., 3:6]

    corners_norm = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2)), dim=-1).view(-1, 3)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]] - 0.5
    corners = boxes[..., None, [W, L, H]] * corners_norm.to(boxes.device).reshape(1, 8, 3)

    rot_mat = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(boxes.shape[0], boxes.shape[1], 1, 1).to(boxes.device)
    rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 1, 0], rot_mat[..., 1, 1] = boxes[..., COS], -boxes[..., SIN], boxes[..., SIN], boxes[..., COS]

    corners = (rot_mat.unsqueeze(2) @ corners.unsqueeze(-1)).squeeze(-1)
    return (corners + boxes[..., None, :3]).squeeze(0)
    
MATCHING_DICT={0:4,1:10,2:5,3:3,4:9,5:1,6:6,7:2,8:7,9:8}

@DETECTORS.register_module()
class Sparse4D(BaseDetector):
    def __init__(
        self,
        img_backbone,
        img_backbone_det= None,
        head = None,
        img_view_transformer=None,
        Vox_Convnet = None,
        use_multi_frame = False,
        img_neck=None,
        img_neck_det=None,
        init_cfg=None,
        train_cfg = None,
        test_cfg = None,
        grid_config_view = None,
        downsample_list = None,
        # use_occ_focal_loss = False,
        loss_occ = None,
        occ_seqeunce = None,
        pretrained=None,
        use_grid_mask=True,
        use_deformable_func=False,
        using_mask='camera',
        depth_branch=None,
        use_voxel_feature=False,
        voxel_transformer=None,
        voxel_transformer_DAP=None,
        use_instance_mask=False,
        unet_dap=None,
        grid_config=None,
        pred_occ = False,
        use_semantic = False,
        loss_occupancy_aux = None,
        use_DFA3D = False,
        num_frames = 2,
        cls_freq=None,
        num_levels=4,
        occ_pred_weight = 0.5,
        det_pred_weight: float = None,
        inter_voxel_net: dict = None,
        voxel_encoder_backbone: dict = None,
        voxel_encoder_neck: dict = None,
        loss_dice: dict = None,
        num_cams: int = 6,
        occupy_threshold: float = 0.3,
        embed_dims: int = 256,
        num_feature_levels: int = 4,
        lift_start_num: int = 0,
        class_beta = 0.9,
        dataset_name = 'occ3d',
        mask_decoder_head: dict = None,
        group_split=None,
        use_img2vox:bool = False,
        use_COTR_version:bool = False,
        FPN_with_pred:bool = False,
        with_prev:bool = False,
        save_high_res_feature:bool = False,
        save_final_feature:bool = True,
        multi_neck:dict = None,
        num_classes:int = 17,
        num_instanace_query:int = 100,
        positional_encoding:dict = None,
        twice_technic:bool = False,
        
    ):
        super(Sparse4D, self).__init__(init_cfg=init_cfg)
        self.with_prev = with_prev
        self.use_voxel_feature = use_voxel_feature
        self.use_COTR_version = use_COTR_version
        self.multi_res = False
        self.num_instanace_query = num_instanace_query
        self.save_high_res_feature = save_high_res_feature
        self.twice_technic = twice_technic
        self.FPN_with_pred = FPN_with_pred
        self.pred_occ = pred_occ
        self.occupy_threshold = occupy_threshold
        self.save_final_feature = save_final_feature
        self.use_instance_mask = use_instance_mask
        self.num_frames = num_frames
        self.using_mask = using_mask
        # self.save_compact_feature = save_compact_feature
        self.dataset_name = dataset_name
        self.use_img2vox = use_img2vox
        self.use_semantic = use_semantic
        self.det_pred_weight = det_pred_weight
        if group_split is not None:
            self.group_split = torch.tensor(group_split, dtype=torch.uint8)
        self.grid_config = grid_config
        self.occ_pred_weight = occ_pred_weight
        self.lift_start_num = lift_start_num
        self.prev_imgs = None
        self.use_multi_frame = use_multi_frame
        self.loss_occupancy_aux = loss_occupancy_aux
        self.use_deformable_func = use_deformable_func
        self.num_levels = num_levels
        # self.use_occ_focal_loss = use_occ_focal_loss
        self.prev_metas = None
        self.cached_vox_feature = None
        if Vox_Convnet is not None:
            self.Vox_Convnet = build_from_cfg(Vox_Convnet, PLUGIN_LAYERS) 
        else:
            self.Vox_Convnet = None  
        if multi_neck is not None:
            self.multi_neck =  build_neck(multi_neck)
        else:
            self.multi_neck = None
        if pretrained is not None:
            backbone.pretrained = pretrained
        self.img_backbone = build_backbone(img_backbone)
        if img_backbone_det is not None:
            self.img_backbone_det = build_backbone(img_backbone_det)
        else:
            self.img_backbone_det = None
        if img_neck is not None:
            self.img_neck = build_neck(img_neck)
        if img_neck_det is not None:
            self.img_neck_det = build_neck(img_neck_det)
        else:
            self.img_neck_det = None
        if img_view_transformer is None:
            self.img_view_transformer = None
        else:
            if downsample_list is not None:
                self.downsample_list = downsample_list
                self.img_view_transformer = []
                if len(downsample_list) >= 1:
                    self.multi_res = True
                for downsample in downsample_list:
                    img_view_transformer.update({"downsample": downsample})
                    self.img_view_transformer.append(build_from_cfg(img_view_transformer,PLUGIN_LAYERS))
                self.img_view_transformer = nn.ModuleList(self.img_view_transformer)
            else:
                self.img_view_transformer = build_from_cfg(img_view_transformer,PLUGIN_LAYERS)
        self.use_grid_mask = use_grid_mask
        if loss_occupancy_aux is not None:
            self.aux_occ_loss = build_from_cfg(loss_occupancy_aux, LOSSES)
        else:
            self.aux_occ_loss = None
        if use_grid_mask:
            self.grid_mask = GridMask(
                True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7
            )
        else:
            self.grid_mask = None
        if depth_branch is not None:
            self.depth_branch = build_from_cfg(depth_branch, PLUGIN_LAYERS)
        else:
            self.depth_branch = None
        self.use_DFA3D = use_DFA3D
        if voxel_encoder_backbone is not None:
            self.voxel_encoder_backbone = build_backbone(voxel_encoder_backbone)
        else:
            self.voxel_encoder_backbone = None
        if voxel_encoder_neck is not None:
            self.voxel_encoder_neck = build_neck(voxel_encoder_neck)
        else:
            self.voxel_encoder_neck = None
        if inter_voxel_net is not None:
            self.inter_voxel_net = build_backbone(inter_voxel_net)
        else:
            self.inter_voxel_net = None
        if use_DFA3D:
            self.level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))
            self.cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
            normal_(self.level_embeds)
            normal_(self.cams_embeds)
        self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding is not None else None

        self.use_voxel_transformer = False
        if loss_dice is not None:
            self.loss_dice = build_from_cfg(loss_dice, LOSSES)
        else:
            self.loss_dice = None
            
        if head is not None:
            self.head = build_head(head)
            if use_deformable_func:
                assert DAF_VALID, "deformable_aggregation needs to be set up."

            self.use_voxel_encoder = False
            self.use_voxel_transformer = False
            self.use_voxel_transformer_DAP = False
            
            if unet_dap is not None:
                unet_dap.update({"voxel_transformer": voxel_transformer})
                self.unet_dap = build_from_cfg(unet_dap, PLUGIN_LAYERS)
                self.use_voxel_transformer = True
            else:
                self.unet_dap = None
                if voxel_transformer is not None:                
                    self.voxel_transformer = build_transformer_layer_sequence(voxel_transformer)
                    self.use_voxel_transformer = True
            if voxel_transformer_DAP is not None:
                self.voxel_transformer_DAP = build_transformer_layer_sequence(voxel_transformer_DAP)
                self.use_voxel_transformer_DAP = True
        else:
            self.head = None
        
        self.image_list = None
        self.meta_list = None
        
        if loss_occ is not None:
            self.loss_occ = build_from_cfg(loss_occ, LOSSES)
            self.use_occ_loss = True
            self.occ_seqeunce = occ_seqeunce
            if cls_freq is not None:
                self.cls_freq = cls_freq
                class_weights = torch.from_numpy(1 / np.log(np.array(self.cls_freq) + 0.001))
                self.loss_occ.class_weight = class_weights

        if mask_decoder_head is not None:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            mask_decoder_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            mask_decoder_head.update(test_cfg=pts_test_cfg)
            self.mask_decoder_head = build_head(mask_decoder_head)
        else:
            self.mask_decoder_head = None
        
        
        self.voxel_size = [int((grid_config['x'][1] - grid_config['x'][0])/grid_config['x'][2]),int((grid_config['y'][1] - grid_config['y'][0])/grid_config['y'][2]),int((grid_config['z'][1] - grid_config['z'][0])/grid_config['z'][2])]
        
    
    @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None, prev=False):
        view_trans_metas = None
        bs = img.shape[0]

        feature_maps_det = None
        origin_feature_maps = None
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)

        else:
            feature_maps = self.img_backbone(img)
  
        if self.img_neck is not None:
            if self.multi_neck is not None:
                lift_feature,inter_feature_maps = self.img_neck(feature_maps[-2:])
                if inter_feature_maps is None:
                    inter_feature_maps =  self.multi_neck(feature_maps)
                else:
                    inter_feature_maps = self.multi_neck(feature_maps[:self.num_levels-2],inter_feature_maps)
                if type(inter_feature_maps)==tuple:
                    inter_feature_maps = [element for element in inter_feature_maps]
                inter_feature_maps = [torch.reshape(feat, (bs, num_cams) + feat.shape[1:]) for feat in inter_feature_maps]
                if self.use_deformable_func:
                    origin_feature_maps = [f.clone() for f in inter_feature_maps]
                    inter_feature_maps = feature_maps_format(inter_feature_maps)
            else:
                lift_feature,_ = self.img_neck(feature_maps)
                inter_feature_maps = None
            
       
        feature_maps = lift_feature
        feature_maps = [torch.reshape(feature_maps, (bs, num_cams) + feature_maps.shape[1:])]


        depths = None
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        if self.use_voxel_feature:
            if self.multi_res or self.lift_start_num > 0:
                lift_feature = feature_maps
            else:
                lift_feature = feature_maps[0].clone()
            if self.lift_start_num > 0:
                lift_feature = lift_feature[self.lift_start_num:self.lift_start_num+len(self.downsample_list)]

        if self.head is not None:
            if self.use_DFA3D:
                return feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps
            if self.use_deformable_func:
                feature_maps = feature_maps_format(feature_maps)
                if feature_maps_det is not None:
                    feature_maps_det = feature_maps_format(feature_maps_det)
        if inter_feature_maps is not None:
            # origin_feature_maps = [f.clone() for f in inter_feature_maps]
            return inter_feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps
        origin_feature_maps = feature_maps

        return feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps

    def encoder_bbox(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]],
                    torch.sin(box[..., YAW]).unsqueeze(-1),
                    torch.cos(box[..., YAW]).unsqueeze(-1),
                    box[..., YAW + 1 :],
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs
    
    @force_fp32(apply_to=("x","voxel_feat","lss_depth","prev_vox_feat"))
    def voxel_encoder(self, x, metas,img_feats=None,view_trans_metas=None,only_voxel=False,mlvl_feats=None,prev_voxel_list=None,key_view_tran_comp=None,occ_pos=None,ref_3d=None,**kwargs):
        # Make 3D voxel feature 
        voxel_feats = []
        lss_depths = []
        vox_occ = None
        voxel_feature_list = None
        origin_voxel_feat = None
        prev_vox_feat= None
        if self.multi_res or self.lift_start_num > 0:
            for i in range(len(self.img_view_transformer)):
                mlp_input = None
                if only_voxel:
                    mlp_input = self.img_view_transformer[i].get_mlp_input(key_view_tran_comp)
                else:
                    mlp_input = self.img_view_transformer[i].get_mlp_input(metas["view_tran_comp"])
                if view_trans_metas is not None:
                    view_trans_metas.update({
                    "frustum":self.img_view_transformer[i].cv_frustum.to(mlp_input),
                    "cv_downsample":4,
                    "downsample":self.img_view_transformer[i].downsample,
                    "grid_config":self.img_view_transformer[i].grid_config,
                    })
                voxel_feat,lss_depth = self.img_view_transformer[i]([x[i].clone()]+metas["view_tran_comp"],metas["projection_mat"],mlp_input,view_trans_metas) # (B, C, Z, Y, X)
                voxel_feats.append(voxel_feat)
                lss_depths.append(lss_depth)
            voxel_feat = torch.stack(voxel_feats,dim=1).sum(dim=1) # (B, C, Z, H, W)
            lss_depth = lss_depths
        else:
            mlp_input = self.img_view_transformer.get_mlp_input(metas["view_tran_comp"])
            voxel_feat,lss_depth = self.img_view_transformer([x]+metas["view_tran_comp"],metas["projection_mat"],mlp_input,view_trans_metas) # (B, C, Z, Y, X)
        if only_voxel:
            return voxel_feat
        if self.FPN_with_pred is not True:
            if prev_voxel_list is not None:
                prev_voxel_list = torch.cat(prev_voxel_list,dim=1)
                voxel_feat = torch.cat([voxel_feat,prev_voxel_list],dim=1)
            if self.voxel_encoder_backbone is not None and self.voxel_encoder_neck is not None:
                voxel_feat = self.voxel_encoder_backbone(voxel_feat)
                voxel_feat,voxel_feature_list = self.voxel_encoder_neck(voxel_feat)
                if type(voxel_feat) in [tuple, list]:
                    voxel_feat = voxel_feat[0]
            if self.inter_voxel_net is not None:
                # origin_voxel_feat = voxel_feat.clone()
                voxel_feat = self.inter_voxel_net(voxel_feat,checksize=True)[0]
            # pdb.set_trace()
            
        if self.head is not None:
            if (
                self.head.instance_bank.cached_anchor is not None
                and voxel_feat.shape[0] != self.head.instance_bank.cached_anchor.shape[0]
            ):
                self.head.instance_bank.reset_vox_feature()
                self.head.instance_bank.metas = None
            prev_vox_feat = self.head.instance_bank.cached_vox_feature
            prev_metas = self.head.instance_bank.metas
        else:
            if self.cached_vox_feature is not None and (voxel_feat.shape[0] != self.cached_vox_feature.shape[0]):
                self.prev_metas = None
                self.cached_vox_feature = None
            prev_vox_feat = self.cached_vox_feature
            prev_metas = self.prev_metas
        

        occ_list = None
        vox_occ_list = None
        if prev_metas is not None:
            prev_times = prev_metas["timestamp"]
            prev_metas = prev_metas['img_metas']

        # pdb.set_trace()
        if self.FPN_with_pred:
            curr_voxel_feat = voxel_feat.clone()

        if self.Vox_Convnet is not None:
            if prev_vox_feat is not None:
                grid_size = prev_vox_feat.shape[2:]
                if self.grid_config is not None:
                    xs = torch.linspace(self.grid_config['x'][0], self.grid_config['x'][1], grid_size[1], device=voxel_feat.device)
                    ys = torch.linspace(self.grid_config['y'][0], self.grid_config['y'][1], grid_size[2], device=voxel_feat.device)
                    zs = torch.linspace(self.grid_config['z'][0], self.grid_config['z'][1], grid_size[0], device=voxel_feat.device)
                else:
                    xs = torch.linspace(-1, 1, grid_size[1], device=voxel_feat.device)
                    ys = torch.linspace(-1, 1, grid_size[2], device=voxel_feat.device)
                    zs = torch.linspace(-1, 1, grid_size[0], device=voxel_feat.device)
                Z,Y,X = torch.meshgrid(zs, ys, xs)
                grid = torch.stack([X,Y,Z], dim=-1).unsqueeze(0)
                _,d,h,w,c = grid.shape
                grid = grid.view(1,d,h,w,c).expand(voxel_feat.shape[0], d, h, w, c) # (B, X, Y, Z, 3) 
                T_temp2cur = torch.tensor(np.stack(
                            [
                                prev_metas[i]["T_global_inv"]  # global to box
                                @ x["T_global"] # box to global
                                for i, x in enumerate(metas["img_metas"])
                            ]
                        ), device=voxel_feat.device, dtype=voxel_feat.dtype)   # current to previous [B,4,4]
                # grid = grid.permute(0,2,1,3,4).contiguous()
                grid=torch.matmul(T_temp2cur[:,None,None,None,:3,:3],grid[...,None]).squeeze(-1) + T_temp2cur[:,None,None,None,:3, 3]
                if self.grid_config is not None:
                    grid[...,0] -= (self.grid_config['x'][0] + self.grid_config['x'][1])/2
                    grid[...,1] -= (self.grid_config['y'][0] + self.grid_config['y'][1])/2
                    grid[...,2] -= (self.grid_config['z'][0] + self.grid_config['z'][1])/2
                    grid[...,0] /= (self.grid_config['x'][1] - self.grid_config['x'][0])/2
                    grid[...,1] /= (self.grid_config['y'][1] - self.grid_config['y'][0])/2
                    grid[...,2] /= (self.grid_config['z'][1] - self.grid_config['z'][0])/2
                # grid = grid.permute(0,3,2,1,4).contiguous() # (B, Z, Y, X, 3)
                prev_vox_feat = F.grid_sample(prev_vox_feat, grid, align_corners=True) # (B,C,Z,X,Y)
                
                # prev_times = prev_metas["timestamp"]
                # prev_metas = prev_metas['img_metas']
                time_interval = metas["timestamp"] - prev_times
                time_interval = time_interval.to(dtype=voxel_feat.dtype)
                mask = torch.logical_and(torch.abs(time_interval) <= 2.0, time_interval != 0)

                for i,m in enumerate(mask):
                    if m == False:
                        if prev_vox_feat.shape[1] != voxel_feat.shape[1]:
                            if prev_vox_feat is not None:
                                prev_vox_feat[i] = prev_vox_feat[i].new_zeros(prev_vox_feat[i].shape)    
                            if self.prev_imgs is not None:
                                self.prev_imgs[i] = self.prev_imgs[i].new_zeros(self.prev_imgs[i].shape)
                        else:
                            prev_vox_feat[i] = voxel_feat[i].clone().detach()
            voxel_feat,occ_list,vox_occ_list = self.Vox_Convnet(voxel_feat,prev_vox_feat,metas,img_feats,mlvl_feats,occ_pos,ref_3d,self.twice_technic,**kwargs) # (B, C, Z, X, Y)


        if self.FPN_with_pred is True:
            if prev_voxel_list is not None:
                prev_voxel_list = torch.cat(prev_voxel_list,dim=1)
                voxel_feat = torch.cat([prev_voxel_list,curr_voxel_feat,voxel_feat],dim=1)
            else:
                voxel_feat = torch.cat([curr_voxel_feat,voxel_feat],dim=1)  # B, 2C, D, H, W

            if self.voxel_encoder_backbone is not None and self.voxel_encoder_neck is not None:
                if self.use_COTR_version is not True:
                    voxel_feat = self.voxel_encoder_backbone(voxel_feat)
                    voxel_feat,_ = self.voxel_encoder_neck(voxel_feat)
                else:
                    voxel_feat = self.voxel_encoder_backbone(voxel_feat)
                    voxel_feat,voxel_feature_list = self.voxel_encoder_neck(voxel_feat)
                if type(voxel_feat) in [tuple, list]:
                    voxel_feat = voxel_feat[0]
            if self.inter_voxel_net is not None:
                origin_voxel_feat = voxel_feat.clone()
                voxel_feat = self.inter_voxel_net(voxel_feat)[0]
        if self.head is not None:
            self.head.instance_bank.cached_vox_feature =voxel_feat.clone()# B, C, D, H, W
        else:
            self.cached_vox_feature = voxel_feat.clone()
            self.prev_metas = metas
        voxel_feat = voxel_feat.permute(0,3,4,2,1).contiguous()  # (B, C, Z, Y, X) -> (B, Y, X, Z, C)  


        if self.use_voxel_transformer:
            voxel_feat,occ_list = self.voxel_transformer(voxel_feat,None,None,None,None)
        if self.head is None:
            voxel_feat = voxel_feat.permute(0,4,1,2,3)
        return voxel_feat, lss_depth , occ_list, vox_occ_list, voxel_feature_list,vox_occ, origin_voxel_feat
    
    def pre_process_DFA3D(self, img, depth):
        feat_flatten = []
        dpt_dist_flatten = []
        spatial_shapes = []
        for lvl , (feat,dpt_dist) in enumerate(zip(img,depth)):
            bs,num,c,h,w = feat.shape
            dpt_dist = dpt_dist.reshape(bs,num,dpt_dist.shape[1],dpt_dist.shape[2],dpt_dist.shape[3])
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)
            dpt_dist = dpt_dist.flatten(3).permute(1, 0, 3, 2)
            # if self.use_cams_embeds:
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
            dpt_dist_flatten.append(dpt_dist)
            
        feat_flatten = torch.cat(feat_flatten, 2)
        dpt_dist_flatten = torch.cat(dpt_dist_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=depth[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        dpt_dist_flatten = dpt_dist_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        # pdb.set_trace()
        return feat_flatten, dpt_dist_flatten, spatial_shapes, level_start_index,vox_occ

    
    def generate_group(self, voxel_semantics):
        group_classes = []
        group_masks = []
        for i in range(len(self.group_split)+1):
            gt_classes = []
            sem_masks = []
            for voxel_semantic in voxel_semantics:
                if not i < 1:
                    w, h, z = voxel_semantic.shape
                    group_split = self.group_split[i-1].to(voxel_semantic)
                    voxel_semantic = group_split[voxel_semantic.flatten().long()].reshape(w, h, z)
                gt_class, sem_mask = self.generate_mask(voxel_semantic)
                gt_classes.append(gt_class.to(voxel_semantic.device))
                sem_masks.append(sem_mask.to(voxel_semantic.device))
            
            group_classes.append(gt_classes)
            group_masks.append(sem_masks)

        return group_classes, group_masks
    @force_fp32(apply_to=("img",))
    def forward(self, img, **data):
        sensor2ego ,intrin, post_rot, post_tran, bda =  data['view_tran_comp']
        data['cam_params'] = [sensor2ego, sensor2ego ,intrin, post_rot, post_tran, bda]
        data['img_params'] = [{'img_shape':[img.shape[-2:]],'sensor2ego':sensor2ego,'intrin':intrin}]
        
        if self.use_multi_frame:
            
            
            if self.image_list is not None:
                if img.shape[0] != self.image_list[0].shape[0]:
                    self.image_list = [img.clone() for _ in range(self.num_frames+1)]
                    self.meta_list = [data.copy() for _ in range(self.num_frames+1)]
                else:
                    time_interval = data["timestamp"] - self.meta_list[-1]["timestamp"]
                    time_mask = torch.abs(time_interval) <= 1.0
                    if self.num_frames >= 1:
                        # 기존의 리스트를 순서대로 유지하면서 새로 추가
                        self.image_list = self.image_list[1:] + [img.clone()]
                        self.meta_list = self.meta_list[1:] + [data.copy()]
                                                
                        # 시간에 따라 유효하지 않은 경우 데이터 업데이트
                        for i, mask in enumerate(time_mask):
                            for t in range(self.num_frames):
                                if not mask:
                                    self.image_list[t][i] = img[i].clone().unsqueeze(0)
                                    self.meta_list[t]['img_metas'][i] = data['img_metas'][i]
                                    self.meta_list[t]['timestamp'][i] = data['timestamp'][i]
                                    for key_num in range(len(data['view_tran_comp'])):
                                        self.meta_list[t]['view_tran_comp'][key_num][i] = data['view_tran_comp'][key_num][i]
                                    
            else:
                self.image_list = [img.clone() for _ in range(self.num_frames+1)]
                self.meta_list = [data.copy() for _ in range(self.num_frames+1)]

        if self.training:
            return self.forward_train(img, **data)
        else:
            return self.forward_test(img, **data)

    def forward_train(self, img, **data):
        voxel_feature=None
        prev_voxel_list = None
        occ_pos = None
        matching_indices = None
        model_outs = None
        
        ###############For using multi-frame################
        if self.use_multi_frame:
            image_list = self.image_list.copy()
            meta_list = self.meta_list.copy()
            prev_voxel_list = []
            
            sensor2keyego = data['view_tran_comp'][0].clone()
            bda_aug = data['view_tran_comp'][-1].clone()
            bda_aug_mat = torch.eye(4, device=img.device).unsqueeze(0).repeat(img.shape[0], 1, 1)
            bda_aug_mat[:,:3, :3] = bda_aug[:,:3, :3]
            bda_aug_mat = bda_aug_mat.unsqueeze(1)
            
            
            bdakeyego2global = torch.tensor([meta['T_global'] for meta in data['img_metas']], device=img.device,dtype=img.dtype).unsqueeze(1)
            # 체크해보기 쉐잎
            for i in range(1,self.num_frames):
                image = image_list[i]
                meta_data = meta_list[i].copy()
                prev_img = image_list[i-1]
                prev_meta = meta_list[i-1].copy()
                prev_voxel = self.extract_feat_prev(image,meta_data,prev_img,prev_meta,sensor2keyego,bda_aug_mat,bdakeyego2global).detach()
                prev_voxel_list.append(prev_voxel)
                
        feature_maps, lift_feature, depths,feature_maps_det,view_trans_metas,origin_feature_maps = self.extract_feat(img, True, data)
        occ_list = None
        vox_occ_list = None
        output = dict()
        
       
        W,H,D = self.voxel_size
        B = img.shape[0]
        occ_mask = torch.zeros((B, H, W, D),device=img.device).to(img.dtype)
        occ_pos = self.positional_encoding(occ_mask, 1).flatten(2).to(img.dtype).permute(0,2,1)  if self.positional_encoding is not None else None
        
        ref_3d = self.get_reference_points(
        H, W, D, dim='3dCustom', bs=B, device=img.device, dtype=img.dtype) # (B,X,Y,Z,3) -> (B,X*Y*Z,1,3)  H,W,D 형식의 input에 맞음
        
            
        if self.use_voxel_feature:
            voxel_feature, depths_voxel,occ_list, vox_occ_list,voxel_feature_list,vox_occ,origin_voxel_feat = self.voxel_encoder(lift_feature, metas=data,img_feats=feature_maps, view_trans_metas=view_trans_metas,prev_voxel_list=prev_voxel_list,mlvl_feats=origin_feature_maps,occ_pos=occ_pos,ref_3d=ref_3d,**data)
        if self.use_DFA3D:
            feat_flatten, dpt_dist_flatten, spatial_shapes, level_start_index = self.pre_process_DFA3D(feature_maps, depths_voxel)
            feature_maps = feat_flatten
        if vox_occ is not None:
            vox_occ_list.append(vox_occ)
        if self.head is not None:
            if self.use_DFA3D:
                model_outs,voxel_feature = self.head(feature_maps=feat_flatten, voxel_feature=voxel_feature, metas=data, occ_list=occ_list, depth = dpt_dist_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index)
            else:
                model_outs,voxel_feature = self.head(feature_maps=feature_maps, voxel_feature=voxel_feature, metas=data, occ_list=occ_list, depth = depths_voxel,occ_pos=occ_pos,ref_3d=ref_3d)
            # voxel_feature_occ.shape = (B, C, h, w, Z) 
            
            if "vox_occ" in model_outs:
                vox_occ_list+=model_outs["vox_occ"]
            if vox_occ_list is not None:
                output,matching_indices = self.head.loss(model_outs, data)
            if self.det_pred_weight is not None:
                # multiply the weight to the every loss 
                for key in output.keys():
                    output[key] *= self.det_pred_weight
        if depths is not None and "gt_depth_det" in data:
            if self.depth_branch is not None:
                output["loss_depth_branch"] = self.depth_branch.loss(
                    depths, data["gt_depth_det"]
                )
        if self.img_view_transformer is not None and "gt_depth" in data:
            if type(depths_voxel) is not list:
                output["loss_dense_depth"] = self.img_view_transformer.get_depth_loss(data["gt_depth"],depths_voxel)
            else:
                output["loss_dense_depth"] = sum(self.img_view_transformer[i].get_depth_loss(data["gt_depth"], depths_voxel[i]) for i in range(len(self.img_view_transformer))) / len(self.img_view_transformer)

        if self.mask_decoder_head is not None:
            if model_outs is not None:
                instance_feature = model_outs["instance_feature"]
            else:
                instance_feature = None
            loss_occ,mask_head_occ = self.forward_mask_decoder_train(voxel_feature, voxel_feature_list, origin_feature_maps, data,instance_feature ,origin_voxel_feat=origin_voxel_feat,matching_indices=matching_indices)
            output.update(loss_occ)
            if vox_occ_list is None:
                vox_occ_list = [mask_head_occ.permute(0,4,1,2,3)]
            else:
                vox_occ_list.append(mask_head_occ.permute(0,4,1,2,3)) # (B,W,H,D,C) -> (B, C, W, H, D)
        if self.pred_occ:
            if 'gt_occ' not in data:
                output_occ = self.loss_occ_pred(data['voxel_semantics'],data['mask_camera'],data['mask_lidar'],vox_occ_list,use_semantic=self.use_semantic,metas=data,using_mask=self.using_mask)
                output.update(output_occ)
            else:
                assert 'gt_occ' in data, 'gt_occ is not in data'
                output_occ = self.loss_surround_pred(data['gt_occ'],vox_occ_list,use_semantic=self.use_semantic,metas=data)
                output.update(output_occ)
        if self.twice_technic == False:
            if self.head is not None:
                self.head.instance_bank.cached_vox_feature = self.head.instance_bank.cached_vox_feature.detach()
            else:
                self.cached_vox_feature = self.cached_vox_feature.detach()
                
        return output

    def generate_mask(self, semantics):
        """Convert semantics to semantic mask for each instance
        Args:
            semantics: [W, H, Z]
        Return:
            classes: [N]
                N unique class in semantics
            masks: [N, W, H, Z]
                N instance masks
        """
        
        w, h, z = semantics.shape
        classes = torch.unique(semantics)

        gt_classes = classes.long()

        masks = []
        for class_id in classes:
            masks.append(semantics == class_id)
        
        if len(masks) == 0:
            masks = torch.zeros(0, w, h, z)
        else:
            masks = torch.stack([x.clone() for x in masks])

        return gt_classes, masks.long()

    @force_fp32(apply_to=('voxel_feats'))
    def forward_mask_decoder_train(self,
                          voxel_feats,
                          voxel_feature_list,
                          mlvl_feats,
                          data,
                          instance_queries,
                          origin_voxel_feat=None,
                          matching_indices=None,
                          ):
        """Forward function'
        Args:
            voxel_feats (torch.Tensor): [B, C, H, W, Z] -> [B, C, Z, H, W]
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """
        voxel_semantics = data['voxel_semantics']
        mask_camera = data['mask_camera']
        mask_lidar = data['mask_lidar']        
        gt_classes, sem_mask = self.generate_group(voxel_semantics)
        instance_queries_list = None
        if self.use_instance_mask:
            instance_queries_list = []
            gt_boxes = data['gt_bboxes_3d']
            gt_boxes = self.encoder_bbox(gt_boxes, device=voxel_feats.device)
            # get_coner_coords

            gt_labels = data['gt_labels_3d']
            gt_labels = [[MATCHING_DICT[label.item()] for label in labels] for labels in gt_labels]
            
            gt_instance_sem_masks = []
            gt_instance_classes = []
            grid_size = self.grid_config['range']
            
            sem_mask_all = sem_mask[0]
            sem_class_all = gt_classes[0]
            
            for b in range(voxel_feats.shape[0]):
                bottom_pos = box3d_to_corners(gt_boxes[b])
                vox_lower_point = torch.tensor(self.grid_config['range'][:3], dtype=bottom_pos.dtype, device=bottom_pos.device)
                vox_res = torch.tensor([0.4,0.4,0.4], dtype=bottom_pos.dtype, device=bottom_pos.device)
                vox_pos = ((bottom_pos - vox_lower_point)/ vox_res)
                W,H,Z = sem_mask[0][0].shape[1:]
                min_pos= vox_pos.min(dim=1)[0].floor().int()
                max_pos = vox_pos.max(dim=1)[0].ceil().int()
                min_pos[:,0] = torch.clamp(min_pos[:,0], 0, W)
                min_pos[:,1] = torch.clamp(min_pos[:,1], 0, H)
                min_pos[:,2] = torch.clamp(min_pos[:,2], 0, Z)
                max_pos[:,0] = torch.clamp(max_pos[:,0], 0, W)
                max_pos[:,1] = torch.clamp(max_pos[:,1], 0, H)
                max_pos[:,2] = torch.clamp(max_pos[:,2], 0, Z)
                gt_instance_sem_mask = []
                gt_instance_class = []
                for num in matching_indices[b][1]:
                    # num=num.item()
                    xs = torch.arange(min_pos[num, 0].item(), max_pos[num, 0].item() , device=voxel_feats.device)
                    ys = torch.arange(min_pos[num, 1].item(), max_pos[num, 1].item() , device=voxel_feats.device)
                    zs = torch.arange(min_pos[num, 2].item(), max_pos[num, 2].item() , device=voxel_feats.device)            
                    mesh_index = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1).reshape(-1, 3)
                
                    #해당 박스가 차지하는 voxel occupancy mask를 구함
                    mask = torch.zeros((W,H,Z), dtype=torch.int64, device=voxel_feats.device)    
                    mask[mesh_index[:,0], mesh_index[:,1], mesh_index[:,2]] = True
                    
                    # Label 확인해보자...
                    
                    instance_class = gt_labels[b][num]
                    instance_sem_mask = sem_mask_all[b][instance_class == sem_class_all[b]]
                    
                    mask = mask & instance_sem_mask
                    if mask.sum() == 0:
                        gt_instance_sem_mask.append(torch.zeros((1,W,H,Z), dtype=mask.dtype, device=mask.device))
                        gt_instance_class.append(torch.tensor([0], device=voxel_feats.device))
                    else:
                        gt_instance_sem_mask.append(mask)            
                        gt_instance_class.append(instance_class-1)
                    
                if gt_instance_sem_mask == []:
                    print({"gt_boxes":gt_boxes[b]})
                    gt_instance_sem_masks.append(torch.zeros((1,W,H,Z), dtype=torch.int64, device=voxel_feats.device))
                    gt_instance_classes.append(torch.tensor([0], device=voxel_feats.device))
                else: 
                    gt_instance_sem_masks.append(torch.cat(gt_instance_sem_mask,dim=0))
                    gt_instance_classes.append(torch.tensor(gt_instance_class, device=voxel_feats.device))
                    
                assert gt_instance_sem_masks[b].shape[0] == matching_indices[b][1].shape[0]
                gt_instance_sem_masks[b] = gt_instance_sem_masks[b][matching_indices[b][1]]
                gt_instance_classes[b] = gt_instance_classes[b][matching_indices[b][1]]
                instance_queries_list.append(instance_queries[b][matching_indices[b][0]].unsqueeze(0))
            gt_classes.append(gt_instance_classes)
            sem_mask.append(gt_instance_sem_masks)
        # pdb.set_trace()
        outs,compact_occ = self.mask_decoder_head(voxel_feats,voxel_feature_list=voxel_feature_list,mlvl_feats=mlvl_feats,threshold=self.occupy_threshold,instance_queries=instance_queries_list,origin_voxel_feat=origin_voxel_feat, **data)
        if self.save_final_feature:
            if self.head is not None:
                # if self.with_prev:
                self.head.instance_bank.cached_vox_feature = compact_occ.permute(0,1,4,3,2)  # b,c,w,h,z -> b,c,z,h,w
                # else:
                    # self.head.instance_bank.cached_vox_feature = compact_occ.new_zeros(compact_occ.shape).permute(0,1,4,3,2)
            else:
                # if self.with_prev:
                self.cached_vox_feature = compact_occ.permute(0,1,4,3,2)  # b,c,w,h,z -> b,c,z,h,w
                # else:
                    # self.cached_vox_feature = compact_occ.new_zeros(compact_occ.shape).permute(0,1,4,3,2)
        loss_inputs = [voxel_semantics, gt_classes, sem_mask,
                       mask_camera, mask_lidar, outs]
        losses = self.mask_decoder_head.loss(*loss_inputs)
        
        return losses,outs['occ_outs']

    def forward_test(self, img, **data):
        if isinstance(img, list):
            return self.aug_test(img, **data)
        else:
            return self.simple_test(img, **data)

    def simple_test(self, img, **data):
        voxel_feature=None
        vox_occ_list = None
        results = None
        output = dict()
        prev_voxel_list = None
        instance_queries = None
        occ_results = None
        if self.use_multi_frame:
            image_list = self.image_list.copy()
            meta_list = self.meta_list.copy()
            prev_voxel_list = []
            sensor2keyego = data['view_tran_comp'][0].clone()
            bda_aug = data['view_tran_comp'][-1].clone()
            bda_aug_mat = torch.eye(4, device=img.device).unsqueeze(0).repeat(img.shape[0], 1, 1)
            bda_aug_mat[:,:3, :3] = bda_aug[:,:3, :3]
            bda_aug_mat = bda_aug_mat.unsqueeze(1)

            bdakeyego2global = torch.tensor([meta['T_global'] for meta in data['img_metas']], device=img.device,dtype=img.dtype)
            
            for i in range(1,self.num_frames):
                image = image_list[i]
                meta_data = meta_list[i].copy()
                prev_img = image_list[i-1]
                prev_meta = meta_list[i-1].copy()
                prev_voxel = self.extract_feat_prev(image,meta_data,prev_img,prev_meta,sensor2keyego,bda_aug_mat,bdakeyego2global)
                prev_voxel_list.append(prev_voxel)
                
        feature_maps, lift_feature, depths,feature_maps_det, view_trans_metas,origin_feature_maps = self.extract_feat(img, False, data)
        # if self.positional_encoding is not None:
        W,H,D = self.voxel_size
        B = img.shape[0]
        occ_mask = torch.zeros((B, H, W, D),device=img.device).to(img.dtype)
        occ_pos = self.positional_encoding(occ_mask, 1).flatten(2).to(img.dtype).permute(0,2,1) if self.positional_encoding is not None else None
            
        ref_3d = self.get_reference_points(
        H, W, D, dim='3dCustom', bs=B, device=img.device, dtype=img.dtype) # (B,X,Y,Z,3) -> (B,X*Y*Z,1,3)  H,W,D 형식의 input에 맞음
            # pdb.set_trace()
            
        if self.use_voxel_feature:
            voxel_feature, depths_voxel,occ_list, vox_occ_list,voxel_feature_list,vox_occ,origin_voxel_feat = self.voxel_encoder(lift_feature, metas=data,img_feats=feature_maps, view_trans_metas=view_trans_metas,prev_voxel_list=prev_voxel_list,mlvl_feats=origin_feature_maps,occ_pos=occ_pos,ref_3d=ref_3d,**data)
            if self.head is None and self.mask_decoder_head is None:
                occ_results = vox_occ_list[-1]
        if self.use_DFA3D:
            feat_flatten, dpt_dist_flatten, spatial_shapes, level_start_index = self.pre_process_DFA3D(feature_maps, depths_voxel)
        if self.head is not None:
            # if feature_maps_detach is not None:
            #     feature_maps = feature_maps_detach
            if feature_maps_det is not None:
                feature_maps = feature_maps_det
            model_outs,voxel_feature = self.head(feature_maps=feature_maps, voxel_feature=voxel_feature, metas=data,depth = depths_voxel,occ_pos=occ_pos,ref_3d=ref_3d)
            if self.mask_decoder_head is None:
                if "vox_occ" in model_outs:
                    occ_results=model_outs["vox_occ"][-1]
            results = self.head.post_process(model_outs)
        # occ_results = None
        
        if self.use_instance_mask:
            instance_mask = model_outs["classification"][-1].sigmoid().max(-1)[0] > 0.3
            instance_queries = model_outs["instance_feature"][instance_mask]
        # if self.use_instance_mask:
        #     top_k_indices = model_outs["classification"][-1].sigmoid().max(-1)[0].sort()[1][:,:self.num_instanace_query].unsqueeze(-1).repeat(1,1,voxel_feature.shape[1])
        #     instance_queries = torch.gather(model_outs["instance_feature"],1,top_k_indices) # [B, N, C]
        # else:
        #     instance_queries = None
        
        if vox_occ_list != []:
            if self.dataset_name == 'surround':
                _ ,occ_scores = torch.max(vox_occ_list.softmax(dim=1),dim=1)
                occ_results = self.evaluation_semantic(occ_scores, data['gt_occ'], vox_occ_list[-1].shape[1])
            # elif self.dataset_name == 'occ3d':
            #     occ_results = vox_occ_list
        if occ_results is None:
            if results is not None:
                output = [{"img_bbox": result} for result in results]
        else:
            if occ_results is not None:
                if self.head is not None:
                    if self.dataset_name == 'surround':
                        output = {"boxes":results[0],"surround_results": occ_results}
                    elif self.dataset_name == 'occ3d':
                        occ_score = occ_results.permute(0,2,3,4,1)
                        occ_res = occ_score.argmax(-1)
                        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
                        output = {"boxes":results[0],"occ_results": occ_res}
                    else:
                        output = {"boxes":results[0]}
                else:
                    if self.dataset_name == 'surround':
                        output = {"surround_results": occ_results}
                    elif self.dataset_name == 'occ3d':
                        # pdb.set_trace()
                        occ_score = occ_results.permute(0,2,3,4,1)
                        occ_res = occ_score.argmax(-1)
                        occ_res = occ_res.squeeze(dim=0).cpu().numpy().astype(np.uint8)
                        output = {"occ_results": occ_res}
        if self.mask_decoder_head is not None:
            #TODO imgfeats 설정
            outs,compact_occ = self.mask_decoder_head(voxel_feature,voxel_feature_list=voxel_feature_list,mlvl_feats=origin_feature_maps,threshold=self.occupy_threshold,instance_queries=instance_queries,origin_voxel_feat=origin_voxel_feat, **data)
            occ = self.mask_decoder_head.get_occ(outs)
            if self.save_final_feature:
                if self.head is not None:
                    self.head.instance_bank.cached_vox_feature = compact_occ.permute(0,1,4,3,2)
                else:
                    self.cached_vox_feature = compact_occ.permute(0,1,4,3,2)
            if type(output) is list:
                output = {"occ_results": occ['occ']}
            else:
                output["occ_results"] = occ['occ']

            #TODO: set occ_results
        return [output]

    def aug_test(self, img, **data):
        # fake test time augmentation
        for key in data.keys():
            if isinstance(data[key], list):
                data[key] = data[key][0]
        return self.simple_test(img[0], **data)

    def gt_to_voxel(self, gt, vox_shape):
        voxel = np.zeros(vox_shape)
        voxel[gt[:, 0].astype(np.int), gt[:, 1].astype(np.int), gt[:, 2].astype(np.int)] = gt[:, 3]

        return voxel
    def evaluation_semantic(self,pred_occ, gt_occ, class_num):
        results = []

        for i in range(pred_occ.shape[0]):
            gt_i, pred_i = gt_occ[i].cpu().numpy(), pred_occ[i].cpu().numpy()
            gt_i = self.gt_to_voxel(gt_i, pred_occ.shape[1:])
            mask = (gt_i != 255)
            score = np.zeros((class_num, 3))
            for j in range(class_num):
                if j == 0: #class 0 for geometry IoU
                    score[j][0] += ((gt_i[mask] != 0) * (pred_i[mask] != 0)).sum()
                    score[j][1] += (gt_i[mask] != 0).sum()
                    score[j][2] += (pred_i[mask] != 0).sum()
                else:
                    score[j][0] += ((gt_i[mask] == j) * (pred_i[mask] == j)).sum()
                    score[j][1] += (gt_i[mask] == j).sum()
                    score[j][2] += (pred_i[mask] == j).sum()

            results.append(score)
        return np.stack(results, axis=0)
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss_occ_pred(self,
             gt_occ, # [B, X , Y , Z]
             mask_camera,  # [B, X , Y , Z]
             mask_lidar,  # [B, X , Y , Z]
             preds_dicts,
             use_semantic = False,
             metas = None,
             using_mask = 'camera'
             ):
        if using_mask == 'camera':
            visible_mask = mask_camera.to(torch.int32)
        elif using_mask == 'lidar':
            visible_mask = mask_lidar.to(torch.int32)
        loss_dict = {}
        mask_voxel_semantics = torch.where(visible_mask>0, gt_occ, torch.ones_like(gt_occ)*17)

        visible_mask = visible_mask.reshape(-1)
        num_total_samples = visible_mask.sum()
        voxel_semantics=gt_occ.long()
        for i in range(len(preds_dicts)):
            preds = preds_dicts[i]
            aux_pred = preds.clone()
            preds = preds.permute(0,2,3,4,1) #  [B,C,W,H,Z] -> [B,W,H,Z,C]
            # pdb.set_trace()
            if self.use_occ_loss:
                if using_mask is not None:
                    loss_occ_i = self.loss_occ(preds.reshape(-1, 18), voxel_semantics.reshape(-1), visible_mask, avg_factor=num_total_samples)
                else:
                    loss_occ_i = self.loss_occ(preds.reshape(-1, self.num_classes+1), voxel_semantics)
                if self.aux_occ_loss is not None:       
                    # mask_voxel_semantics = torch.where(visible_mask>0,voxel_semantics,torch.tensor(17).to(voxel_semantics.device))
                    loss_occ_aux_i = self.aux_occ_loss(aux_pred.permute(0,1,4,2,3).contiguous(), mask_voxel_semantics.permute(0,3,1,2).contiguous()) # B, C, L, H, W
                    # pdb.set_trace()

            else:
                loss_occ_i =  sem_scal_loss(pred, voxel_semantics.long(),unknown_index=17) + geo_scal_loss(pred, voxel_semantics.long(),unknown_index=17)
                loss_occ_i += criterion(pred, voxel_semantics.long())
                if self.aux_occ_loss is not None:       
                    loss_occ_i += self.aux_occ_loss(pred, voxel_semantics.long())

            loss_occ_i *=  self.occ_pred_weight
            if self.aux_occ_loss is not None:  
                loss_occ_aux_i *=  self.occ_pred_weight
            if i != (len(preds_dicts)-1):
                loss_occ_i *=0.5
                if self.aux_occ_loss is not None:  
                    loss_occ_aux_i *=0.5
            if self.aux_occ_loss is not None:
                loss_dict['loss_pred_occ_aux_{}'.format(i)] = loss_occ_aux_i
            loss_dict['loss_pred_occ_{}'.format(i)] = loss_occ_i
            
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def loss_surround_pred(self,
             gt_occ,
             preds_dicts,
             use_semantic = False,
             metas = None,
             ):
        ratio = 1
        X,Y,Z = preds_dicts[0].shape[2:]
        if metas is not None:
            bda_mat = metas['view_tran_comp'][-1]
            for i,bda in enumerate(bda_mat):
                flip = bda.diagonal()
                if flip[0] == -1:
                    gt_occ[i][:,0] = (gt_occ[i][:,0]+1 - X/2) * -1 + X/2
                if flip[1] == -1:
                    gt_occ[i][:,1] = (gt_occ[i][:,1]+1 - Y/2) * -1 + Y/2
        gt = multiscale_supervision(gt_occ, ratio, preds_dicts[0].shape) # [B, X , Y , Z]         
        # pdb.set_trace()       
        if not use_semantic:
            loss_dict = {}
            for i in range(len(preds_dicts)):
                preds_dicts[i] = preds_dicts[i]
                pred = preds_dicts[i][:,0]
                
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) + geo_scal_loss(pred, gt.long(), semantic=False))
                    
                loss_occ_i =  loss_occ_i * ((0.5)**(len(preds_dicts) - 1 -i)) #* focal_weight

                loss_dict['loss_pred_occ_{}'.format(i)] = loss_occ_i * self.occ_pred_weight
    
        else:
            pred = preds_dicts
            criterion = nn.CrossEntropyLoss(
                ignore_index=255, reduction="mean"
            )
            
            loss_dict = {}
            if self.use_occ_loss:
                gt_masking = gt[(gt!=255)]
            if self.aux_occ_loss is not None:
                gt_aux = gt.clone()
                gt_aux[gt==255]=0
            for i in range(len(preds_dicts)):
                pred = preds_dicts[i]
                if self.use_occ_loss:
                    pred_masking = pred.permute(0,2,3,4,1)[(gt!=255)]
                    if self.occ_seqeunce is not None:
                        if self.occ_seqeunce[i] == 'object':
                            occ_mask = torch.logical_and(gt_masking>0,gt_masking<=10)
                            loss_occ_i = self.loss_occ(pred_masking[occ_mask],gt_masking[occ_mask].to(dtype=torch.long),avg_factor = occ_mask.sum()) * 3
                            if self.aux_occ_loss is not None:       
                                gt_obj = gt_aux.clone()
                                gt_obj[gt_aux>10] = 0
                                loss_occ_i += self.aux_occ_loss(pred, gt_obj.long())
                        elif self.occ_seqeunce[i] == 'back':
                            occ_mask = torch.logical_or(gt_masking==0,gt_masking>10)
                            loss_occ_i = self.loss_occ(pred_masking[occ_mask],gt_masking[occ_mask].to(dtype=torch.long),avg_factor = occ_mask.sum())
                            loss_occ_i +=  (sem_scal_loss(pred_masking[(gt_masking>9)],gt_masking[(gt_masking>9)].to(dtype=torch.long)) + geo_scal_loss(pred_masking[(gt_masking>9)],gt_masking[(gt_masking>9)].to(dtype=torch.long)))
                            # loss_occ_i +=  (geo_scal_loss(pred_masking[(gt_masking>9)],gt_masking[(gt_masking>9)].to(dtype=torch.long)))
                            if self.aux_occ_loss is not None:       
                                gt_back = gt_aux.clone()
                                gt_back[gt_back<=10] = 0
                                loss_occ_i += self.aux_occ_loss(pred, gt_back.long())
                        else:
                            loss_occ_i = self.loss_occ(pred_masking.reshape(-1,17),gt_masking.flatten().to(dtype=torch.long),avg_factor = (gt!=255).sum())

                            loss_occ_i +=  (sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long()))* 0.1
                            if self.aux_occ_loss is not None:       
                                loss_occ_i += self.aux_occ_loss(pred, gt_aux.long())
                    else:
                            loss_occ_i = self.loss_occ(pred_masking.reshape(-1,17),gt_masking.flatten().to(dtype=torch.long),avg_factor = (gt!=255).sum())
                            # loss_occ_i +=  (sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long())) * 0.1
                            if self.aux_occ_loss is not None:       
                                loss_occ_i += self.aux_occ_loss(pred, gt_aux.long())
                else:
                    loss_occ_i =  sem_scal_loss(pred, gt.long()) + geo_scal_loss(pred, gt.long())
                    loss_occ_i += criterion(pred, gt.long())
                    if self.aux_occ_loss is not None:       
                        loss_occ_i += self.aux_occ_loss(pred, gt_aux.long())

                loss_occ_i *=  self.occ_pred_weight
                if i != (len(preds_dicts)-1):
                    loss_occ_i *=0.5
                loss_dict['loss_pred_occ_{}'.format(i)] = loss_occ_i

                    
        return loss_dict
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
    
    
    
    
    
@DETECTORS.register_module()
class Sparse4D_BEVDepth(Sparse4D):

    def __init__(self,
                 with_prev=True,
                 use_multi_fpn=True,
                 **kwargs):
        super(Sparse4D_BEVDepth, self).__init__(**kwargs)
        self.use_multi_fpn = use_multi_fpn
        self.with_prev = with_prev,
        self.extra_ref_frames = 1

    def extract_stereo_ref_feat(self, x):
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
        else:
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
            
            
    # @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat_prev(self, img, metas,prev_img,prev_meta,sensor2keyego,bda_aug,bdakeyego2global):
        feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps= self.extract_feat(img, True, metas,prev_img=prev_img,prev_meta=prev_meta,prev=True)
        
        sensor2ego ,intrin, post_rot, post_tran, bda =  metas['view_tran_comp']
        bda_mat = torch.eye(4, device=img.device).unsqueeze(0).repeat(img.shape[0], 1, 1)
        bda_mat[:,:3, :3] = bda
        bda_mat = bda_mat.unsqueeze(1)
        ego2global = torch.stack([torch.tensor(meta['T_global'],device=img.device,dtype=bda.dtype) for meta in metas['img_metas']],dim=0).unsqueeze(1)
        adj2keyego = torch.inverse(bda_aug)@ torch.inverse(bdakeyego2global) @ ego2global @ bda_mat @ sensor2ego
        metas['view_tran_comp'] = [adj2keyego, intrin, post_rot, post_tran, bda.squeeze(1)]
        key_view_tran_comp = [sensor2keyego, intrin, post_rot, post_tran, bda.squeeze(1)]
        prev_vox_feat = self.voxel_encoder(lift_feature, metas=metas,img_feats=feature_maps, view_trans_metas=view_trans_metas,only_voxel=True,key_view_tran_comp=key_view_tran_comp)
        return prev_vox_feat
    


    # @auto_fp16(apply_to=("img",), out_fp32=True)
    def extract_feat(self, img, return_depth=False, metas=None, prev_img=None,prev_meta=None,prev=False):
        view_trans_metas = None
        bs = img.shape[0]
        origin_feature_maps = None
        prev_metas=None
        prev_imgs=None
        if prev_img is None:
            if self.prev_imgs is None:
                prev_imgs = img.clone().detach()
                prev_metas = metas
            else:
                if (bs != self.prev_imgs.shape[0]):
                    prev_metas = metas
                    prev_imgs = img.clone().detach()
                else:
                    if self.head is not None:
                        prev_metas = self.head.instance_bank.metas.copy()
                    else:
                        prev_metas = self.prev_metas.copy()
                    prev_imgs = self.prev_imgs.clone().detach()
        else:
            prev_imgs = prev_img
            prev_metas = prev_meta
        if not prev:
            prev_times = prev_metas["timestamp"]
                
            time_interval = metas["timestamp"] - prev_times
            time_interval = time_interval.to(dtype=img.dtype)
            mask = torch.abs(time_interval) <= 1.0
            for i,m in enumerate(mask):
                if m == False:
                    prev_imgs[i] = img[i].clone()       
                    prev_metas["timestamp"][i] = metas["timestamp"][i]
                    prev_metas["img_metas"][i] = metas["img_metas"][i]
                    for key_num in range(len(metas['view_tran_comp'])):
                        prev_metas['view_tran_comp'][key_num][i] = metas['view_tran_comp'][key_num][i]
                    
                
                
        prev_input = prev_metas['view_tran_comp']
        curr_input = metas['view_tran_comp']

        prev_ego2global = torch.tensor(np.array([meta['T_global'] for meta in prev_metas['img_metas']]), device=img.device)
        curr_ego2global = torch.tensor(np.array([meta['T_global'] for meta in metas['img_metas']]), device=img.device)
        sensor2egos_curr = curr_input[0].double()
        ego2globals_curr = curr_ego2global[:,None, ...].double()
        bda_curr = curr_input[-1].double()
        bda_curr_mat = torch.eye(4, device=img.device).unsqueeze(0).repeat(img.shape[0], 1, 1).double()
        bda_curr_mat[:,:3, :3] = bda_curr[:,:3, :3]
        bda_curr_mat = bda_curr_mat.unsqueeze(1)

        sensor2egos_adj =  prev_input[0].double()
        ego2globals_adj =  prev_ego2global[:, None, ...].double()
        bda_adj = prev_input[-1].double()        
        bda_adj_mat = torch.eye(4, device=img.device).unsqueeze(0).repeat(img.shape[0], 1, 1).double()
        bda_adj_mat[:,:3, :3] = bda_adj[:,:3, :3]
        bda_adj_mat = bda_adj_mat.unsqueeze(1)
        
        
        curr2adjsensor =\
            torch.inverse(sensor2egos_adj)@torch.inverse(bda_adj_mat)@torch.inverse(ego2globals_adj)\
                @ ego2globals_curr @ bda_curr_mat @ sensor2egos_curr
        curr2adjsensor = curr2adjsensor.float()
        # curr2adjsensor = torch.split(curr2adjsensor, 1, 1)
        # curr2adjsensor = [p.squeeze(1) for p in curr2adjsensor]
        # curr2adjsensor.extend([None for _ in range(self.extra_ref_frames)])
        
        feature_maps_det = None
        if img.dim() == 5:  # multi-view
            num_cams = img.shape[1]
            img = img.flatten(end_dim=1)
        else:
            num_cams = 1
        if self.use_grid_mask:
            img = self.grid_mask(img)
        if "metas" in signature(self.img_backbone.forward).parameters:
            feature_maps = self.img_backbone(img, num_cams, metas=metas)
        else:
            feature_maps = self.img_backbone(img)
        stereo_feat = feature_maps[0].clone()
        
        sensor2ego, intrin, post_rot, post_tran, bda = curr_input
        k2s_sensor = curr2adjsensor

        feat_prev_iv = None
        with torch.no_grad():
            feat_prev_iv = self.extract_stereo_ref_feat(prev_imgs)
        view_trans_metas = dict(k2s_sensor=k2s_sensor,
                     intrins=intrin,
                     post_rots=post_rot,
                     post_trans=post_tran,
                     cv_downsample=4,
                     cv_feat_list=[feat_prev_iv, stereo_feat]
                     )
        lift_feature = None
        depths = None
        if not self.use_multi_fpn:
            
            if self.img_neck is not None:
                feature_maps = self.img_neck(feature_maps)
            if type(feature_maps)==tuple:
                feature_maps = [element for element in feature_maps]
            # for i, feat in enumerate(feature_maps):
            #     feature_maps[i] = torch.reshape(feat, (bs, num_cams) + feat.shape[1:])
            feature_maps = [torch.reshape(feat, (bs, num_cams) + feat.shape[1:]) for feat in feature_maps]

            if return_depth and self.depth_branch is not None:
                depths = self.depth_branch(feature_maps, metas.get("focal"))
            if self.use_voxel_feature:
                if self.multi_res or self.lift_start_num > 0:
                    lift_feature = feature_maps
                else:
                    lift_feature = feature_maps[0].clone()
                if self.lift_start_num > 0:
                    lift_feature = lift_feature[self.lift_start_num:self.lift_start_num+len(self.downsample_list)]
            if prev_img is None:
                self.prev_imgs = img.clone().detach()
            # origin_feature_maps = [f.clone() for f in feature_maps]
            origin_feature_maps=feature_maps
            if self.head is not None:
                if self.use_DFA3D:
                    return feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps
                if self.use_deformable_func:
                    feature_maps = feature_maps_format(feature_maps)
                    if feature_maps_det is not None:
                        feature_maps_det = feature_maps_format(feature_maps_det)
            return feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps
            

        if self.img_neck is not None:
            neck_feature_maps,inter_feature_maps = self.img_neck(feature_maps[-2:])
        
        if prev:
            feature_maps = [torch.reshape(neck_feature_maps, (bs, num_cams) + neck_feature_maps.shape[1:])]
            lift_feature = feature_maps[self.lift_start_num:self.lift_start_num+len(self.downsample_list)]
            return None,lift_feature,None,None,view_trans_metas,None
        
        if self.multi_neck is not None:
            if inter_feature_maps is None:
                inter_feature_maps =  self.multi_neck(feature_maps)
            else:
                inter_feature_maps = self.multi_neck(feature_maps[:-2],inter_feature_maps)
        if type(inter_feature_maps)==tuple:
            inter_feature_maps = [element for element in inter_feature_maps]
        # pdb.set_trace()
        inter_feature_maps = [torch.reshape(feat, (bs, num_cams) + feat.shape[1:]) for feat in inter_feature_maps]
        feature_maps = [torch.reshape(neck_feature_maps, (bs, num_cams) + neck_feature_maps.shape[1:])]
        if return_depth and self.depth_branch is not None:
            depths = self.depth_branch(feature_maps, metas.get("focal"))
        if self.use_voxel_feature:
            if self.multi_res or self.lift_start_num > 0:
                lift_feature = feature_maps
            else:
                lift_feature = feature_maps[0].clone()
            if self.lift_start_num > 0:
                lift_feature = lift_feature[self.lift_start_num:self.lift_start_num+len(self.downsample_list)]
        if prev_img is None:
            self.prev_imgs = img.clone().detach()
        origin_feature_maps = [f.clone() for f in inter_feature_maps]
        if self.head is not None:
            if self.use_DFA3D:
                return feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps
            if self.use_deformable_func:
                inter_feature_maps = feature_maps_format(inter_feature_maps)
        return inter_feature_maps, lift_feature, depths, feature_maps_det, view_trans_metas,origin_feature_maps