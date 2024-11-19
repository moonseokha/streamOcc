# Copyright (c) Horizon Robotics. All rights reserved.
from typing import List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean
from .modules import Stage2Assigner,MultipleInputEmbedding
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from .blocks import DeformableFeatureAggregation as DFG
import pdb
__all__ = ["Sparse4DHead"]

@HEADS.register_module()
class Sparse4DHead(BaseModule):
    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        refine_layer: dict,
        ffn: dict = None,
        deformable_model: dict = None,
        num_decoder: int = 6,
        deformable_model_vox: dict = None,
        use_occ_loss=False,
        num_single_frame_decoder: int = -1,
        ffn_o2o: dict = None,
        ffn_simple:dict = None,
        ffn_first:dict = None,
        temp_graph_model: dict = None,
        interaction_graph_model: dict = None,
        interaction_graph_model_wo_occ: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        loss_occ_focal: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        ffn_o2m: dict = None,
        ffn_vox: dict = None,
        use_gnn_mask: bool = False,
        use_o2m_loss: bool = False,
        same_o2o: bool = False,
        first_o2m: bool = False,
        top_k_query: int = 40,
        top_k_thr: float = None,
        only_init: bool = False,
        tracking_train: bool = False,
        only_o2m:bool = False,
        with_o2m:bool = False,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        gt_id_key: str = "instance_id",
        gt_occ_key: str = "gt_segmentation",
        occ_weight: int = 200,
        allow_low_quality_matches: bool = False,
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        occ_seqeunce: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        use_voxel_feature: bool = False,
        use_dap: bool = False,
        predictor: dict = None,
        loss_next: dict = None,
        num_classes: int = 18,
        reg_weights_next: List = None,
        use_nms_filter: bool = False,
        save_final_voxel_feature : bool = False,
        use_vox_atten : bool = False,
        nms_thr : float = 0.3,
        positional_encoding: dict = None,
        embed_dims: int = 256,
        use_mask_head: bool = False,

        **kwargs,
    ):
        super(Sparse4DHead, self).__init__(init_cfg)
        # pdb.set_trace()
        self.use_nms_filter = use_nms_filter
        self.use_voxel_feature = use_voxel_feature
        self.use_vox_atten = use_vox_atten
        self.use_mask_head = use_mask_head
        self.num_decoder = num_decoder
        self.num_classes = num_classes
        self.save_final_voxel_feature = save_final_voxel_feature
        self.num_single_frame_decoder = num_single_frame_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.use_dap = use_dap
        self.only_init = only_init
        self.dn_loss_weight = dn_loss_weight
        self.use_o2m_loss = use_o2m_loss
        self.decouple_attn = decouple_attn
        self.top_k_query=top_k_query
        self.same_o2o = same_o2o
        self.gt_id_key = gt_id_key
        self.gt_occ_key = gt_occ_key
        self.occ_weight = occ_weight
        self.nms_thr = nms_thr
        self.use_gnn_mask = use_gnn_mask
        self.tracking_train = tracking_train
        self.first_o2m = first_o2m
        self.only_o2m = only_o2m
        self.with_o2m = with_o2m
        if with_o2m:
            k=6
        else:
            k=1
        self.matcher_o2m = Stage2Assigner(allow_low_quality_matches=allow_low_quality_matches,k=k,class_score_thr=0.0)
        self.top_k_thr = top_k_thr
        self.prediction_length = 0
        
        
        # if use_vox_atten:
        #     self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding is not None else None
            # self.mask_embed = nn.Embedding(1, embed_dims)

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights
        self.use_occ_loss = use_occ_loss
        if use_occ_loss:
            self.occ_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(occ_weight))


        if operation_order is None:
            operation_order = [
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)
        if loss_next is not None:
            self.loss_next = build(loss_next, LOSSES)
            self.reg_weights_next = reg_weights_next
        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.embed_dims = self.instance_bank.embed_dims
        self.num_temp_instances = self.instance_bank.num_temp_instances
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        # if use_o2m_loss is not None:
        #     self.anchor_encoder_o2m = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.use_prediction = False if predictor is None else True
        if predictor is not None:
            self.predictor = build_from_cfg(predictor, PLUGIN_LAYERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)

        if "deformable_vox" not in self.operation_order:
            deformable_model.update({"use_voxel_feature": use_voxel_feature})            
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],

            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        if "ffn" in self.operation_order:
            self.op_config_map.update({
                "ffn": [ffn, FEEDFORWARD_NETWORK],
            })
        if "deformable" in self.operation_order:
            self.op_config_map.update({
                "deformable": [deformable_model, ATTENTION],
            })
        if "deformable_vox" in self.operation_order:
            self.op_config_map.update({
                "deformable_vox": [deformable_model_vox, ATTENTION],
            })
            if "ffn_vox" in self.operation_order:
                self.op_config_map.update({
                    "ffn_vox": [ffn_vox, FEEDFORWARD_NETWORK],
                })
            if "interaction_gnn" in self.operation_order:
                self.op_config_map.update({
                    "interaction_gnn": [interaction_graph_model, PLUGIN_LAYERS],
                })
            if "interaction_gnn_wo_occ" in self.operation_order:
                self.op_config_map.update({
                    "interaction_gnn_wo_occ": [interaction_graph_model_wo_occ, PLUGIN_LAYERS],
                })
            if "voxel_concat" in self.operation_order:
                deconv_cfg = dict(type='deconv3d', bias=False)
                conv3d_cfg=dict(type='Conv3d', bias=False)
                # gn_norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
                out_dims = self.embed_dims//4
                # self.multi_cat = MultipleInputEmbedding(in_channels=[self.embed_dims, self.embed_dims], out_channel=self.embed_dims)
                conv = build_conv_layer(conv3d_cfg, self.embed_dims*2, self.embed_dims, kernel_size=1, stride=1)
                
                self.cat_block = nn.Sequential(conv,
                                nn.BatchNorm3d(self.embed_dims),
                                nn.ReLU(inplace=True))                 
                
                if not self.use_mask_head:
                # self.cat_block = nn.Sequential(conv,
                #                 build_norm_layer(gn_norm_cfg, self.embed_dims)[1],
                #                 nn.ReLU(inplace=True)) 
                    upsample = build_upsample_layer(deconv_cfg, self.embed_dims, out_dims, kernel_size=2, stride=2)
                    # self.up_block = nn.Sequential(upsample,
                    #                 build_norm_layer(gn_norm_cfg, out_dims)[1],
                    #                 nn.ReLU(inplace=True)
                    #                 )
                    self.up_block = nn.Sequential(upsample,
                                    nn.BatchNorm3d(out_dims),
                                    nn.ReLU(inplace=True)
                                    )
                    self.vox_occ_net = build_conv_layer(
                            conv3d_cfg,
                            in_channels=out_dims,
                            out_channels=self.num_classes,
                            kernel_size=1,
                            stride=1,
                            padding=0)
        if use_o2m_loss:
            if "refine_o2m" in self.operation_order:
                self.op_config_map.update(
                    {
                        "ffn_simple":[ffn_simple, FEEDFORWARD_NETWORK] if ffn_simple is not None else None,
                        "norm_o2m": [norm_layer, NORM_LAYERS],
                        "ffn_o2m": [ffn,FEEDFORWARD_NETWORK] if ffn_o2m is None else [ffn_o2m,FEEDFORWARD_NETWORK],
                        "refine_o2m": [refine_layer, PLUGIN_LAYERS],
                    }
                    )
            else:
                self.op_config_map.update(
                    {
                        "ffn_simple":[ffn_simple, FEEDFORWARD_NETWORK] if ffn_simple is not None else None,
                        "norm_o2m": [norm_layer, NORM_LAYERS],
                        "ffn_o2m": [ffn,FEEDFORWARD_NETWORK] if ffn_o2m is None else [ffn_o2m,FEEDFORWARD_NETWORK],
                    }
                    )
        if "voxel_concat" in self.operation_order:
            self.layers = nn.ModuleList(
                [
                    build(*self.op_config_map.get(op, [None, None]))
                    for op in self.operation_order[:-1]
                ]
            )
        else:
            self.layers = nn.ModuleList(
                [
                    build(*self.op_config_map.get(op, [None, None]))
                    for op in self.operation_order
                ]
            )
        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if op == "voxel_concat":
                continue
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )

    def forward(
        self,
        feature_maps: Union[torch.Tensor, List],
        voxel_feature: torch.Tensor,
        metas: dict,
        occ_list: List[torch.Tensor] = None,
        depth: List[torch.Tensor] = None,
        spatial_shapes: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        occ_pos: torch.Tensor = None,
        ref_3d: torch.Tensor = None,
    ):
        if "voxel_concat" in self.operation_order:
            # voxel_feature = voxel_feature.detach()
            pre_voxel_feature = voxel_feature.clone() # [B, H, W, D, C]
        voxel_pos = None
        voxel_feature_occ = None
        if occ_pos is not None:
            voxel_pos = occ_pos
        # if self.use_vox_atten:
        #     B, Y, X, Z, C = voxel_feature.shape
        #     # mask_token =  self.mask_embed.weight.view(1,1,1,1,C).expand(B,Y,X,Z,C).to(voxel_feature.dtype)
        #     # voxel_feature = torch.where(voxel_feature == 0, mask_token, voxel_feature)            
        #     voxel_pos = self.positional_encoding(torch.zeros((B,Y,X,Z), device=voxel_feature.device).to(voxel_feature.dtype)).to(voxel_feature.dtype) # [B, dim, 100*8, 100]
        #     # voxel_pos = voxel_pos.flatten(2).permute(0,2,1).reshape(B,Y,X,Z,C)
        #     # voxel_feature += voxel_pos
        #     ref_3d = self.get_reference_points(
        #     Y, X, Z, dim='3dCustom', bs=B, device=voxel_feature.device, dtype=voxel_feature.dtype)
        

        if spatial_shapes is None:
            if isinstance(feature_maps, torch.Tensor):
                feature_maps = [feature_maps]
            batch_size = feature_maps[0].shape[0]
        else:
            batch_size = feature_maps.shape[2]
        # ========= get instance info ============
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None
        if self.training:
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
                instance_id,
            ) = self.instance_bank.get(
                batch_size, metas, dn_metas=self.sampler.dn_metas,use_prediction=self.use_prediction
            )
        else:
            (
                instance_feature,
                anchor,
                temp_instance_feature,
                temp_anchor,
                time_interval,
                past_id,
            ) = self.instance_bank.get(
                batch_size, metas, dn_metas=self.sampler.dn_metas,use_prediction=self.use_prediction
            )
        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        
        # if self.instance_bank.instance_id is not None:
        #     past_id = self.instance_bank.instance_id.clone()
        # else:
        #     past_id = None
        # past_id = self.instance_bank.instance_id.clone()
        attn_mask = None
        attn_mask_temp = None
        dn_metas = None
        temp_dn_reg_target = None
        if self.training and hasattr(self.sampler, "get_dn_anchors"):
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    torch.from_numpy(x["instance_id"]).cuda()
                    for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key],
                metas[self.gt_reg_key],
                gt_instance_id,
            )
        if dn_metas is not None:
            (
                dn_anchor,
                dn_reg_target,
                dn_cls_target,
                dn_attn_mask,
                valid_mask,
                dn_id_target,
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )
            anchor = torch.cat([anchor, dn_anchor], dim=1)
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_free_instance, :num_free_instance] = False
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask
            attn_mask_temp = attn_mask.clone()
            if self.only_init and self.use_gnn_mask:
                attn_mask_temp[self.num_temp_instances:num_free_instance, :self.num_temp_instances] = True
        else:
            num_instance = instance_feature.shape[1]
            attn_mask = anchor.new_ones(
                (num_instance, num_instance), dtype=torch.bool
            )
            attn_mask[:num_instance, :num_instance] = False
            attn_mask_temp = attn_mask.clone()
            if self.only_init and self.use_gnn_mask:
                attn_mask_temp[self.num_temp_instances:num_instance, :self.num_temp_instances] = True
        anchor_embed = self.anchor_encoder(anchor)
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        # =================== forward the layers ====================
        prediction = []
        classification = []
        quality = []
        o2m_prediction = []
        o2m_classification = []
        o2m_quality = []
        o2m_instance_feature = None
        anchors = []
        vox_occ_list = [] 
        for i, op in enumerate(self.operation_order):
            # if self.layers[i] is None:
            #     continue
            if op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                    attn_mask= attn_mask_temp
                    if temp_instance_feature is None
                    else None,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                    attn_mask= attn_mask if self.prediction_length==0 else attn_mask_temp,
                )
            elif op == "norm" or op == "ffn" or op=="ffn_o2o" or op=="ffn_first" or op=="ffn_simple" or op=="ffn_vox":
                instance_feature = self.layers[i](instance_feature)
            elif op == "interaction_gnn_wo_occ":
                cls_index = []
                all_filter_flag = [False] * instance_feature.shape[0]
                if self.training:
                    if self.use_nms_filter:
                        
                        o2m_reg = prediction[-1].clone().detach()[:,:num_free_instance][..., : len(self.reg_weights)]
                        o2m_cls = classification[-1].clone().detach()[:,:num_free_instance]
                        # data_reg_target = self.sampler.encode_reg_target(metas[self.gt_reg_key],o2m_reg.device)
                        
                        _, _, _, _, _ ,_,o2o_indices= self.sampler.sample(
                        o2m_cls.clone(),
                        o2m_reg.clone(),
                        metas[self.gt_cls_key],
                        metas[self.gt_reg_key],
                        )

                        o2m_indices,cost_matrix = self.matcher_o2m(o2m_cls,o2m_reg, metas[self.gt_cls_key],metas[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                        if self.with_o2m:
                            for j in range(len(o2m_indices)):
                                # iou_cost = cost_matrix[j][o2o_indices[j][1],o2o_indices[j][0]] # [gt_ind]
                                # iou_mask = (iou_cost > 0.4)
                                indices_j = o2m_indices[j][0]
                                if indices_j.shape[0] == 0:
                                    all_filter_flag[j] = True
                                    indices = torch.tensor([0])
                                    # above_thr = (classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum()
                                    # if above_thr>0:
                                    #     print("non_filtered query but over score 0.3: ",above_thr)
                                else:
                                    # o2m_indices[j][0]에 있는 index 중 o2o_indices[j][0]에 포함되어 있는 index만 가져온다.
                                    # print("filtered number of queries: ",indices_j.shape[0])
                                    # print("over score 0.3: ",(classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum())
                                    indices = indices_j
                                # else:
                                    # indices = torch.unique(torch.cat([o2o_indices[j][0],o2m_indices[j][0]],dim=0))

                                cls_index.append(indices)
                        else:
                            for j in range(len(o2o_indices)):                                
                                iou_cost = cost_matrix[j][o2o_indices[j][1],o2o_indices[j][0]] # [gt_ind]
                                iou_mask = (iou_cost > 0.4)
                                indices = o2o_indices[j][0][iou_mask]
                                if iou_mask.sum() == 0:
                                    indices = torch.tensor([0])
                                    all_filter_flag[j] = True
                                # indices_j = o2m_indices[j][0]
                                # if indices_j.shape[0] == 0:
                                #     all_filter_flag[j] = True
                                #     indices = torch.tensor([0])
                                #     # above_thr = (classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum()
                                #     # if above_thr>0:
                                #     #     print("non_filtered query but over score 0.3: ",above_thr)
                                # else:
                                #     # o2m_indices[j][0]에 있는 index 중 o2o_indices[j][0]에 포함되어 있는 index만 가져온다.
                                #     # print("filtered number of queries: ",indices_j.shape[0])
                                #     # print("over score 0.3: ",(classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum())
                                #     indices = indices_j
                                # # else:
                                #     # indices = torch.unique(torch.cat([o2o_indices[j][0],o2m_indices[j][0]],dim=0))

                                cls_index.append(indices)
                    else:
                        if self.top_k_thr != None:
                            for j in range(classification[-1].shape[0]):
                                values,_ = classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)
                                #sort the values and indices
                                values,sub_indices = torch.sort(values,descending=True)

                                indices = sub_indices[values>self.top_k_thr]
                                if indices.shape[0] == 0:
                                    indices =  sub_indices[0:1]
                                else:
                                    if indices.shape[0]>=self.top_k_query:
                                        indices = indices[:self.top_k_query]
                                cls_index.append(indices)
                        else:
                            _,cls_index = classification[-1][:,:num_free_instance].max(dim=-1)[0].topk(self.top_k_query)
                else:
                    num_free_instance = instance_feature.shape[1]

                    if self.use_nms_filter:
                        mask = (classification[-1].sigmoid().max(dim=-1)[0]>self.nms_thr)
                        cls_index = [torch.where(mask)[1]]
                    else:
                        if self.top_k_thr != None:
                            for j in range(classification[-1].shape[0]):
                                values,_ = classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)
                                #sort the values and indices
                                values,sub_indices = torch.sort(values,descending=True)

                                indices = sub_indices[values>self.top_k_thr]
                                if indices.shape[0] == 0:
                                    indices =  sub_indices[0:1]
                                else:
                                    if indices.shape[0]>=self.top_k_query:
                                        indices = indices[:self.top_k_query]
                                cls_index.append(indices)
                        else:
                            _,cls_index = classification[-1][:,:num_free_instance].max(dim=-1)[0].topk(self.top_k_query)
                query_feature = instance_feature[:,:num_free_instance].clone()
                query_anchor_embed = anchor_embed[:,:num_free_instance].clone()
                for j,flag_mask in enumerate(all_filter_flag):
                    if flag_mask:
                        query_feature[j] = torch.zeros_like(query_feature[j])
                        query_anchor_embed[j] = torch.zeros_like(query_anchor_embed[j])
                voxel_feature,_ = self.layers[i](
                    # instance_feature[:,:num_free_instance].clone().detach(),
                    query_feature,
                    voxel_feature.clone(),
                    anchor[:,:num_free_instance],
                    query_anchor_embed,
                    cls_index,
                    metas= metas,
                    img_feats= feature_maps,
                    ref_3d= ref_3d.clone() if ref_3d is not None else None,
                    voxel_pos = voxel_pos,
                )
            elif op == "interaction_gnn":
                cls_index = []
                all_filter_flag = [False] * instance_feature.shape[0]
                if self.training:
                    if self.use_nms_filter:
                        
                        o2m_reg = prediction[-1].clone().detach()[:,:num_free_instance][..., : len(self.reg_weights)]
                        o2m_cls = classification[-1].clone().detach()[:,:num_free_instance]
                        # data_reg_target = self.sampler.encode_reg_target(metas[self.gt_reg_key],o2m_reg.device)
                        
                        _, _, _, _, _ ,_,o2o_indices= self.sampler.sample(
                        o2m_cls.clone(),
                        o2m_reg.clone(),
                        metas[self.gt_cls_key],
                        metas[self.gt_reg_key],
                        )

                        o2m_indices,cost_matrix = self.matcher_o2m(o2m_cls,o2m_reg, metas[self.gt_cls_key],metas[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                        for j in range(len(o2m_indices)):
                            # iou_cost = cost_matrix[j][o2o_indices[j][1],o2o_indices[j][0]] # [gt_ind]
                            # iou_mask = (iou_cost > 0.4)
                            indices_j = o2m_indices[j][0]
                            if indices_j.shape[0] == 0:
                                all_filter_flag[j] = True
                                indices = torch.tensor([0])
                                # above_thr = (classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum()
                                # if above_thr>0:
                                #     print("non_filtered query but over score 0.3: ",above_thr)
                            else:
                                # o2m_indices[j][0]에 있는 index 중 o2o_indices[j][0]에 포함되어 있는 index만 가져온다.
                                # print("filtered number of queries: ",indices_j.shape[0])
                                # print("over score 0.3: ",(classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)[0]>0.3).sum())
                                indices = indices_j
                            # else:
                                # indices = torch.unique(torch.cat([o2o_indices[j][0],o2m_indices[j][0]],dim=0))

                            cls_index.append(indices)
                    else:
                        if self.top_k_thr != None:
                            for j in range(classification[-1].shape[0]):
                                values,_ = classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)
                                #sort the values and indices
                                values,sub_indices = torch.sort(values,descending=True)

                                indices = sub_indices[values>self.top_k_thr]
                                if indices.shape[0] == 0:
                                    indices =  sub_indices[0:1]
                                else:
                                    if indices.shape[0]>=self.top_k_query:
                                        indices = indices[:self.top_k_query]
                                cls_index.append(indices)
                        else:
                            _,cls_index = classification[-1][:,:num_free_instance].max(dim=-1)[0].topk(self.top_k_query)
                else:
                    num_free_instance = instance_feature.shape[1]

                    if self.use_nms_filter:
                        mask = (classification[-1].sigmoid().max(dim=-1)[0]>self.nms_thr)
                        cls_index = [torch.where(mask)[1]]
                    else:
                        if self.top_k_thr != None:
                            for j in range(classification[-1].shape[0]):
                                values,_ = classification[-1][j,:num_free_instance].sigmoid().max(dim=-1)
                                #sort the values and indices
                                values,sub_indices = torch.sort(values,descending=True)

                                indices = sub_indices[values>self.top_k_thr]
                                if indices.shape[0] == 0:
                                    indices =  sub_indices[0:1]
                                else:
                                    if indices.shape[0]>=self.top_k_query:
                                        indices = indices[:self.top_k_query]
                                cls_index.append(indices)
                        else:
                            _,cls_index = classification[-1][:,:num_free_instance].max(dim=-1)[0].topk(self.top_k_query)
                query_feature = instance_feature[:,:num_free_instance].clone()
                query_anchor_embed = anchor_embed[:,:num_free_instance].clone()
                for j,flag_mask in enumerate(all_filter_flag):
                    if flag_mask:
                        query_feature[j] = torch.zeros_like(query_feature[j])
                        query_anchor_embed[j] = torch.zeros_like(query_anchor_embed[j])
                voxel_feature,vox_occ = self.layers[i](
                    # instance_feature[:,:num_free_instance].clone().detach(),
                    query_feature,
                    voxel_feature.clone(),
                    anchor[:,:num_free_instance],
                    query_anchor_embed, 
                    cls_index,
                    metas= metas,
                    img_feats= feature_maps,
                    ref_3d= ref_3d.clone() if ref_3d is not None else None,
                    voxel_pos = voxel_pos,
                )
                if vox_occ is not None:
                    vox_occ_list.append(vox_occ)
            elif op =="ffn_o2m" or op =="norm_o2m":
                if self.use_o2m_loss:
                    if op == "norm_o2m":
                        o2m_instance_feature = self.layers[i](o2m_instance_feature)
                    else:
                        o2m_instance_feature = self.layers[i](instance_feature)
                else:
                    continue
            elif op == "deformable" or op == "deformable_vox":
                instance_feature = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    feature_maps,
                    voxel_feature,
                    metas,
                    img_feats=feature_maps,
                    depth = depth,
                    spatial_shapes = spatial_shapes,
                    level_start_index = level_start_index,
                )
            
            
            elif op == "voxel_concat":
                voxel_feature_occ = self.cat_block(torch.cat([voxel_feature.permute(0,4,1,2,3),pre_voxel_feature.permute(0,4,1,2,3)],dim=1)) # B, H,W,D,C -> B, C, H, W, D 
                if self.save_final_voxel_feature:
                    self.instance_bank.cached_vox_feature = voxel_feature_occ.clone().permute(0,1,4,2,3)
                if not self.use_mask_head and self.training:
                    up_voxel_feature = self.up_block(voxel_feature_occ) # B, C, H, W, D
                    voxel_occ = self.vox_occ_net(up_voxel_feature).permute(0,1,3,2,4) # B, C, W, H, D
                    vox_occ_list.append(voxel_occ)
                    
            elif op == "refine_o2m":
                if o2m_instance_feature is not None:
                    o2m_anchor, o2m_cls, o2m_qt = self.layers[i](
                        o2m_instance_feature,
                        anchor,
                        anchor_embed,
                        time_interval=time_interval,
                        # return_cls=(
                        #     self.training
                        #     or self.prediction_length == self.num_single_frame_decoder - 1
                        #     or i == len(self.operation_order) - 2
                        # ),
                    )
                    o2m_prediction.append(o2m_anchor)
                    o2m_classification.append(o2m_cls)
                    o2m_quality.append(o2m_qt)
                else:
                    continue
            elif op == "refine":
                anchors.append(anchor)
                # if self.use_o2m_loss and self.prediction_length != 0 and "refine_o2m" not in self.operation_order and o2m_instance_feature is not None:
                #     o2m_anchor, o2m_cls, o2m_qt = self.layers[i](
                #         o2m_instance_feature,
                #         anchor,
                #         anchor_embed,
                #         time_interval=time_interval,
                #         return_cls=(
                #             self.training
                #             or self.prediction_length == self.num_single_frame_decoder - 1
                #             or i == len(self.operation_order) - 1
                #         ),
                #     )
                #     o2m_prediction.append(o2m_anchor)
                #     o2m_classification.append(o2m_cls)
                #     o2m_quality.append(o2m_qt)
                    
                
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training or 
                        self.prediction_length == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                if self.use_o2m_loss and self.first_o2m:
                    if self.prediction_length == 0:
                        o2m_prediction.append(anchor)
                        o2m_classification.append(cls)
                        o2m_quality.append(qt)
                    else:
                        prediction.append(anchor)
                        classification.append(cls)
                        quality.append(qt)
                else:
                    prediction.append(anchor)
                    classification.append(cls)
                    quality.append(qt)
                self.prediction_length += 1
                
                if self.prediction_length == self.num_single_frame_decoder:
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,
                            temp_dn_cls_target,
                            temp_valid_mask,
                            dn_id_target,
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,
                            self.instance_bank.mask,
                        )
                if "voxel_concat" in self.operation_order:
                    if i != len(self.operation_order) - 2:
                        anchor_embed = self.anchor_encoder(anchor)
                else:
                    if i != len(self.operation_order) - 1:
                        anchor_embed = self.anchor_encoder(anchor)
                if (self.prediction_length > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}
        output.update({"anchor": anchors})
        if len(vox_occ_list)!=0 :
            output.update({"vox_occ": vox_occ_list})
        # split predictions of learnable instances and noisy instances
        # if dn_metas is None:
        #     print("dn_metas is None")
        if dn_metas is not None:
        
            dn_classification = [
                x[:, num_free_instance:] for x in classification
            ]
            classification = [x[:, :num_free_instance] for x in classification]
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]
            quality = [
                x[:, :num_free_instance] if x is not None else None
                for x in quality
            ]
            if self.use_o2m_loss:
                o2m_dn_classification = [x[:, num_free_instance:] for x in o2m_classification]
                o2m_dn_prediction = [x[:, num_free_instance:] for x in o2m_prediction]
                if self.first_o2m:
                    dn_classification = o2m_dn_classification + dn_classification
                    dn_prediction = o2m_dn_prediction + dn_prediction
                else:
                    dn_classification = dn_classification + o2m_dn_classification
                    dn_prediction = dn_prediction + o2m_dn_prediction
                    
                if self.only_init:
                    o2m_prediction = [x[:, :num_free_instance]if i==0 else x[:, self.num_temp_instances:num_free_instance] for i,x in enumerate(o2m_prediction)]
                    o2m_classification = [x[:, :num_free_instance]if i==0 else x[:, self.num_temp_instances:num_free_instance] for i,x in enumerate(o2m_classification)]
                    o2m_quality = [x[:, :num_free_instance]if i==0 else x[:, self.num_temp_instances:num_free_instance] for i,x in enumerate(o2m_quality)]
                    
                else:
                    o2m_prediction = [x[:, :num_free_instance] for x in o2m_prediction]
                    o2m_classification = [x[:, :num_free_instance] for x in o2m_classification]
                    o2m_quality = [x[:, :num_free_instance] for x in o2m_quality]
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # cache dn_metas for temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature,
                dn_anchor,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
            }
        )    
        
        if self.use_o2m_loss:
            output.update(
                {
                    "o2m_prediction": o2m_prediction,
                    "o2m_classification": o2m_classification,
                    "o2m_quality": o2m_quality,
                }    
            )
        if self.use_occ_loss:
            output.update(
                {
                    "occ_list":occ_list,
                }
            )
        # prediction module
        if self.use_prediction:
            if temp_instance_feature is not None:
                past_pos = torch.zeros_like(anchor[...,:8])
                past_instance_feature = torch.zeros_like(instance_feature)
                past_instance_feature[:,:600] = temp_instance_feature
                past_pos[:,:600] = torch.cat([temp_anchor[...,:3],temp_anchor[...,6:]],dim=-1)
            else:
                past_instance_feature = torch.zeros_like(instance_feature)
                past_pos = torch.zeros_like(anchor[...,:8])
            current_pos = torch.cat([anchor[...,:3],anchor[...,6:]],dim=-1)
            current_instance_feature = instance_feature
            B,N,pos_dim = past_pos.shape
            _,_,embed_dim = past_instance_feature.shape
            past_pos = past_pos.reshape(-1,pos_dim)
            past_instance_feature = past_instance_feature.reshape(-1,embed_dim)
            current_pos = current_pos.reshape(-1,pos_dim)
            current_instance_feature = current_instance_feature.reshape(-1,embed_dim)
            next_pos = self.predictor(past_pos,past_instance_feature,current_pos,current_instance_feature)
            next_pos = next_pos.reshape(B,N,-1)
            output["next_pos"] = next_pos
            next_anchor = anchor.clone()
            next_anchor[...,:3] += next_pos[...,:3]
            next_anchor[...,6:8] += next_pos[...,3:]
            # for i, indexes in enumerate(nonzero_indices):
            #     pdb.set_trace()
            #     anchor[i, indexes, :3] += next_pos[i, indexes, :3]
            #     anchor[i, indexes, 6:8] += next_pos[i, indexes, 3:]
            
        # cache current instances for temporal modeling
            self.instance_bank.cache(
            instance_feature, next_anchor, cls, metas, feature_maps
            )   
        else:
            self.instance_bank.cache(
                instance_feature, anchor, cls, metas, feature_maps
            )


        # if not self.training:
        #     instance_id = self.instance_bank.get_instance_id(
        #         cls, anchor, self.decoder.score_threshold
        #     )
        # else:
        #     if instance_id is None:
        #         instance_id = cls.new_full(cls.shape, -1)[...,0].long()
        # output["instance_id"] = instance_id
        self.prediction_length = 0
        if voxel_feature_occ == None:
            voxel_feature_occ = voxel_feature.permute(0,4,1,2,3)
        output['instance_feature'] = instance_feature
        return output , voxel_feature_occ
    

    
    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        # ===================== prediction losses ======================
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        quality = model_outs["quality"]
        # instance_id = model_outs["instance_id"]
        matching_indices = None
        cal_o2m_loss = False
        if "o2m_classification" in model_outs:
            cal_o2m_loss = True
            o2m_cls_scores = model_outs["o2m_classification"]
            o2m_reg_preds = model_outs["o2m_prediction"]
            o2m_quality = model_outs["o2m_quality"]
            
            
            
        output = {}
        # gt_idx = [torch.tensor(idx["instance_id"]) for idx in data["img_metas"]]
        if self.tracking_train:
            for decoder_idx, (cls, reg, qt) in enumerate(
                zip(cls_scores, reg_preds, quality)
            ):                    
                reg = reg[..., : len(self.reg_weights)]
                if decoder_idx == 0:
                    cls_target, reg_target, reg_weights, _, _ ,_,_= self.sampler.sample(
                        cls,
                        reg,
                        data[self.gt_cls_key],
                        data[self.gt_reg_key],
                    )
                    reg_target = reg_target[..., : len(self.reg_weights)]
                    mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
                elif decoder_idx == 1:
                    cls_target, reg_target, reg_weights,instance_id_track,reg_target_next,output_reg_weights_next,_ = self.sampler.sample(
                        cls,
                        reg,
                        data[self.gt_cls_key],
                        data[self.gt_reg_key],
                        instance_id,
                        # gt_idx,
                        data['future_gt_bboxes_3d'] if 'future_gt_bboxes_3d' in data else None,
                        layer_idx = decoder_idx,
                    )
                    reg_target = reg_target[..., : len(self.reg_weights)]
                    mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
                else:
                    cls_target, reg_target, reg_weights,instance_id,reg_target_next,output_reg_weights_next,_ = self.sampler.sample(
                        cls,
                        reg,
                        data[self.gt_cls_key],
                        data[self.gt_reg_key],
                        instance_id_track,
                        # gt_idx,
                        data['future_gt_bboxes_3d'] if 'future_gt_bboxes_3d' in data else None,
                        layer_idx = decoder_idx,
                    )
                    # pdb.set_trace()
                    mask_next = torch.logical_not(torch.all(reg_target_next==0,dim=-1))
                    reg_target_next[...,:3] -= reg_target[...,:3]
                    reg_target_next[...,3:] -= reg_target[...,6:8]
                    reg_target = reg_target[..., : len(self.reg_weights)]
                    mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
                num_pos = max(
                    reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
                )
                if self.cls_threshold_to_reg > 0:
                    threshold = self.cls_threshold_to_reg
                    mask = torch.logical_and(
                        mask, cls.max(dim=-1).values.sigmoid() > threshold
                    )
                cls = cls.flatten(end_dim=1)
                cls_target = cls_target.flatten(end_dim=1)
                cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)
                mask = mask.reshape(-1)
                reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
                reg_target = reg_target.flatten(end_dim=1)[mask]
                reg = reg.flatten(end_dim=1)[mask]
                reg_weights = reg_weights.flatten(end_dim=1)[mask]
                reg_target = torch.where(
                    reg_target.isnan(), reg.new_tensor(0.0), reg_target
                )
                cls_target = cls_target[mask]
                if qt is not None:
                    qt = qt.flatten(end_dim=1)[mask]

                reg_loss = self.loss_reg(
                    reg,
                    reg_target,
                    weight=reg_weights,
                    avg_factor=num_pos,
                    suffix=f"_{decoder_idx}",
                    quality=qt,
                    cls_target=cls_target,
                )

                output[f"loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)
                
            if self.use_prediction:
            #     if self.cls_threshold_to_reg > 0:
            #         threshold = self.cls_threshold_to_reg
            #         mask_next = torch.logical_and(
            #             mask_next, cls_scores[-1][:,:mask_next.shape[1]].max(dim=-1).values.sigmoid() > threshold
            #         )
                # pdb.set_trace()
                reg_next = model_outs["next_pos"]
                reg_weights_next = output_reg_weights_next * reg_next.new_tensor(self.reg_weights_next)
                reg_target_next = reg_target_next[..., : len(self.reg_weights_next)]
                mask_next = mask_next.reshape(-1)
                reg_target_next = reg_target_next.flatten(end_dim=1)[mask_next]
                reg_next = reg_next.flatten(end_dim=1)[mask_next]
                reg_weights_next = reg_weights_next.flatten(end_dim=1)[mask_next]
                reg_next = torch.where(
                    reg_next.isnan(), reg.new_tensor(0.0), reg_next
                )
                num_pos_next = max(
                    reduce_mean(torch.sum(mask_next).to(dtype=reg.dtype)), 1.0
                )
                reg_next_loss = self.loss_next(reg_next, reg_target_next, weight=reg_weights_next, avg_factor=num_pos_next)
                output["loss_next"] = reg_next_loss
                
            self.instance_bank.update_instance_id(instance_id,cls)
                
                
        else:
            o2o_indices_list = []
            for decoder_idx, (cls, reg, qt) in enumerate(
                zip(cls_scores, reg_preds, quality)):
                if self.only_o2m:
                    o2m_reg = reg[..., : len(self.reg_weights)]
                    data_reg_target = self.sampler.encode_reg_target(data[self.gt_reg_key],o2m_reg.device)
                    
                    # o2o_cls_target, o2o_reg_target, o2o_reg_weights, _, _ ,_,o2o_indices= self.sampler.sample(
                    # o2m_cls,
                    # o2m_reg,
                    # data[self.gt_cls_key],
                    # data[self.gt_reg_key],
                    # )
                    # # pdb.set_trace()
                    # if self.same_o2o:
                    #     if self.first_o2m:
                    #         if decoder_idx != 0:
                    #             o2o_indices = o2o_indices_list[decoder_idx-1]
                    #     else:
                    #         o2o_indices = o2o_indices_list[decoder_idx+1]
                    o2m_indices,cost_matrix = self.matcher_o2m(cls,o2m_reg, data[self.gt_cls_key],data[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                    matching_indices = o2m_indices
                    bs, num_pred, num_cls = cls.shape
                    o2m_cls_target = (data[self.gt_cls_key][0].new_ones([bs, num_pred], dtype=torch.long) * num_cls)
                    o2m_reg_target = o2m_reg.new_zeros(o2m_reg.shape)
                    # for o2m_index in o2m_indices:
                    #     if o2m_index[0] is not None:
                    #         o2m_num+=len(o2m_index[0])
                    instance_reg_weights = []
                    for i in range(len(data_reg_target)):
                        weights = torch.logical_not(data_reg_target[i].isnan()).to(
                            dtype=data_reg_target[i].dtype
                        )
                        if self.sampler.cls_wise_reg_weights is not None:
                            for class_num, weight in self.sampler.cls_wise_reg_weights.items():
                                weights = torch.where(
                                    (data[self.gt_cls_key][i] == class_num)[:, None],
                                    weights.new_tensor(weight),
                                    weights,
                                )
                        instance_reg_weights.append(weights)
                    o2m_reg_weights = o2m_reg.new_zeros(o2m_reg.shape)
                    for i in range(len(o2m_indices)):
                        if len(data[self.gt_cls_key][i]) == 0:
                            continue
                        # o2m_reg_target[i,o2o_indices[i][0]] = data_reg_target[i][o2o_indices[i][1]]
                        # o2m_cls_target[i,o2o_indices[i][0]] = data[self.gt_cls_key][i][o2o_indices[i][1]]
                        o2m_reg_target[i,o2m_indices[i][0]] = data_reg_target[i][o2m_indices[i][1]]
                        o2m_cls_target[i,o2m_indices[i][0]] = data[self.gt_cls_key][i][o2m_indices[i][1]]
                        # o2m_reg_weights[i,o2o_indices[i][0]] = instance_reg_weights[i][o2o_indices[i][1]]
                        o2m_reg_weights[i, o2m_indices[i][0]] = instance_reg_weights[i][o2m_indices[i][1]]
            
                    o2m_mask = torch.logical_not(torch.all(o2m_reg_target == 0, dim=-1))
                    o2m_mask_valid = o2m_mask.clone()
                    o2m_num_pos = max(reduce_mean(torch.sum(o2m_mask).to(dtype=o2m_reg.dtype)), 1.0)
                    
                    if self.cls_threshold_to_reg > 0:
                        threshold = self.cls_threshold_to_reg
                        o2m_mask = torch.logical_and(
                            o2m_mask, cls.max(dim=-1).values.sigmoid() > threshold
                        )

                    o2m_cls = cls.flatten(end_dim=1)
                    o2m_cls_target = o2m_cls_target.flatten(end_dim=1)
                    # o2m_cls_loss = self.loss_cls(o2m_cls.detach(), o2m_cls_target, avg_factor=o2m_num_pos)
                    cls_loss = self.loss_cls(o2m_cls, o2m_cls_target, avg_factor=o2m_num_pos)
                                            
                    o2m_mask = o2m_mask.reshape(-1)
                    o2m_reg_weights = o2m_reg_weights * o2m_reg.new_tensor(self.reg_weights)
                    o2m_reg_target = o2m_reg_target.flatten(end_dim=1)[o2m_mask]
                    o2m_reg = o2m_reg.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_weights = o2m_reg_weights.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_target = torch.where(
                        o2m_reg_target.isnan(), o2m_reg.new_tensor(0.0), o2m_reg_target
                    )
                    o2m_cls_target = o2m_cls_target[o2m_mask]
                    
                    if qt is not None:
                        o2m_qt = qt.flatten(end_dim=1)[o2m_mask]
                        
                    reg_loss = self.loss_reg(
                        o2m_reg,
                        o2m_reg_target,
                        weight=o2m_reg_weights,
                        avg_factor=o2m_num_pos,
                        suffix=f"_{decoder_idx}",
                        quality=o2m_qt,
                        cls_target=o2m_cls_target,
                    )
                    # pdb.set_trace()
                elif self.with_o2m:
                    
                    o2m_reg = reg[..., : len(self.reg_weights)]
                    data_reg_target = self.sampler.encode_reg_target(data[self.gt_reg_key],o2m_reg.device)
                    
                    o2o_cls_target, o2o_reg_target, o2o_reg_weights, _, _ ,_,o2o_indices= self.sampler.sample(
                    cls,
                    o2m_reg,
                    data[self.gt_cls_key],
                    data[self.gt_reg_key],
                    )
                    # pdb.set_trace()
                    # if self.same_o2o:
                    #     if self.first_o2m:
                    #         if decoder_idx != 0:
                    #             o2o_indices = o2o_indices_list[decoder_idx-1]
                    #     else:
                    #         o2o_indices = o2o_indices_list[decoder_idx+1]
                    o2m_indices,cost_matrix = self.matcher_o2m(cls,o2m_reg, data[self.gt_cls_key],data[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                    bs, num_pred, num_cls = cls.shape
                    o2m_cls_target = (data[self.gt_cls_key][0].new_ones([bs, num_pred], dtype=torch.long) * num_cls)
                    o2m_reg_target = o2m_reg.new_zeros(o2m_reg.shape)
                    # for o2m_index in o2m_indices:
                    #     if o2m_index[0] is not None:
                    #         o2m_num+=len(o2m_index[0])
                    instance_reg_weights = []
                    for i in range(len(data_reg_target)):
                        weights = torch.logical_not(data_reg_target[i].isnan()).to(
                            dtype=data_reg_target[i].dtype
                        )
                        if self.sampler.cls_wise_reg_weights is not None:
                            for class_num, weight in self.sampler.cls_wise_reg_weights.items():
                                weights = torch.where(
                                    (data[self.gt_cls_key][i] == class_num)[:, None],
                                    weights.new_tensor(weight),
                                    weights,
                                )
                        instance_reg_weights.append(weights)
                    o2m_reg_weights = o2m_reg.new_zeros(o2m_reg.shape)
                    for i in range(len(o2m_indices)):
                        if len(data[self.gt_cls_key][i]) == 0:
                            continue
                        o2m_reg_target[i,o2o_indices[i][0]] = data_reg_target[i][o2o_indices[i][1]]
                        o2m_cls_target[i,o2o_indices[i][0]] = data[self.gt_cls_key][i][o2o_indices[i][1]]
                        o2m_reg_target[i,o2m_indices[i][0]] = data_reg_target[i][o2m_indices[i][1]]
                        o2m_cls_target[i,o2m_indices[i][0]] = data[self.gt_cls_key][i][o2m_indices[i][1]]
                        o2m_reg_weights[i,o2o_indices[i][0]] = instance_reg_weights[i][o2o_indices[i][1]]
                        o2m_reg_weights[i, o2m_indices[i][0]] = instance_reg_weights[i][o2m_indices[i][1]]
                    # matching_indices = o2m_indices
                    # for i in range(len(matching_indices)):
                    #     o2o_index = o2o_indices[i]
                    #     for num in range(o2o_index[0].shape[0]):
                    #         if o2o_index[0][num] not in matching_indices[i][0]:
                    #             matching_indices[i][0].append(o2o_index[num][0])
                    #             matching_indices[i][1].append(o2o_index[num][1])
                                
                        
            
                    o2m_mask = torch.logical_not(torch.all(o2m_reg_target == 0, dim=-1))
                    o2m_mask_valid = o2m_mask.clone()
                    o2m_num_pos = max(reduce_mean(torch.sum(o2m_mask).to(dtype=o2m_reg.dtype)), 1.0)
                    
                    if self.cls_threshold_to_reg > 0:
                        threshold = self.cls_threshold_to_reg
                        o2m_mask = torch.logical_and(
                            o2m_mask, cls.max(dim=-1).values.sigmoid() > threshold
                        )

                    o2m_cls = cls.flatten(end_dim=1)
                    o2m_cls_target = o2m_cls_target.flatten(end_dim=1)
                    # o2m_cls_loss = self.loss_cls(o2m_cls.detach(), o2m_cls_target, avg_factor=o2m_num_pos)
                    cls_loss = self.loss_cls(o2m_cls, o2m_cls_target, avg_factor=o2m_num_pos)
                                            
                    o2m_mask = o2m_mask.reshape(-1)
                    o2m_reg_weights = o2m_reg_weights * o2m_reg.new_tensor(self.reg_weights)
                    o2m_reg_target = o2m_reg_target.flatten(end_dim=1)[o2m_mask]
                    o2m_reg = o2m_reg.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_weights = o2m_reg_weights.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_target = torch.where(
                        o2m_reg_target.isnan(), o2m_reg.new_tensor(0.0), o2m_reg_target
                    )
                    o2m_cls_target = o2m_cls_target[o2m_mask]
                    
                    if qt is not None:
                        o2m_qt = qt.flatten(end_dim=1)[o2m_mask]
                        
                    reg_loss = self.loss_reg(
                        o2m_reg,
                        o2m_reg_target,
                        weight=o2m_reg_weights,
                        avg_factor=o2m_num_pos,
                        suffix=f"_{decoder_idx}",
                        quality=o2m_qt,
                        cls_target=o2m_cls_target,
                    )
                    
                    
                else:
                    reg = reg[..., : len(self.reg_weights)]
                    cls_target, reg_target, reg_weights, _, _ , _, o2o_indices= self.sampler.sample(
                        cls,
                        reg,
                        data[self.gt_cls_key],
                        data[self.gt_reg_key],
                    )
                    o2o_indices_list.append(o2o_indices)
                    matching_indices = o2o_indices
                    reg_target = reg_target[..., : len(self.reg_weights)]
                    mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))
                    mask_valid = mask.clone()
                    num_pos = max(
                        reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0
                    )
                    if self.cls_threshold_to_reg > 0:
                        threshold = self.cls_threshold_to_reg
                        mask = torch.logical_and(mask, cls.max(dim=-1).values.sigmoid() > threshold)
                    cls = cls.flatten(end_dim=1)
                    cls_target = cls_target.flatten(end_dim=1)
                    cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

                    mask = mask.reshape(-1)
                    reg_weights = reg_weights * reg.new_tensor(self.reg_weights)
                    reg_target = reg_target.flatten(end_dim=1)[mask]
                    reg = reg.flatten(end_dim=1)[mask]
                    reg_weights = reg_weights.flatten(end_dim=1)[mask]
                    reg_target = torch.where(
                        reg_target.isnan(), reg.new_tensor(0.0), reg_target
                    )
                    cls_target = cls_target[mask]
                    if qt is not None:
                        qt = qt.flatten(end_dim=1)[mask]
                    reg_loss = self.loss_reg(
                        reg,
                        reg_target,
                        weight=reg_weights,
                        avg_factor=num_pos,
                        suffix=f"_{decoder_idx}",
                        quality=qt,
                        cls_target=cls_target,
                    )

                output[f"loss_cls_{decoder_idx}"] = cls_loss
                output.update(reg_loss)
            o2m_num = 0
            if cal_o2m_loss :
                for decoder_idx, (o2m_cls, o2m_reg, o2m_qt) in enumerate(
                    zip(o2m_cls_scores, o2m_reg_preds, o2m_quality)
                ):
                    o2m_reg = o2m_reg[..., : len(self.reg_weights)]
                    data_reg_target = self.sampler.encode_reg_target(data[self.gt_reg_key],o2m_reg.device)
                    
                    o2o_cls_target, o2o_reg_target, o2o_reg_weights, _, _ ,_,o2o_indices= self.sampler.sample(
                    o2m_cls,
                    o2m_reg,
                    data[self.gt_cls_key],
                    data[self.gt_reg_key],
                    )
                    # pdb.set_trace()
                    if self.same_o2o:
                        if self.first_o2m:
                            if decoder_idx != 0:
                                o2o_indices = o2o_indices_list[decoder_idx-1]
                        else:
                            o2o_indices = o2o_indices_list[decoder_idx+1]
                    o2m_indices,cost_matrix = self.matcher_o2m(o2m_cls,o2m_reg, data[self.gt_cls_key],data[self.gt_reg_key],return_cost_matrix=True) # [pred_ind,gt_ind]
                    bs, num_pred, num_cls = o2m_cls.shape
                    o2m_cls_target = (data[self.gt_cls_key][0].new_ones([bs, num_pred], dtype=torch.long) * num_cls)
                    o2m_reg_target = o2m_reg.new_zeros(o2m_reg.shape)
                    for o2m_index in o2m_indices:
                        if o2m_index[0] is not None:
                            o2m_num+=len(o2m_index[0])

                    instance_reg_weights = []
                    for i in range(len(data_reg_target)):
                        weights = torch.logical_not(data_reg_target[i].isnan()).to(
                            dtype=data_reg_target[i].dtype
                        )
                        if self.sampler.cls_wise_reg_weights is not None:
                            for class_num, weight in self.sampler.cls_wise_reg_weights.items():
                                weights = torch.where(
                                    (data[self.gt_cls_key][i] == class_num)[:, None],
                                    weights.new_tensor(weight),
                                    weights,
                                )
                        instance_reg_weights.append(weights)
                    o2m_reg_weights = o2m_reg.new_zeros(o2m_reg.shape)
                    for i in range(len(o2m_indices)):
                        if len(data[self.gt_cls_key][i]) == 0:
                            continue
                        o2m_reg_target[i,o2o_indices[i][0]] = data_reg_target[i][o2o_indices[i][1]]
                        o2m_cls_target[i,o2o_indices[i][0]] = data[self.gt_cls_key][i][o2o_indices[i][1]]
                        o2m_reg_target[i,o2m_indices[i][0]] = data_reg_target[i][o2m_indices[i][1]]
                        o2m_cls_target[i,o2m_indices[i][0]] = data[self.gt_cls_key][i][o2m_indices[i][1]]
                        o2m_reg_weights[i,o2o_indices[i][0]] = instance_reg_weights[i][o2o_indices[i][1]]
                        o2m_reg_weights[i, o2m_indices[i][0]] = instance_reg_weights[i][o2m_indices[i][1]]
            
                    o2m_mask = torch.logical_not(torch.all(o2m_reg_target == 0, dim=-1))
                    o2m_mask_valid = o2m_mask.clone()
                    o2m_num_pos = max(reduce_mean(torch.sum(o2m_mask).to(dtype=o2m_reg.dtype)), 1.0)
                    
                    if self.cls_threshold_to_reg > 0:
                        threshold = self.cls_threshold_to_reg
                        o2m_mask = torch.logical_and(
                            o2m_mask, o2m_cls.max(dim=-1).values.sigmoid() > threshold
                        )

                    o2m_cls = o2m_cls.flatten(end_dim=1)
                    o2m_cls_target = o2m_cls_target.flatten(end_dim=1)
                    # o2m_cls_loss = self.loss_cls(o2m_cls.detach(), o2m_cls_target, avg_factor=o2m_num_pos)
                    o2m_cls_loss = self.loss_cls(o2m_cls, o2m_cls_target, avg_factor=o2m_num_pos)
                                            
                    o2m_mask = o2m_mask.reshape(-1)
                    o2m_reg_weights = o2m_reg_weights * o2m_reg.new_tensor(self.reg_weights)
                    o2m_reg_target = o2m_reg_target.flatten(end_dim=1)[o2m_mask]
                    o2m_reg = o2m_reg.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_weights = o2m_reg_weights.flatten(end_dim=1)[o2m_mask]
                    o2m_reg_target = torch.where(
                        o2m_reg_target.isnan(), o2m_reg.new_tensor(0.0), o2m_reg_target
                    )
                    o2m_cls_target = o2m_cls_target[o2m_mask]
                    
                    if o2m_qt is not None:
                        o2m_qt = o2m_qt.flatten(end_dim=1)[o2m_mask]
                        
                    o2m_reg_loss = self.loss_reg(
                        o2m_reg,
                        o2m_reg_target,
                        weight=o2m_reg_weights,
                        avg_factor=o2m_num_pos,
                        suffix=f"_{decoder_idx}_o2m",
                        quality=o2m_qt,
                        cls_target=o2m_cls_target,
                    )

                    output[f"loss_cls_{decoder_idx}_o2m"] = o2m_cls_loss
                    output.update(o2m_reg_loss)
            # # pdb.set_trace()
            #     if o2m_num > 0 :
            #         print("o2m_num: ",o2m_num)
            #         print("=====================================")
            #         o2m_num = 0
            if self.use_occ_loss:                
                occ_list = model_outs["occ_list"]
                occ_gt = torch.stack(data[self.gt_occ_key]).flatten(1).to(torch.float32)
                for voxel_index, occ in enumerate(occ_list):
                    occ = occ.sigmoid().squeeze(-1)
                    occ_loss = self.occ_loss(occ, occ_gt)
                    output[f"loss_occ_{voxel_index}"] = occ_loss
                    
        if "dn_prediction" not in model_outs:
            return output,matching_indices
        

        # ===================== denoising losses ======================
        dn_cls_scores = model_outs["dn_classification"]
        dn_reg_preds = model_outs["dn_prediction"]

        (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_idx, (cls, reg) in enumerate(
            zip(dn_cls_scores, dn_reg_preds)
        ):
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            cls_loss = self.loss_cls(
                cls.flatten(end_dim=1)[dn_valid_mask],
                dn_cls_target,
                avg_factor=num_dn_pos,
            )
            if cal_o2m_loss:
                if self.first_o2m:
                    if decoder_idx < len(o2m_reg_preds):
                        reg_loss = self.loss_reg(
                            reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                                ..., : len(self.reg_weights)
                            ],
                            dn_reg_target,
                            avg_factor=num_dn_pos,
                            weight=reg_weights,
                            suffix=f"_dn_o2m_{decoder_idx}",
                        )
                        output[f"loss_cls_dn_o2m_{decoder_idx}"] = cls_loss
                        output.update(reg_loss)
                    else:
                        reg_loss = self.loss_reg(
                            reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                                ..., : len(self.reg_weights)
                            ],
                            dn_reg_target,
                            avg_factor=num_dn_pos,
                            weight=reg_weights,
                            suffix=f"_dn_{decoder_idx-len(o2m_reg_preds)}",
                        )
                        output[f"loss_cls_dn_{decoder_idx-len(o2m_reg_preds)}"] = cls_loss
                        output.update(reg_loss)
                else:
                    if decoder_idx < len(reg_preds):
                        reg_loss = self.loss_reg(
                            reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                                ..., : len(self.reg_weights)
                            ],
                            dn_reg_target,
                            avg_factor=num_dn_pos,
                            weight=reg_weights,
                            suffix=f"_dn_{decoder_idx}",
                        )
                        output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
                        output.update(reg_loss)
                    else:
                        reg_loss = self.loss_reg(
                            reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                                ..., : len(self.reg_weights)
                            ],
                            dn_reg_target,
                            avg_factor=num_dn_pos,
                            weight=reg_weights,
                            suffix=f"_dn_o2m_{decoder_idx-len(o2m_reg_preds)}",
                        )
                        output[f"loss_cls_dn_o2m_{decoder_idx-len(o2m_reg_preds)}"] = cls_loss
                        output.update(reg_loss)
                        
            else:
                reg_loss = self.loss_reg(
                    reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                        ..., : len(self.reg_weights)
                    ],
                    dn_reg_target,
                    avg_factor=num_dn_pos,
                    weight=reg_weights,
                    suffix=f"_dn_{decoder_idx}",
                )
                output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
                output.update(reg_loss)
        return output,matching_indices

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(
            end_dim=1
        )[dn_valid_mask]
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(
            end_dim=1
        )[dn_valid_mask][..., : len(self.reg_weights)]
        dn_pos_mask = dn_cls_target >= 0
        dn_reg_target = dn_reg_target[dn_pos_mask]
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)),
            1.0,
        )
        return (
            dn_valid_mask,
            dn_cls_target,
            dn_reg_target,
            dn_pos_mask,
            reg_weights,
            num_dn_pos,
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1):
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )


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