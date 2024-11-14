
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.utils.checkpoint as cp
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer, Conv3d, xavier_init
from mmcv.cnn.utils.weight_init import constant_init
from mmcv.cnn.bricks.conv_module import ConvModule
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import build_assigner, build_sampler, multi_apply, reduce_mean
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmcv.cnn.bricks.transformer import build_positional_encoding, build_transformer_layer_sequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import mmcv
import cv2 as cv
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss, build_head
from mmcv.runner import BaseModule, force_fp32
import pdb

@HEADS.register_module()
class MaskHead(BaseModule):
    """Head of COTR. (revised)
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 in_channels=64,
                 embed_dims=256,
                 num_classes=18,
                 num_query=100,
                 group_detr=1,
                 group_classes=[17],
                 transformer=None, # IVT transformer
                 transformer_decoder=None, # GroupDecoder
                 predictor=None,
                 train_cfg=None,
                 test_cfg=None,
                 loss_occ=None,
                 cls_freq=None,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 use_camera_mask=False,
                 use_lidar_mask=False,
                 positional_encoding=None,
                 use_COTR_version=False,
                 save_high_res_feature=False,
                 only_background=False,
                 save_high_res_feature_all_0=False,
                 concat_final = False,
                 **kwargs):
        super(MaskHead, self).__init__()

        self.fp16_enabled = False
        self.concat_final = concat_final
        self.use_COTR_version = use_COTR_version
        self.use_camera_mask = use_camera_mask
        self.only_background = only_background
        self.use_lidar_mask = use_lidar_mask
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_queries = num_query * group_detr
        self.group_detr = group_detr
        self.save_high_res_feature_all_0=save_high_res_feature_all_0
        self.save_high_res_feature = save_high_res_feature
        self.group_classes = group_classes
        self.embed_dims = embed_dims
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg is not None:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)

        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.loss_occ = build_loss(loss_occ)
        if cls_freq is not None:
            self.cls_freq = cls_freq
            cls_weight = torch.from_numpy(1 / np.log(np.array(self.cls_freq) + 0.001))
            self.loss_occ.class_weight = cls_weight
        # self.decoder_input_projs = Conv3d(in_channels, embed_dims, kernel_size=1)
        # self.enhance_conv = ConvModule(
        #                 in_channels*2,
        #                 in_channels,
        #                 kernel_size=3,
        #                 stride=1,
        #                 padding=1,
        #                 bias=True,
        #                 conv_cfg=dict(type='Conv3d'))
        if transformer is not None:
            self.transformer = build_transformer(transformer)
            self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding is not None else None
            self.enhance_projs = Conv3d(embed_dims, embed_dims, kernel_size=1)

        else:
            self.transformer = None
            self.positional_encoding = None
        self.transformer_decoder = build_transformer_layer_sequence(transformer_decoder)
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims*2)
        self.reference_points = nn.Linear(self.embed_dims, 3)
        self.predictor = build_head(predictor)
        # not elegant
        if use_COTR_version:
            self.compact_proj = Conv3d(in_channels, embed_dims, kernel_size=1)
            self.enhance_projs = Conv3d(embed_dims, in_channels, kernel_size=1)
            self.up0 = nn.Sequential(
                nn.ConvTranspose3d(in_channels*5,in_channels*2,(1,3,3),padding=(0,1,1)),
                nn.BatchNorm3d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels*2, in_channels*2, (1, 2, 2), stride=(1, 2, 2)),
                nn.BatchNorm3d(in_channels*2),
                nn.ReLU(inplace=True),
            )
            self.up1 = nn.Sequential(
                nn.ConvTranspose3d(in_channels*4,in_channels*2,(1,3,3),padding=(0,1,1)),
                nn.BatchNorm3d(in_channels*2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose3d(in_channels*2, in_channels*2, (1, 2, 2), stride=(1, 2, 2)),
                nn.BatchNorm3d(in_channels*2),
                nn.ReLU(inplace=True),
            )
            self.final_conv = ConvModule(
                            in_channels*3,
                            embed_dims,
                            kernel_size=1,
                            stride=1,
                            # padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'))
        else:
            self.up1 = nn.Sequential(
                # nn.ConvTranspose3d(embed_dims,embed_dims//2,(3,3,3),padding=(1,1,1)),
                # nn.BatchNorm3d(embed_dims//2),
                # nn.ReLU(inplace=True),
                nn.ConvTranspose3d(embed_dims, embed_dims//2, (2, 2, 2), stride=(2, 2, 2)),
                nn.BatchNorm3d(embed_dims//2),
                nn.ReLU(inplace=True),
            )
            if self.concat_final:
                self.final_conv = ConvModule(
                            (embed_dims//2)+in_channels,
                            embed_dims,
                            kernel_size=1,
                            stride=1,
                            # padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv3d'))
            else:
                self.final_conv = ConvModule(
                                embed_dims//2,
                                embed_dims,
                                kernel_size=1,
                                stride=1,
                                # padding=1,
                                bias=True,
                                conv_cfg=dict(type='Conv3d'))
        if self.save_high_res_feature:
            self.occ_predictor_1 = nn.Sequential(
                        nn.Linear(embed_dims, in_channels),
                        nn.Softplus())
            self.occ_predictor_2 =nn.Linear(in_channels, num_classes+1)
        else:
            self.occ_predictor_1 = nn.Sequential(
                        nn.Linear(embed_dims, in_channels//2),
                        nn.Softplus())
            self.occ_predictor_2 =nn.Linear(in_channels//2, num_classes+1)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.transformer is not None:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, occ_feature, voxel_feature_list=None,mlvl_feats=None,threshold=0.3,instance_queries=None,origin_voxel_feat=None, **kwargs):
        """Forward function.
        Args:
            img_feats: [occ_feature, img_feature]
            occ_feature (tuple[Tensor]): Occ Features from the upstream
                network, each is a 5D-tensor with shape
                (B, C, Z, H, W).
            img_feature (Tensor): img features of current frame
            depth (Tensor): depth prediction of the img features
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        bs = occ_feature.shape[0]
        dtype = occ_feature.dtype

        if voxel_feature_list is not None and self.use_COTR_version:
            occ2, occ1, occ0 = voxel_feature_list  # occ?: [B, C, Z, H, W]
        if mlvl_feats is not None:
            if self.transformer is not None:
                occ_h, occ_w, occ_z = occ_feature.shape[-3:]
                # -----------------Encoder---------------------
                occ_queries = occ_feature.flatten(2).permute(0, 2, 1) # [b, h*w*z, C]
                occ_mask = torch.zeros((bs, occ_h, occ_w, occ_z),
                                       device=occ_queries.device).to(dtype)
                # if self.transformer is not None:
                occ_pos = self.positional_encoding(occ_mask, 1).to(dtype)
                enhanced_occ_feature = self.transformer(
                                            mlvl_feats,
                                            occ_queries,
                                            occ_h,
                                            occ_w,
                                            occ_z,
                                            occ_pos=occ_pos,
                                            **kwargs) # [b, h*w, c]
                enhanced_occ_feature = enhanced_occ_feature.permute(0, 2, 1).view(bs, -1, occ_h, occ_w, occ_z)
                compact_occ = self.enhance_projs(enhanced_occ_feature).permute(0,1,4,2,3)   # [b, c, h, w, z] -> [b, c, z, h, w]
            else:
                if self.use_COTR_version:
                    compact_occ = self.enhance_projs(occ_feature.permute(0, 1, 4, 3, 2))
                else:
                    compact_occ = occ_feature.permute(0, 1, 4, 2, 3) #[b, c, h, w, z] -> [b, c', z, h, w]

                # compact_occ = self.enhance_projs(occ_feature.permute(0, 1, 4, 2, 3)) # [b, c, h, w, z] -> [b, c', z, h, w]

        if self.use_COTR_version is False:
            occ_1 = self.up1(compact_occ)
            if self.concat_final:
                occ_1 = torch.cat([occ_1, origin_voxel_feat], dim=1)
                fullres_occ = self.final_conv(occ_1).permute(0, 1, 4, 3, 2) # bczhw -> bcwhz
            else:
                fullres_occ = self.final_conv(occ_1).permute(0, 1, 4, 3, 2) # bczhw -> bcwhz
            # compact_occ = self.enhance_projs(compact_occ)
        else:
            occ_0 = self.up0(torch.cat([compact_occ, occ0], dim=1))
            occ1 = F.interpolate(occ1, size=occ_0.shape[-3:],mode='trilinear', align_corners=True)
            occ_1 = self.up1(torch.cat([occ_0, occ1], dim=1))

            fullres_occ = self.final_conv(torch.cat([occ2, occ_1], dim=1)).permute(0, 1, 4, 3, 2) # bczhw -> bcwhz
            compact_occ = self.compact_proj(compact_occ)
        # -----------------Encoder---------------------
        
        # with only encoder, we can speed up inference 
        # and have a coarse occ results
        compact_occ = compact_occ.permute(0, 1, 4, 3, 2) #bczhw -> bcwhz

        if self.training or not self.test_cfg.only_encoder:
            # -----------------Decoder---------------------
            object_query_embeds = self.query_embedding.weight.to(dtype)
            if not self.training:  # NOTE: Only difference to normal head
                object_query_embeds = object_query_embeds[:self.num_queries // self.group_detr]
            object_query_pos, object_query = torch.split(
                object_query_embeds, self.embed_dims, dim=1)
            object_query_pos = object_query_pos.unsqueeze(0).expand(bs, -1, -1)
            object_query = object_query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(object_query_pos)[:, :, None, :]# [bs, num_query, num_level, 3]
            reference_points = reference_points.sigmoid()

            # [bs, n, c] --> [n, bs, c]
            object_query = object_query.permute(1, 0, 2)
            object_query_pos = object_query_pos.permute(1, 0, 2)
            # compact_occ = self.compact_proj(compact_occ)
            w, h, z = compact_occ.shape[-3:]
            # [bs, c, w, h, z] --> [w*h*z, bs, c]
            occ_value = compact_occ.flatten(-3).permute(2, 0, 1)

            decoder_spatial_shapes = torch.tensor([
                                        [w, h, z],
                                    ], device=object_query.device)
            lsi = torch.tensor([0,], device=object_query.device)

            decoder_out = self.transformer_decoder(
                                query=object_query,
                                key=None,
                                value=occ_value,
                                query_pos=object_query_pos,
                                reference_points=reference_points,
                                spatial_shapes=decoder_spatial_shapes,
                                level_start_index=lsi,
                                **kwargs
                            )
            # -----------------Decoder---------------------

        # -----------------Predictor---------------------
        occ_outs = fullres_occ.permute(0, 2, 3, 4, 1) #bcwhz -> bwhzc
        fullres_occ_inter = self.occ_predictor_1(occ_outs)
        occ_outs = self.occ_predictor_2(fullres_occ_inter)
        fullres_occ_inter = fullres_occ_inter.permute(0, 4, 1, 2, 3) #bwhzc -> bcwhz
        if self.save_high_res_feature:
            occupancy_scores = F.softmax(occ_outs, dim=-1)
            occupancy_scores_max,occupancy_scores_argmax = torch.max(occupancy_scores,dim=-1) 
            occupancy_mask = occupancy_scores_max.new_ones(occupancy_scores_max.shape)
            if self.save_high_res_feature_all_0:
                threshold = 1
            occupancy_mask[occupancy_scores_max < threshold] = 0 
            if self.only_background:
                mask = torch.logical_and(occupancy_scores_argmax > 10, occupancy_scores_argmax!=17)      
                occupancy_mask = torch.logical_and(mask, occupancy_mask)      
            # (occupancy_scores_argmax <= 10) or (occupancy_scores_argmax==17)
            out_occ_mask_feature = fullres_occ_inter*occupancy_mask[:,None,...].bool()
            if self.training:
                out_occ_feature = out_occ_mask_feature.clone()
            else:
                out_occ_feature = out_occ_mask_feature
        else:
            if self.training:
                out_occ_feature = compact_occ.clone()
            else:
                out_occ_feature = compact_occ
        if self.training or not self.test_cfg.only_encoder:
            # if instance_queries is not None:
            #     decoder_out = torch.cat([decoder_out, instance_queries.unsqueeze(0).permute(0,2,1,3)], dim=1)
            maskocc_feature, maskocc_outs = self.predictor(fullres_occ, decoder_out, instance_queries)
            outs = {
                'maskocc_feature': maskocc_feature,
                'occ_outs':occ_outs,
                'maskocc_outs':maskocc_outs,
            }
        else:
            outs = {
                'maskocc_feature': None,
                'occ_outs':occ_outs,
                'maskocc_outs':None,
            }

        return outs,out_occ_feature
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, mask_cameras_list, mask_lidars_list):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list, mask_preds_list, 
                    gt_labels_list, gt_masks_list, mask_cameras_list,
                    mask_lidars_list)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, 
                           mask_cameras, mask_lidars):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        assign_result = self.assigner.assign(cls_score, mask_pred,
                                             gt_labels, gt_masks,
                                             mask_camera=mask_cameras,
                                             mask_lidar=mask_lidars)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks, mask_camera=mask_cameras,
                                              mask_lidar=mask_lidars)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        num_classes = gt_labels[-1]
        labels = gt_labels.new_full((num_queries, ), num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_labels.new_ones(num_queries)
        # class_weights_tensor = torch.tensor(self.loss_cls.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((num_queries, ))
        mask_weights[pos_inds] = 1.0

        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    def loss_occ_single(self, voxel_semantics, preds, visible_mask=None):

        voxel_semantics=voxel_semantics.long()
        voxel_semantics=voxel_semantics.reshape(-1)
        if visible_mask is not None:
            visible_mask = visible_mask.to(torch.int32)
            preds = preds.reshape(-1, self.num_classes+1)
            visible_mask = visible_mask.reshape(-1)
            num_total_samples = visible_mask.sum()
            loss_occ = self.loss_occ(preds, voxel_semantics, visible_mask, 
                                     avg_factor=num_total_samples)
        else:
            preds = preds.reshape(-1, self.num_classes+1)
            loss_occ = self.loss_occ(preds, voxel_semantics)

        return loss_occ
    
    def loss_single(self, cls_scores, mask_preds, gt_labels_list, 
                    gt_masks_list, mask_cameras_list, mask_lidars_list):
        bs = cls_scores.shape[0]
        loss_cls = None
        loss_dice = None
        loss_mask = None
        for b in range(bs): # cause each batch mask will be different shapes
            cls_score = cls_scores[b].unsqueeze(0)
            mask_pred = mask_preds[b].unsqueeze(0)
            gt_label_list = gt_labels_list[b].unsqueeze(0)
            gt_mask_list = gt_masks_list[b].unsqueeze(0)
            mask_camera_list = mask_cameras_list[b].unsqueeze(0)
            mask_lidar_list = mask_lidars_list[b].unsqueeze(0)

            num_imgs = cls_score.size(0)
            cls_scores_list = [cls_score[i] for i in range(num_imgs)]
            mask_preds_list = [mask_pred[i] for i in range(num_imgs)]

            (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
            num_total_pos,
            num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list, gt_label_list, 
                        gt_mask_list, mask_camera_list, mask_lidar_list)

            # shape (batch_size, num_queries)
            labels = torch.stack(labels_list, dim=0)
            # shape (batch_size, num_queries)
            label_weights = torch.stack(label_weights_list, dim=0)
            # shape (num_total_gts, h, w)
            mask_targets = torch.cat(mask_targets_list, dim=0)
            # shape (batch_size, num_queries)
            mask_weights = torch.stack(mask_weights_list, dim=0)

            # classfication loss
            # shape (batch_size * num_queries, )
            cls_score = cls_score.flatten(0, 1)
            labels = labels.flatten(0, 1)
            label_weights = label_weights.flatten(0, 1)
            class_weight = cls_score.new_tensor(self.loss_cls.class_weight)
            
            if loss_cls is None:
                loss_cls = self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=class_weight[labels].sum(),
                )
            else:
                loss_cls += self.loss_cls(
                    cls_score,
                    labels,
                    label_weights,
                    avg_factor=class_weight[labels].sum(),
                )

            num_total_masks = reduce_mean(cls_score.new_tensor([num_total_pos]))
            num_total_masks = max(num_total_masks, 1)

            # extract positive ones
            # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
            mask_pred = mask_pred[mask_weights > 0]
            # mask_weights = mask_weights[mask_weights > 0]
            
            if mask_targets.shape[0] == 0:
                # zero match
                print(f'zero match')
                loss_dice = mask_pred.sum()
                loss_mask = mask_pred.sum()
                return loss_cls, loss_mask, loss_dice
            
            if self.use_camera_mask:
                mask_pred = mask_pred[:, mask_camera_list[0]]
                mask_targets = mask_targets[:, mask_camera_list[0]]
            elif self.use_lidar_mask:
                mask_pred = mask_pred[:, mask_lidar_list[0]]
                mask_targets = mask_targets[:, mask_lidar_list[0]]
            
            mask_pred = mask_pred.flatten(1)
            mask_targets = mask_targets.flatten(1)

            # the weighted version
            # num_total_mask_weights = reduce_mean(mask_weights.sum())
            if loss_dice is None:
                loss_dice = self.loss_dice(mask_pred, mask_targets, 
                                        avg_factor=num_total_masks)
            else:
                loss_dice += self.loss_dice(mask_pred, mask_targets, 
                                        avg_factor=num_total_masks)
            
            # mask loss
            # FocalLoss support input of shape (n, num_class)
            hwz= mask_pred.shape[1]
            # shape (num_total_gts, hwz) -> (num_total_gts * h * w * z, 1)
            mask_pred = mask_pred.reshape(-1, 1)
            # shape (num_total_gts, hwz) -> (num_total_gts * h * w * z)
            mask_targets = mask_targets.reshape(-1)
            # target is (1 - mask_targets) !!! 
            # reason follow https://github.com/open-mmlab/mmdetection/issues/8580
            
            
            if loss_mask is None:
                loss_mask = self.loss_mask(mask_pred, 1 - mask_targets, avg_factor=num_total_masks * hwz)
            else:
                loss_mask += self.loss_mask(
                    mask_pred, 1 - mask_targets, avg_factor=num_total_masks * hwz)

        return loss_cls/bs, loss_mask/bs, loss_dice/bs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             voxel_semantics,
             gt_classes,
             sem_mask,
             mask_camera,
             mask_lidar,
             preds_dicts,):

        num_dec_layers = self.transformer_decoder.num_layers
        all_mask_cameras_list = [mask_camera.bool() for _ in range(num_dec_layers)]
        all_mask_lidars_list = [mask_lidar.bool() for _ in range(num_dec_layers)]

        all_cls_scores = preds_dicts['maskocc_outs']['cls_preds']
        all_mask_preds = preds_dicts['maskocc_outs']['mask_preds']

        loss_dict=dict()

        loss_dict['loss_cls'] = 0
        loss_dict['loss_mask'] = 0
        loss_dict['loss_dice'] = 0
        #loss from other decoder layer
        for num_dec_layer in range(num_dec_layers - 1):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = 0
            loss_dict[f'd{num_dec_layer}.loss_mask'] = 0
            loss_dict[f'd{num_dec_layer}.loss_dice'] = 0

        # assert gt_classes.min()>=0 and gt_classes.max()<=17
        num_query_per_group = self.num_queries // self.group_detr
        for group_index in range(self.group_detr):
            group_query_start = group_index * num_query_per_group
            group_query_end = (group_index+1) * num_query_per_group
            group_cls_scores = all_cls_scores[group_index]
            group_mask_scores = all_mask_preds[:, :, group_query_start:group_query_end, ...]
            group_gt_classes = gt_classes[group_index]
            group_gt_classes_list = [group_gt_classes for _ in range(num_dec_layers)]
            group_gt_masks = sem_mask[group_index]
            group_gt_masks_list = [group_gt_masks for _ in range(num_dec_layers)]
            self.loss_cls.class_weight = [1.0]*self.group_classes[group_index] + [0.1]
            self.loss_mask.class_weight = [1.0,1.0]
            losses_cls, losses_mask, losses_dice = multi_apply(
                self.loss_single, group_cls_scores, group_mask_scores, group_gt_classes_list,
                group_gt_masks_list, all_mask_cameras_list, all_mask_lidars_list)
            # loss from last decoder layer
            norm_num = self.group_detr - 1 if group_index != 0 else 1
            loss_dict['loss_cls'] += losses_cls[-1] / norm_num
            loss_dict['loss_mask'] += losses_mask[-1] / norm_num
            loss_dict['loss_dice'] += losses_dice[-1] / norm_num
            # loss from other decoder layer
            num_dec_layer = 0
            for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                    losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
                loss_dict[f'd{num_dec_layer}.loss_cls'] += loss_cls_i / norm_num
                loss_dict[f'd{num_dec_layer}.loss_mask'] += loss_mask_i / norm_num
                loss_dict[f'd{num_dec_layer}.loss_dice'] += loss_dice_i / norm_num
                num_dec_layer += 1
                
        if 'instance_mask_preds' in preds_dicts['maskocc_outs']:
            instance_preds = preds_dicts['maskocc_outs']['instance_mask_preds']
            if instance_preds is not None:
                gt_mask_list = sem_mask[-1]
                loss_instance_dice = 0
                for i in range(len(instance_preds)):
                    mask_pred = instance_preds[i].squeeze(0) # [N,W,H,D]
                    mask_camera = all_mask_cameras_list[0][i] # [W,H,D]
                    gt_mask = gt_mask_list[i] # [N,W,H,D]
                    mask_pred_agent = mask_pred[:,mask_camera]
                    gt_mask = gt_mask[:,mask_camera]
                    loss_instance_dice += self.loss_dice(mask_pred_agent, gt_mask, avg_factor=gt_mask.sum())
                loss_dict['loss_instance_dice'] = (loss_instance_dice / len(instance_preds)) * 10
        return loss_dict
    
    def decoder_inference(self, mask_cls_results, mask_pred_results, reference_points=None, sampling_locations=None):
        # mask_cls_results [bs, N, num_class]
        # mask_pred_results[bs, N, W, H, Z]        
        processed_results = []
        for index, mask_result in enumerate(zip(mask_cls_results, mask_pred_results)):
            mask_cls_result, mask_pred_result = mask_result
            scores, labels = F.softmax(mask_cls_result, dim=-1).max(-1) # [N]
            mask_pred = mask_pred_result.sigmoid() # [N, W, H, Z]
            keep = labels.ne(self.num_classes) & (scores > self.test_cfg.mask_threshold)
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[keep]
            cur_mask_cls = mask_cls_result[keep]
            cur_mask_cls = cur_mask_cls[:, :-1]

            cur_prob_masks = cur_scores.view(-1, 1, 1, 1) * cur_masks #[N, w, h, z] 0~1 x 0~1

            w, h, z = cur_masks.shape[-3:]
            semseg = torch.ones((w, h, z), dtype=torch.int32, device=cur_masks.device) * (self.num_classes)

            if cur_masks.shape[0] == 0:
                # we didn't detect any mask :(
                processed_results.append(semseg)
            else:
                cur_mask_ids = cur_prob_masks.argmax(0)
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    mask = cur_mask_ids == k
                    mask_area = mask.sum().item()   # argmax(cls_score * mask_score)
                    cur_mask = cur_masks[k] >= self.test_cfg.occupy_threshold  # mask_only
                    original_area = cur_mask.sum().item()  # mask_score > threshold

                    if mask_area > 0 and original_area > 0:
                        if mask_area / original_area < self.test_cfg.overlap_threshold:
                            continue
                        if (mask_area > original_area):
                            if (pred_class < 11):
                                semseg[cur_mask] = pred_class
                            # elif (mask_area > self.test_cfg.max_mask_len):
                            elif (mask_area > 3 * original_area):
                                semseg[cur_mask] = pred_class
                            else:
                                semseg[mask] = pred_class
                        else:
                            semseg[cur_mask] = pred_class
                            semseg[mask] = pred_class
                processed_results.append(semseg)
                # print(f"semseg.shape:{semseg.shape}")

        processed_results = torch.stack(processed_results)
        return processed_results
    
    def merge(self, pred1, pred2,instance_pred=None):
        # pred1 obtained from mask head
        # pred2 obtained from fullres occ
        # instance_pred obtained from instance queries
        for cls in range(self.num_classes):
            # mask1 = pred1 == cls
            # if mask1.sum().item() > self.test_cfg.max_mask_len:
            #     pred1[mask1] = self.num_classes
            if cls >= 11:
                mask1 = pred1 == cls
                pred1[mask1] = self.num_classes  # ignore background
                # if instance_pred is not None:
                #     mask1 = instance_pred == cls-
            if instance_pred is not None:
                if cls < 11 and cls !=0:
                    mask1 = instance_pred == (cls-1)
                    pred1[mask1] = cls
            mask2 = pred2 == cls
            pred1[mask2] = cls
            
            
                    
        return pred1
    


    @force_fp32(apply_to=('preds_dicts'))
    def get_occ(self, preds_dicts, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        occ_dict = {}
        occ_outs = preds_dicts['occ_outs']
        # occ_outs = occ_outs.softmax(-1)
        occ_outs = occ_outs.argmax(-1)
        if self.test_cfg.only_encoder:
            occ_dict['occ'] = occ_outs.squeeze(dim=0)
            return occ_dict
        mask_cls_results = preds_dicts['maskocc_outs']['cls_preds'][0][-1] #[bs, N, num_class]
        mask_pred_results = preds_dicts['maskocc_outs']['mask_preds'][-1] #[bs, N, W, H, Z]

        # if len(preds_dicts['maskocc_outs']['cls_preds']) > 1:
        #     instance_mask_cls_results = preds_dicts['maskocc_outs']['cls_preds'][1][-1] #[bs, N, num_class]
        #     instance_mask_pred_results = mask_pred_results[:,mask_cls_results.shape[1]:] 
        #     mask_pred_results = mask_pred_results[:,:mask_cls_results.shape[1]]
        #     instance_occ_results = self.decoder_inference(instance_mask_cls_results, instance_mask_pred_results)
        # else:
        instance_occ_results = None

        occ_results = self.decoder_inference(mask_cls_results, mask_pred_results)

        if self.test_cfg.inf_merge:
            occ_results = self.merge(occ_results, occ_outs, instance_occ_results)

        occ_dict['occ'] = occ_results.squeeze(dim=0).cpu().numpy().astype(np.uint8)

        return occ_dict