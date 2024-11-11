import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from projects.mmdet3d_plugin.core.box3d import *
from ..base_target import BaseTargetWithDenoising
from ..modules import Stage2Assigner
import pdb




# from scipy.spatial import ConvexHull
# from numpy import *

# # def polygon_clip(subjectPolygon, clipPolygon):
# #    """ Clip a polygon with another polygon.

# #    Ref: https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping#Python

# #    Args:
# #      subjectPolygon: a list of (x,y) 2d points, any polygon.
# #      clipPolygon: a list of (x,y) 2d points, has to be *convex*
# #    Note:
# #      **points have to be counter-clockwise ordered**

# #    Return:
# #      a list of (x,y) vertex point for the intersection polygon.
# #    """
# #    def inside(p):
# #       return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
# #    def computeIntersection():
# #       dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
# #       dp = [ s[0] - e[0], s[1] - e[1] ]
# #       n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
# #       n2 = s[0] * e[1] - s[1] * e[0] 
# #       n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
# #       return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
# #    outputList = subjectPolygon
# #    cp1 = clipPolygon[-1]
 
# #    for clipVertex in clipPolygon:
# #       cp2 = clipVertex
# #       inputList = outputList
# #       outputList = []
# #       s = inputList[-1]
 
# #       for subjectVertex in inputList:
# #          e = subjectVertex
# #          if inside(e):
# #             if not inside(s):
# #                outputList.append(computeIntersection())
# #             outputList.append(e)
# #          elif inside(s):
# #             outputList.append(computeIntersection())
# #          s = e
# #       cp1 = cp2
# #       if len(outputList) == 0:
# #           return None
# #    return(outputList)

# # def poly_area(x,y):
# #     """ Ref: http://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates """
# #     return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

# # def convex_hull_intersection(p1, p2):
# #     """ Compute area of two convex hull's intersection area.
# #         p1,p2 are a list of (x,y) tuples of hull vertices.
# #         return a list of (x,y) for the intersection and its volume
# #     """
# #     inter_p = polygon_clip(p1,p2)
# #     if inter_p is not None:
# #         hull_inter = ConvexHull(inter_p)
# #         return inter_p, hull_inter.volume
# #     else:
# #         return None, 0.0  

# # def box3d_vol(corners):
# #     ''' corners: (8,3) no assumption on axis direction '''
# #     a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
# #     b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
# #     c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
# #     return a*b*c

# # def is_clockwise(p):
# #     x = p[:,0]
# #     y = p[:,1]
# #     return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

# # def box3d_iou(corners1, corners2):
# #     ''' Compute 3D bounding box IoU.

# #     Input:
# #         corners1: numpy array (8,3), assume up direction is negative Y
# #         corners2: numpy array (8,3), assume up direction is negative Y
# #     Output:
# #         iou: 3D bounding box IoU
# #         iou_2d: bird's eye view 2D bounding box IoU

# #     todo (kent): add more description on corner points' orders.
# #     '''
# #     # corner points are in counter clockwise order
# #     rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
# #     rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 
    
# #     area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
# #     area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
   
# #     inter, inter_area = convex_hull_intersection(rect1, rect2)
# #     iou_2d = inter_area/(area1+area2-inter_area)
# #     ymax = min(corners1[0,1], corners2[0,1])
# #     ymin = max(corners1[4,1], corners2[4,1])

# #     inter_vol = inter_area * max(0.0, ymax-ymin)
    
# #     vol1 = box3d_vol(corners1)
# #     vol2 = box3d_vol(corners2)
# #     iou = inter_vol / (vol1 + vol2 - inter_vol)
# #     return iou, iou_2d




__all__ = ["SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        allow_low_quality_matches: bool = False,
        bbox_3d: bool = True,
    ):
        super(SparseBox3DTarget, self).__init__(
            num_dn_groups, num_temp_dn_groups
        )
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.bbox_3d = bbox_3d
        self.add_neg_dn = add_neg_dn
        self.matcher_o2m = Stage2Assigner(allow_low_quality_matches=allow_low_quality_matches)


    def encode_reg_target(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    box[..., [W, L, H]].log(),
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
    
    def encode_reg_target_future(self, box_target, device=None):
        outputs = []
        for box in box_target:
            output = torch.cat(
                [
                    box[..., [X, Y, Z]],
                    torch.sin(box[..., 3]).unsqueeze(-1),
                    torch.cos(box[..., 3]).unsqueeze(-1),
                ],
                dim=-1,
            )
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)
        return outputs

    def sample(
        self,
        cls_pred,
        box_pred,
        cls_target,
        box_target,
        instance_id = None,
        gt_id = None,
        next_pos = None,
        layer_idx = -1,
    ):
        bs, num_pred, num_cls = cls_pred.shape
        cls_cost = self._cls_cost(cls_pred, cls_target)
        box_target = self.encode_reg_target(box_target, box_pred.device)
        if next_pos is not None:
            box_target_future = self.encode_reg_target_future(next_pos, box_pred.device)
        instance_reg_weights = []
        
        for i in range(len(box_target)):
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],
                        weights.new_tensor(weight),
                        weights,
                    )
            instance_reg_weights.append(weights)
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)
        flag = False
        indices = []
        
        
        if instance_id is not None:
            instance_id_rest = torch.zeros_like(instance_id) - 1
        
        if self.bbox_3d:
            _,cost_mat = self.matcher_o2m(cls_pred,box_pred, cls_target,box_target,return_cost_matrix=True)
        
        for i in range(bs):
            # if self.bbox_3d == False:
            if cls_cost[i] is not None and box_cost[i] is not None:
                # 비용 행렬 계산
                cost = (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)
        
                if instance_id is not None:
                    index = torch.where(instance_id[i] != -1)[0]
                    for idx in index:                            
                        ind = instance_id[i][idx].item()
                        cost_inf = np.full((cost.shape[1]), 1e8)
                        if layer_idx == 1:
                            # get iou
                            if ind in gt_id[i]:
                                if self.bbox_3d:
                                    iou_cost  = cost_mat[i][torch.where(gt_id[i] == ind)[0][0].item()][idx]
                                else:
                                    tracked_bbox = box_pred[i][idx]
                                    tracked_bbox_gt = box_target[i][torch.where(gt_id[i] == ind)[0][0].item()]
                                    iou_cost = self._iou(tracked_bbox, tracked_bbox_gt)
                                if iou_cost < 0.3:
                                    continue
                                else:
                                    instance_id_rest[i][idx] = ind
                            else:
                                instance_id_rest[i][idx] = ind
                        if ind in gt_id[i]:
                            target_index = torch.where(gt_id[i] == ind)[0][0]
                            cost_inf[target_index] = 0
                        cost[idx] = cost_inf
                            
                    
                assign = linear_sum_assignment(cost)

                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                indices.append([None, None])
                
        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )
        output_box_target = box_pred.new_zeros(box_pred.shape)
        output_box_target_next = box_pred.new_zeros(box_pred.shape)[:,:,:5]
        output_reg_weights = box_pred.new_zeros(box_pred.shape)
        output_reg_weights_next = box_pred.new_zeros(box_pred.shape)

        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue

            output_cls_target[i, pred_idx] = cls_target[i][target_idx]
            output_box_target[i, pred_idx] = box_target[i][target_idx]
            if next_pos is not None:
                output_box_target_next[i, pred_idx] = box_target_future[i][target_idx]
                output_reg_weights_next[i, pred_idx] = instance_reg_weights[i][target_idx]
            if instance_id is not None:
                if layer_idx != 1:
                    instance_id_rest[i, pred_idx] = gt_id[i][target_idx].to(device=instance_id.device)
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][target_idx]
            
        output_reg_weights_next = torch.cat([output_reg_weights_next[...,:3],output_reg_weights_next[...,3:5]],dim=-1)
        if instance_id is not None:
            instance_id = instance_id_rest
        return output_cls_target, output_box_target, output_reg_weights,instance_id, output_box_target_next,output_reg_weights_next,indices
   
    def _iou(self, box1, box2):
        x1,y1,z1 = box1[:3]
        x2,y2,z2 = box2[:3]
        w1,l1,h1 = box1[3:6].exp()
        w2,l2,h2 = box2[3:6].exp()

        
        pos_1_1 = [x1-w1/2,y1-l1/2]
        pos_1_2 = [x1+w1/2,y1+l1/2]
        pos_2_1 = [x2-w2/2,y2-l2/2]
        pos_2_2 = [x2+w2/2,y2+l2/2]
        
        box1_area = w1*l1
        box2_area = w2*l2
        
        x1 = max(pos_1_1[0], pos_2_1[0])
        y1 = max(pos_1_1[1], pos_2_1[1])
        x2 = min(pos_1_2[0], pos_2_2[0])
        y2 = min(pos_1_2[1], pos_2_2[1])
        
        # get the overlap area

        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        inter = w * h
        return inter / (box1_area + box2_area - inter)

        
    def _cls_cost(self, cls_pred, cls_target):
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()
        cost = []
        for i in range(bs):
            if len(cls_target[i]) > 0:
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )
                cost.append(
                    (pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]])
                    * self.cls_weight
                )
            else:
                cost.append(None)
        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        bs = box_pred.shape[0]
        cost = []
        for i in range(bs):
            if len(box_target[i]) > 0:
                cost.append(
                    torch.sum(
                        torch.abs(box_pred[i, :, None] - box_target[i][None])
                        * instance_reg_weights[i][None]
                        * box_pred.new_tensor(self.reg_weights),
                        dim=-1,
                    )
                    * self.box_weight
                )
            else:
                cost.append(None)
        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        if self.num_dn_groups <= 0:
            return None
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None
        cls_target = torch.stack(
            [
                F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                for x in cls_target
            ]
        )
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )

        bs, num_gt, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(self.num_dn_groups, 1)
            box_target = box_target.tile(self.num_dn_groups, 1, 1)
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(self.num_dn_groups, 1)

        noise = torch.rand_like(box_target) * 2 - 1
        noise *= box_target.new_tensor(self.dn_noise_scale)
        dn_anchor = box_target + noise
        if self.add_neg_dn:
            noise_neg = torch.rand_like(box_target) + 1
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )
            noise_neg *= flag
            noise_neg *= box_target.new_tensor(self.dn_noise_scale)
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_gt *= 2

        box_cost = self._box_cost(
            dn_anchor, box_target, torch.ones_like(box_target)
        )
        dn_box_target = torch.zeros_like(dn_anchor)
        dn_cls_target = -torch.ones_like(cls_target) * 3
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()
            anchor_idx, gt_idx = linear_sum_assignment(cost)
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]
            dn_cls_target[i, anchor_idx] = cls_target[i, gt_idx]
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[i, gt_idx]
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_gt, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_gt)
            .permute(1, 0, 2)
            .flatten(1)
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None
        valid_mask = dn_cls_target >= 0
        if self.add_neg_dn:
            cls_target = (
                torch.cat([cls_target, cls_target], dim=1)
                .reshape(self.num_dn_groups, bs, num_gt)
                .permute(1, 0, 2)
                .flatten(1)
            )
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )  # valid denotes the items is not from pad.
        attn_mask = dn_box_target.new_ones(
            num_gt * self.num_dn_groups, num_gt * self.num_dn_groups
        )
        for i in range(self.num_dn_groups):
            start = num_gt * i
            end = start + num_gt
            attn_mask[start:end, start:end] = 0
        attn_mask = attn_mask == 1
        dn_cls_target = dn_cls_target.long()
        return (
            dn_anchor,
            dn_box_target,
            dn_cls_target,
            attn_mask,
            valid_mask,
            dn_id_target,
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_noraml_anchor,
        temporal_valid_mask,
    ):
        bs, num_anchor = instance_feature.shape[:2]
        if temporal_valid_mask is None:
            self.dn_metas = None
        if self.dn_metas is None or num_noraml_anchor >= num_anchor:
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # split instance_feature and anchor into non-dn and dn
        num_dn = num_anchor - num_noraml_anchor
        dn_instance_feature = instance_feature[:, -num_dn:]
        dn_anchor = anchor[:, -num_dn:]
        instance_feature = instance_feature[:, :num_noraml_anchor]
        anchor = anchor[:, :num_noraml_anchor]

        # reshape all dn metas from (bs,num_all_dn,xxx)
        # to (bs, dn_group, num_dn_per_group, xxx)
        num_dn_groups = self.num_dn_groups
        num_dn = num_dn // num_dn_groups
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)
        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # update temp_dn_metas by instance_id
        temp_dn_feat = self.dn_metas["dn_instance_feature"]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]

        # bs, num_temp_dn_groups, num_temp_dn, num_dn
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(dim=3)
        temp_cls_target = torch.where(
            torch.all(torch.logical_not(match), dim=-1),
            self.dn_metas["dn_cls_target"].new_tensor(-1),
            self.dn_metas["dn_cls_target"],
        )
        temp_valid_mask = self.dn_metas["valid_mask"]
        temp_dn_anchor = self.dn_metas["dn_anchor"]

        # handle the misalignment the length of temp_dn to dn caused by the
        # change of num_gt, then concat the temp_dn and dn
        temp_dn_metas = [
            temp_dn_feat,
            temp_dn_anchor,
            temp_reg_target,
            temp_cls_target,
            temp_valid_mask,
            temp_dn_id,
        ]
        dn_metas = [
            dn_feat,
            dn_anchor,
            dn_reg_target,
            dn_cls_target,
            valid_mask,
            dn_id,
        ]
        output = []
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):
            if num_temp_dn < num_dn:
                pad = (0, num_dn - num_temp_dn)
                if temp_meta.dim() == 4:
                    pad = (0, 0) + pad
                else:
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                temp_meta = temp_meta[:, :, :num_dn]
            mask = temporal_valid_mask[:, None, None]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)
            temp_meta = torch.where(
                mask, temp_meta, meta[:, :num_temp_dn_groups]
            )
            meta = torch.cat([temp_meta, meta[:, num_temp_dn_groups:]], dim=1)
            meta = meta.flatten(1, 2)
            output.append(meta)
        output[0] = torch.cat([instance_feature, output[0]], dim=1)
        output[1] = torch.cat([anchor, output[1]], dim=1)
        return output

    def cache_dn(
        self,
        dn_instance_feature,
        dn_anchor,
        dn_cls_target,
        valid_mask,
        dn_id_target,
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = (
            torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        )
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(
                bs, num_dn_groups, num_temp_dn
            )[:, temp_group_mask]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
