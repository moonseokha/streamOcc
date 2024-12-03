import torch
import torch.nn.functional as F
import torch.nn as nn
from .embedding import MultipleInputEmbedding,SingleInputEmbedding
from .utils import init_weights
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj
from torch_geometric.typing import OptTensor
from torch_geometric.typing import Size
from torch_geometric.utils import softmax
from typing import Optional
import pdb
from mmcv.cnn import xavier_init, constant_init, build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
import cv2
from mmdet.models import build_backbone
from ..blocks import DAF
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (
    FEEDFORWARD_NETWORK,
)
from mmdet.core import multi_apply
from mmcv.cnn.bricks.conv_module import ConvModule

import time

__all__ = ["Interaction_Net"]


X, Y, Z, W, L, H, SIN, COS = 0, 1, 2, 3, 4, 5, 6, 7 
def box3d_to_corners(box3d):
    # Define constants
    
    boxes = box3d.clone().detach()
    boxes[..., 3:6] = boxes[..., 3:6].exp()

    corners_norm = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2)), dim=-1).view(-1, 3)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]] - 0.5
    corners = boxes[..., None, [W, L, H]] * corners_norm.to(boxes.device).reshape(1, 8, 3)

    rot_mat = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(boxes.shape[0], boxes.shape[1], 1, 1).to(boxes.device)
    rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 1, 0], rot_mat[..., 1, 1] = boxes[..., COS], -boxes[..., SIN], boxes[..., SIN], boxes[..., COS]

    corners = (rot_mat.unsqueeze(2) @ corners.unsqueeze(-1)).squeeze(-1)
    return corners + boxes[..., None, :3] 



def calculate_voxel_indices(min_pos,max_pos, indices, batch_num, vox_size, vox_flatten_size, device):
    xs = torch.arange(min_pos[0].item(), max_pos[0].item() + 1, device=device)
    ys = torch.arange(min_pos[1].item(), max_pos[1].item() + 1, device=device)
    zs = torch.arange(min_pos[2].item(), max_pos[2].item() + 1, device=device)
    mesh_indices = torch.meshgrid(xs, ys, zs)
    mesh_indices = torch.stack(mesh_indices, dim=-1).reshape(-1, 3)
    voxel_indices = (mesh_indices[:, 1] * vox_size[1] * vox_size[2] + mesh_indices[:, 0] * vox_size[2] + mesh_indices[:, 2])
    voxel_indices = torch.clamp(voxel_indices, min=0, max=vox_size[0] * vox_size[1] * vox_size[2] - 1) + (batch_num * vox_flatten_size)
    
    return voxel_indices, torch.full_like(voxel_indices, indices, device=device)

def optimized_voxel_extraction(vox_feature, agent_pos, indices, vox_lower_point, vox_res, vox_size):
    """
    Voxel Grid에서 bounding box에 속하는 grid indices와 query indices를 추출합니다.
    Args:
        vox_feature: [B, H, W, D, embed_dim] 형태의 voxel feature tensor.
        agent_pos: 각 에이전트의 3D 위치 정보. # [B, N, 11]
        indices: 각 batch 내의 선택된 agent의 인덱스. # [B] list of tensor 
        vox_lower_point: voxel grid의 시작 좌표. 
        vox_res: voxel grid의 해상도.
        vox_size: voxel grid의 크기.
    Returns:
        vox_indices: bounding box 내부에 포함된 voxel grid의 인덱스들.
        query_indices: 각 voxel에 대응하는 query의 인덱스들.
    """
    B, H, W, D, embed = vox_feature.shape
    vox_flatten_size = H * W * D
    device = vox_feature.device

    # flatten agent_pos, indices for batch-level processing
    agent_num = agent_pos.shape[1]
    agent_pos = agent_pos.reshape(-1, agent_pos.shape[-1])
    batch_num = torch.cat([torch.tensor([b]*indices[b].shape[0]).to(device) for b in range(B)])
    indices = torch.cat(indices).to(device) + torch.tensor(agent_num, device=device) * batch_num
    query_indices = [num for num in range(len(indices))]
    B = indices.shape[0]

    # get agent_corner pos
    if indices.shape[0] == 0:
        vox_indices = torch.tensor([], device=device, dtype=torch.long)
        query_indices = torch.tensor([], device=device, dtype=torch.long)
        return vox_indices, query_indices

    else:
        corners = box3d_to_corners(agent_pos[indices].unsqueeze(0)).squeeze(0)
        vox_pos = ((corners - torch.tensor(vox_lower_point, dtype=agent_pos.dtype, device=device)) / torch.tensor(vox_res, dtype=agent_pos.dtype, device=device)).round()
        min_pos, max_pos = vox_pos.min(dim=1)[0], vox_pos.max(dim=1)[0]
        min_pos[:,0] = torch.clamp(min_pos[:,0], 0, vox_size[0]-1)
        min_pos[:,1] = torch.clamp(min_pos[:,1], 0, vox_size[1]-1)
        min_pos[:,2] = torch.clamp(min_pos[:,2], 0, vox_size[2]-1)
        max_pos[:,0] = torch.clamp(max_pos[:,0], 0, vox_size[0]-1)
        max_pos[:,1] = torch.clamp(max_pos[:,1], 0, vox_size[1]-1)
        max_pos[:,2] = torch.clamp(max_pos[:,2], 0, vox_size[2]-1)
        results = multi_apply(calculate_voxel_indices, 
                            min_pos, 
                            max_pos,
                            query_indices,
                            batch_num, 
                            [vox_size]*B,
                            [vox_flatten_size] * B, 
                            [device] * B)
        vox_indices, query_indices = results
    return torch.cat(vox_indices), torch.cat(query_indices)

    
@PLUGIN_LAYERS.register_module()
class Interaction_Net(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 conv_cfg: dict = None,
                 grid_config: dict = None,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 conv3d_layers: int = 1,
                 num_classes: int = 18,
                 down_ratio: int = 4,
                 use_edge_pos: bool = False,
                 img_to_vox: bool = False,
                 num_cams: int = 6,
                 num_levels: int = 4,
                 num_points: int = 6,
                 num_groups: int = 8,
                 query_to_vox: bool = True,
                 without_occ: bool = False,
                 vox_att_cfg: dict = None,
                 ffn: dict = None,
                 ) -> None:

        super(Interaction_Net, self).__init__()
        self.use_edge_pos=use_edge_pos
        self.without_occ = without_occ
        self.use_vox_atten = True if vox_att_cfg is not None else False
        deconv_cfg = dict(type='deconv3d')
        conv3d_cfg=dict(type='Conv3d',bias=False)
        gn_norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        out_dims = embed_dims//down_ratio
        self.embed_dims = embed_dims
        self.conv3d_layers = conv3d_layers
        self.num_classes = num_classes
        self.img_to_vox = img_to_vox
        self.query_to_vox = query_to_vox
        if img_to_vox:
            self.attention_weights =  nn.Linear(embed_dims, num_groups * num_cams * num_levels * num_points)
            self.sampling_offsets = nn.Linear(embed_dims,  num_points * 3)
            self.out_proj = nn.Linear(embed_dims,embed_dims)
            self.proj_drop = nn.Dropout(0.1)
            self.num_cams = num_cams
            self.num_levels = num_levels
            self.num_points = num_points
            self.num_groups = num_groups

        self.down_ratio = down_ratio
        self.vox_size = [int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][-1]),
                         int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][-1]),
                         int((grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][-1])]
        self.vox_res = [grid_config['y'][-1],grid_config['x'][-1],grid_config['z'][-1]]
        self.vox_lower_point = [grid_config['y'][0],
                                 grid_config['x'][0],
                                 grid_config['z'][0]]
        self.norm_cross = nn.LayerNorm(embed_dims)

        if query_to_vox:
            self.cross_attn_Q2V = CrossAttentionLayer(embed_dim=embed_dims,
                                                        num_heads=num_heads,
                                                        dropout=dropout,
                                                        use_edge_pos=use_edge_pos)
        if vox_att_cfg is None and conv_cfg is not None:
            self.conv_net = nn.ModuleList(
                [
                    build_backbone(conv_cfg)
                    for _ in range(conv3d_layers)
                ]
            )
        elif vox_att_cfg is not None:
            self.vox_att_cfg = build_attention(vox_att_cfg)
            self.norm = nn.LayerNorm(embed_dims)
            if ffn is not None:
                self.ffn = build_from_cfg(ffn,FEEDFORWARD_NETWORK)
            self.norm_ffn = nn.LayerNorm(embed_dims)
        else:
            self.vox_att_cfg = None
            self.use_vox_atten = True
        if not without_occ:
            
            # upsample = build_upsample_layer(deconv_cfg, embed_dims, out_dims, kernel_size=2, stride=2)
            # self.up_block = nn.Sequential(upsample,
            #                 nn.BatchNorm3d(out_dims),
            #                 nn.ReLU(inplace=True)
            #                 )
            self.up_block = nn.Sequential(
                    # nn.ConvTranspose3d(embed_dims,embed_dims//2,(3,3,3),padding=(1,1,1)),
                    # nn.BatchNorm3d(embed_dims//2),
                    # nn.ReLU(inplace=True),
                    nn.ConvTranspose3d(embed_dims, out_dims, (2, 2, 2), stride=(2, 2, 2)),
                    nn.BatchNorm3d(out_dims),
                    nn.ReLU(inplace=True),
                )
            self.proj_layer = ConvModule(
                                    out_dims,
                                    embed_dims,
                                    kernel_size=1,
                                    stride=1,
                                    # padding=1,
                                    bias=True,
                                    conv_cfg=dict(type='Conv3d'))
            self.vox_occ_net = nn.Conv3d(embed_dims, num_classes, 1,1,0)
            # self.vox_occ_net = build_conv_layer(
            #         conv3d_cfg,
            #         in_channels=out_dims,
            #         out_channels=self.num_classes,
            #         kernel_size=1,
            #         stride=1,
            #         padding=0)
            
        self.apply(init_weights)

    def forward(self,
                agent_feat: torch.Tensor = None, # [B, N, embed_dim]
                vox_feat:torch.Tensor = None,  #[B,H,W,Z,embed_dim]
                agent_pos: torch.Tensor = None, # [B,N,11]
                agent_pos_embed: torch.Tensor = None, # [B,N,embed_dim]
                indices = None,
                metas = None,
                img_feats = None,
                ref_3d = None,
                voxel_pos = None,
                ) -> torch.Tensor:
        

        
        vox_indices = []
        query_indices = []
        vox_occ = None
        vox_flatten_size = vox_feat.shape[1] * vox_feat.shape[2] * vox_feat.shape[3]
        B, H, W, D, embed = vox_feat.shape
        up_vox_occ = None
        query_total = 0
        i2v_pos = []

        # vox_indices, query_indices = optimized_voxel_extraction(vox_feat, agent_pos, indices, self.vox_lower_point, self.vox_res, self.vox_size)
        if self.query_to_vox:
            for b in range(B):
                bottom_pos = box3d_to_corners(agent_pos[b][indices[b]].unsqueeze(0)).squeeze(0)  # [B, N, 8, 3]
                vox_pos = ((bottom_pos - torch.tensor(self.vox_lower_point, dtype=agent_pos.dtype, device=agent_pos.device))
                        / torch.tensor(self.vox_res, dtype=agent_pos.dtype, device=agent_pos.device)).round()
                min_pos= vox_pos.min(dim=1)[0]
                max_pos = vox_pos.max(dim=1)[0]
                min_pos[:,0] = torch.clamp(min_pos[:,0], 0, self.vox_size[0]-1)
                min_pos[:,1] = torch.clamp(min_pos[:,1], 0, self.vox_size[1]-1)
                min_pos[:,2] = torch.clamp(min_pos[:,2], 0, self.vox_size[2]-1)
                max_pos[:,0] = torch.clamp(max_pos[:,0], 0, self.vox_size[0]-1)
                max_pos[:,1] = torch.clamp(max_pos[:,1], 0, self.vox_size[1]-1)
                max_pos[:,2] = torch.clamp(max_pos[:,2], 0, self.vox_size[2]-1)

                
                for num in range(vox_pos.shape[0]):
                    xs = torch.arange(min_pos[num, 0].item(), max_pos[num, 0].item() + 1, device=vox_feat.device)
                    ys = torch.arange(min_pos[num, 1].item(), max_pos[num, 1].item() + 1, device=vox_feat.device)
                    zs = torch.arange(min_pos[num, 2].item(), max_pos[num, 2].item() + 1, device=vox_feat.device)

                    mesh_index = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1).reshape(-1, 3)
                    if self.img_to_vox:
                        i2v_pos.extend(torch.tensor(mesh_index))
                    indexes = (mesh_index[..., 1] * self.vox_size[1] * self.vox_size[2] +
                            mesh_index[..., 0] * self.vox_size[2] +
                            mesh_index[..., 2])
                    indexes = torch.clip(indexes, min=0, max=(self.vox_size[0] * self.vox_size[1] * self.vox_size[2] - 1))
                    indexes += b * vox_flatten_size
                    unique_indexes = torch.unique(indexes)
                    vox_indices.extend(unique_indexes.tolist())
                    query_indices.extend([query_total] * len(unique_indexes))
                    query_total+=1
                if self.img_to_vox:
                    i2v_pos = torch.unique(torch.stack(i2v_pos),dim=0).to(torch.long)
                    i2v_indices = torch.tensor(i2v_pos[..., 1]*self.vox_size[1] * self.vox_size[2] +i2v_pos[...,0]*self.vox_size[2]+ i2v_pos[...,2]).to(torch.long)
                    occupied_vox = torch.gather(vox_feat[b].clone(), 0, i2v_indices[:,None].repeat(1,vox_feat.shape[-1])).unsqueeze(0) # [1, N, embed_dim]
                    bs = occupied_vox.shape[0]
                    sampling_offsets = self.sampling_offsets(occupied_vox)
                    sampling_offsets = sampling_offsets.view(bs, -1, self.num_points, 3)
                    attention_weights = self.attention_weights(occupied_vox) 
                    attention_weights = attention_weights.view(bs, -1,  self.num_points*self.num_cams*self.num_levels * self.num_groups)
                    attention_weights = attention_weights.softmax(-2).view(bs, -1, self.num_points, self.num_cams, self.num_levels, self.num_groups)
                    # offset_normalizer = torch.tensor([vox_feat.shape[1], vox_feat.shape[2],vox_feat.shape[3]]).to(vox_feat.device)
                    sampling_locations = i2v_pos[None, :, None, :] + sampling_offsets
                    sampling_locations += - torch.tensor(self.vox_lower_point, dtype=agent_pos.dtype, device=agent_pos.device)
                    sampling_locations = sampling_locations * torch.tensor(self.vox_res, dtype=agent_pos.dtype, device=agent_pos.device)
                    points_2d = (
                            self.project_points(
                                sampling_locations, #[bs, num_anchor, num_pts, 3]
                                metas["projection_mat"][b:b + 1],
                                metas.get("image_wh")[b:b + 1],
                            )
                            .permute(0, 2, 3, 1, 4)
                            .reshape(bs, -1, self.num_points, self.num_cams, 2)
                    )
                    img_feat = [img_feats[0][b].unsqueeze(0), img_feats[1].unsqueeze(0), img_feats[2]]
                    features = DAF(*img_feat, points_2d, attention_weights).reshape(bs, -1, self.embed_dims)   
                    features = self.proj_drop(self.out_proj(features))
                    occupied_vox = occupied_vox + features
                    vox_feat[b] = torch.scatter(vox_feat[b].clone(),0,i2v_indices[:,None].repeat(1,vox_feat.shape[-1]),occupied_vox.squeeze(0))
                    i2v_pos = []

            # # if self.img_to_vox:
            # #     vox_with_query = vox_feat.reshape(B,H,W,D,-1).permute(0,4,1,2,3)
            
                edge_index_Q2V = torch.stack([torch.tensor(query_indices, device=vox_feat.device, dtype=torch.long),
                                            torch.tensor(vox_indices, device=vox_feat.device, dtype=torch.long)], dim=0)  
                
                # #############   Query to Voxel    #####################
                selected_agent_feat = torch.cat([agent_feat[b][indices[b]] for b in range(agent_feat.shape[0])],dim=0)
                if self.use_edge_pos:
                    selected_agent_pos = torch.cat([agent_pos_embed[b][indices[b]] for b in range(agent_pos_embed.shape[0])],dim=0).reshape(-1,self.embed_dims)
                    vox_with_query = self.cross_attn_Q2V(x=(selected_agent_feat,vox_feat.reshape(-1,self.embed_dims)),edge_index=edge_index_Q2V,edge_attr=selected_agent_pos,voxel_pos_embedding=voxel_pos)
                else:
                    vox_with_query = self.cross_attn_Q2V(x=(selected_agent_feat,vox_feat.reshape(-1,self.embed_dims)),edge_index=edge_index_Q2V,voxel_pos_embedding=voxel_pos)
                vox_with_query = self.norm_cross(vox_with_query)
                vox_with_query = vox_with_query.reshape(-1,self.vox_size[0],self.vox_size[1],self.vox_size[2],self.embed_dims)

                vox_with_query = vox_with_query.permute(0,4,1,2,3) # [B, C, H, W, D]
        else:
            vox_with_query =  vox_feat.reshape(B,H,W,D,-1).permute(0,4,1,2,3)
        ############   Voxel_Net   ################
        if self.use_vox_atten == False:
            for i in range(self.conv3d_layers):
                vox_with_query = self.conv_net[i](vox_with_query)[0]
        else:
            if self.vox_att_cfg is not None:
                query = vox_with_query.reshape(B, embed, -1).permute(0, 2, 1) # [B, N, C]
                query = self.vox_att_cfg(query,
                                        value = query,
                                        identity = query,
                                        query_pos = voxel_pos,
                                        reference_points=ref_3d,
                                        spatial_shapes=torch.tensor([H, W, D],device=query.device).to(torch.long),
                                        level_start_index = torch.tensor([0], device=vox_feat.device)
                                        )
                query = self.norm(query)
                
                query = self.norm_ffn(self.ffn(query))
                vox_with_query = query.reshape(B, H, W, D, embed).permute(0,4,1,2,3)
            
        if not self.without_occ:
            up_vox_occ = self.up_block(vox_with_query) # (B, C, H, W, D)
            up_vox_occ = self.proj_layer(up_vox_occ)
            vox_occ = self.vox_occ_net(up_vox_occ).permute(0,1,3,2,4)  # (B, C, W, H, D)
        vox_with_query = vox_with_query.permute(0,2,3,4,1) # (B, H, W, D, C)
        if not self.without_occ:
            return vox_with_query,vox_occ,up_vox_occ
        else:
            return vox_with_query,vox_occ,up_vox_occ
    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

# class CrossAttentionLayer(MessagePassing):

#     def __init__(self,
#                  embed_dim: int,
#                  num_heads: int = 8,
#                  dropout: float = 0.1,
#                  use_edge_pos: bool = False,
#                  **kwargs) -> None:
#         super(CrossAttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
#         self.embed_dim = embed_dim
#         self.num_heads = num_heads
#         self.use_edge_pos = use_edge_pos
#         self.lin_q_node = nn.Linear(embed_dim, embed_dim)
#         self.lin_k_node = nn.Linear(embed_dim, embed_dim)
#         self.lin_v_node = nn.Linear(embed_dim, embed_dim)
#         if self.use_edge_pos:
#             self.lin_k_edge = nn.Linear(embed_dim, embed_dim)

#         self.attn_drop = nn.Dropout(dropout)
#         self.lin_ih = nn.Linear(embed_dim, embed_dim)
#         self.lin_hh = nn.Linear(embed_dim, embed_dim)
#         self.out_proj = nn.Linear(embed_dim, embed_dim)
#         self.lin_self = nn.Linear(embed_dim,embed_dim)
#         self.proj_drop = nn.Dropout(dropout)
#         self.norm1 = nn.LayerNorm(embed_dim)
#         self.norm3 = nn.LayerNorm(embed_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim * 4),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#             nn.Linear(embed_dim * 4, embed_dim),
#             nn.Dropout(dropout))
#         self.alpha = 0
#         self.index = 0

#     def forward(self,
#                 x: torch.Tensor,
#                 edge_index: Adj,
#                 edge_attr: OptTensor = None,
#                 voxel_pos_embedding: torch.Tensor = None,
#                 size: Size = None,) -> torch.Tensor:
#         x_source,x_target = x
#         if voxel_pos_embedding is not None:
#             x_target_with_pos = x_target + voxel_pos_embedding.flatten(0,1)
#             x_target = x_target + self._mha_block(x_target_with_pos,self.norm1(x_source), edge_index,edge_attr, size)
#         else:
#             x_target = x_target + self._mha_block( x_target,self.norm1(x_source), edge_index,edge_attr, size)
#         x_target = x_target + self._ff_block(self.norm3(x_target))
#         return x_target

#     def message(self,
#                 x_i: torch.Tensor,
#                 x_j: torch.Tensor,
#                 edge_attr_j: OptTensor,
#                 index: torch.Tensor,
#                 ptr: OptTensor,
#                 size_i: Optional[int]) -> torch.Tensor:
#         query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         if self.use_edge_pos:
#             key_edge = self.lin_k_edge(edge_attr_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
#         scale = (self.embed_dim // self.num_heads) ** 0.5

#         if self.use_edge_pos:
#             alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
#         else:
#             alpha = (query * key_node).sum(dim=-1) / scale
#         alpha = softmax(alpha, index, ptr, size_i)
#         self.alpha = alpha
#         self.index = index
#         alpha = self.attn_drop(alpha)

#         return value_node * alpha.unsqueeze(-1)
#     def update(self,
#                inputs: torch.Tensor,
#                x: torch.Tensor) -> torch.Tensor:
#         x = x[1]
#         inputs = inputs.view(-1, self.embed_dim)
#         gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
#         return (1-gate)*self.lin_self(x)+gate*inputs
#         # return inputs * gate

#     def _mha_block(self,
#                    x_target: torch.Tensor,
#                    x_source: torch.Tensor,
#                    edge_index: Adj,
#                    edge_attr: OptTensor,
#                    size: Size) -> torch.Tensor:
#         x = self.out_proj(self.propagate(edge_index=edge_index, x=(x_source,x_target), edge_attr=edge_attr,size=size))
#         return self.proj_drop(x)

#     def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
#         return self.mlp(x)






# import torch
# import torch.nn.functional as F
# import torch.nn as nn
# from .embedding import MultipleInputEmbedding,SingleInputEmbedding
# from .utils import init_weights
# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.typing import Adj
# from torch_geometric.typing import OptTensor
# from torch_geometric.typing import Size
# from torch_geometric.utils import softmax
# from typing import Optional
# import pdb
# from mmcv.cnn import xavier_init, constant_init, build_conv_layer, build_norm_layer, build_upsample_layer
# from mmcv.cnn.bricks.transformer import build_attention
# from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
# import cv2
# from mmdet.models import build_backbone
# from ..blocks import DAF
# from mmcv.utils import build_from_cfg
# from mmcv.cnn.bricks.registry import (
#     FEEDFORWARD_NETWORK,
# )
# from torch import Tensor

# import time

# torch._C._jit_set_bailout_depth(1) 
# __all__ = ["Interaction_Net"]

# X, Y, Z, W, L, H, SIN, COS = 0, 1, 2, 3, 4, 5, 6, 7 
# def box3d_to_corners(box3d):
#     # Define constants
    
#     boxes = box3d.clone().detach()
#     boxes[..., 3:6] = boxes[..., 3:6].exp()

#     corners_norm = torch.stack(torch.meshgrid(torch.arange(2), torch.arange(2), torch.arange(2)), dim=-1).view(-1, 3)
#     corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]] - 0.5
#     corners = boxes[..., None, [W, L, H]] * corners_norm.to(boxes.device).reshape(1, 8, 3)

#     rot_mat = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(boxes.shape[0], boxes.shape[1], 1, 1).to(boxes.device)
#     rot_mat[..., 0, 0], rot_mat[..., 0, 1], rot_mat[..., 1, 0], rot_mat[..., 1, 1] = boxes[..., COS], -boxes[..., SIN], boxes[..., SIN], boxes[..., COS]

#     corners = (rot_mat.unsqueeze(2) @ corners.unsqueeze(-1)).squeeze(-1)
#     return corners + boxes[..., None, :3] 
# # @torch.jit.script
# # def box3d_to_corners(box3d: Tensor, W: int, L: int, H: int, SIN: int, COS: int) -> Tensor:
# #     boxes = box3d.clone()
# #     boxes[:, 3:6] = boxes[:, 3:6].exp()  # JIT에서는 다차원 슬라이싱도 명확히 표현 필요

# #     corners_norm = torch.tensor([
# #         [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
# #         [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
# #     ], dtype=box3d.dtype, device=box3d.device) - 0.5

# #     # 필요한 차원별로 명시적 인덱싱
# #     sizes = boxes[:, [W, L, H]].unsqueeze(1)  # [N, 1, 3]
# #     corners = sizes * corners_norm.unsqueeze(0)  # [N, 8, 3]

# #     # 회전 행렬 구성
# #     cos, sin = boxes[:, COS].unsqueeze(-1), boxes[:, SIN].unsqueeze(-1)
# #     rot_mat = torch.zeros(boxes.size(0), 3, 3, dtype=boxes.dtype, device=boxes.device)
# #     rot_mat[:, 0, 0] = cos.squeeze(-1)
# #     rot_mat[:, 0, 1] = -sin.squeeze(-1)
# #     rot_mat[:, 1, 0] = sin.squeeze(-1)
# #     rot_mat[:, 1, 1] = cos.squeeze(-1)
# #     rot_mat[:, 2, 2] = 1  # Z축 회전은 변경 없음  # [N,3,3]

# #     # 회전 및 변환
# #     corners = torch.matmul(rot_mat.unsqueeze(1), corners.unsqueeze(-1)).squeeze(-1)
# #     corners += boxes[:, :3].unsqueeze(1)  # 위치 변환 적용
# #     return corners


# # @torch.jit.script
# # def compute_voxel_indices(agent_pos: Tensor, indices: Tensor, vox_lower_point: Tensor, vox_res: Tensor, 
# #                           vox_size: Tensor, vox_flatten_size: int, batch_num: Tensor, device: torch.device):
# #     vox_indices = torch.empty(0, dtype=torch.long, device=device)
# #     query_indices = torch.empty(0, dtype=torch.long, device=device)
# #     query_total = 0

# #     # for b in range(B):
# #     bottom_pos = box3d_to_corners(agent_pos[indices], W=3, L=4, H=5, COS=6, SIN=7)
# #     vox_pos = ((bottom_pos - vox_lower_point) / vox_res).round()  # [N, 8, 3]

# #     min_pos = torch.zeros_like(vox_pos.min(dim=1)[0])
# #     max_pos = torch.zeros_like(vox_pos.max(dim=1)[0])

# #     min_pos[:, 0] = torch.clamp(vox_pos.min(dim=1)[0].select(1, 0), min=0, max=vox_size[0] - 1)
# #     min_pos[:, 1] = torch.clamp(vox_pos.min(dim=1)[0].select(1, 1), min=0, max=vox_size[1] - 1)
# #     min_pos[:, 2] = torch.clamp(vox_pos.min(dim=1)[0].select(1, 2), min=0, max=vox_size[2] - 1)

# #     max_pos[:, 0] = torch.clamp(vox_pos.max(dim=1)[0].select(1, 0), min=0, max=vox_size[0] - 1)
# #     max_pos[:, 1] = torch.clamp(vox_pos.max(dim=1)[0].select(1, 1), min=0, max=vox_size[1] - 1)
# #     max_pos[:, 2] = torch.clamp(vox_pos.max(dim=1)[0].select(1, 2), min=0, max=vox_size[2] - 1)
# #         # Parallelized processing for all boxes
# #     for num in range(vox_pos.shape[0]):
# #         xs = torch.arange(min_pos[num, 0].item(), max_pos[num, 0].item() + 1, device=device)
# #         ys = torch.arange(min_pos[num, 1].item(), max_pos[num, 1].item() + 1, device=device)
# #         zs = torch.arange(min_pos[num, 2].item(), max_pos[num, 2].item() + 1, device=device)

# #         # Generate meshgrid and flatten
# #         mesh_index = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1).reshape(-1, 3)
# #         indexes = (
# #             mesh_index[..., 1] * vox_size[1] * vox_size[2]
# #             + mesh_index[..., 0] * vox_size[2]
# #             + mesh_index[..., 2]
# #         )
# #         indexes = torch.clip(indexes, min=0, max=(vox_size[0] * vox_size[1] * vox_size[2] - 1))
# #         indexes += batch_num[num] * vox_flatten_size

# #         # Unique indexes and query mapping
# #         unique_indexes = torch.unique(indexes)
# #         vox_indices = torch.cat((vox_indices, unique_indexes))
# #         query_indices = torch.cat((query_indices, torch.full((unique_indexes.size(0),), query_total, dtype=torch.long, device=device)))
# #         query_total += 1
# #     return vox_indices, query_indices
# # @torch.jit.script
# # def compute_voxel_indices(
# #     agent_pos: Tensor, indices: Tensor, vox_lower_point: Tensor, vox_res: Tensor, 
# #     vox_size: Tensor, vox_flatten_size: int, batch_num: Tensor, device: torch.device
# # ):
# #     # Bounding box corners
# #     bottom_pos = box3d_to_corners(agent_pos[indices], W=3, L=4, H=5, SIN=6, COS=7)
# #     vox_pos = ((bottom_pos - vox_lower_point) / vox_res).round()  # [N, 8, 3]

# #     # Vox size를 각각 분리
# #     vox_size_x, vox_size_y, vox_size_z = vox_size[0].item(), vox_size[1].item(), vox_size[2].item()
    
# #     min_pos= vox_pos.min(dim=1)[0]
# #     max_pos = vox_pos.max(dim=1)[0]
# #     min_pos[:,0] = torch.clamp(min_pos[:,0], 0, vox_size_x-1)
# #     min_pos[:,1] = torch.clamp(min_pos[:,1], 0, vox_size_y-1)
# #     min_pos[:,2] = torch.clamp(min_pos[:,2], 0, vox_size_z-1)
# #     max_pos[:,0] = torch.clamp(max_pos[:,0], 0, vox_size_x-1)
# #     max_pos[:,1] = torch.clamp(max_pos[:,1], 0, vox_size_y-1)
# #     max_pos[:,2] = torch.clamp(max_pos[:,2], 0, vox_size_z-1)


# #     # Generate voxel indices in a single operation
# #     xs = [torch.arange(min_pos[i, 0].item(), max_pos[i, 0].item() + 1, device=device) for i in range(min_pos.size(0))]
# #     ys = [torch.arange(min_pos[i, 1].item(), max_pos[i, 1].item() + 1, device=device) for i in range(min_pos.size(0))]
# #     zs = [torch.arange(min_pos[i, 2].item(), max_pos[i, 2].item() + 1, device=device) for i in range(min_pos.size(0))]

# #     # Create full 3D grid per box using list comprehension and tensor stacking
# #     mesh_indices = [
# #         torch.stack(torch.meshgrid(xs[i], ys[i], zs[i]), dim=-1).reshape(-1, 3) for i in range(len(xs))
# #     ]

# #     # Process all indices in parallel
# #     all_indices = []
# #     all_queries = []
# #     query_total = 0

# #     for i, mesh_index in enumerate(mesh_indices):
# #         # Convert 3D grid to linear indices
# #         linear_indices = (
# #             mesh_index[:, 1] * vox_size_y * vox_size_z +
# #             mesh_index[:, 0] * vox_size_z +
# #             mesh_index[:, 2]
# #         )
# #         # Adjust with batch offsets
# #         linear_indices += batch_num[i] * vox_flatten_size

# #         # Store unique indices and query mappings
# #         unique_indices = torch.unique(linear_indices)
# #         all_indices.append(unique_indices)
# #         all_queries.append(torch.full((unique_indices.size(0),), query_total, dtype=torch.long, device=device))
# #         query_total += 1

# #     # Concatenate all results
# #     # all_indices가 [] 인 경우 예외처리로 torch.tensor([])반환
# #     if len(all_indices) == 0:
# #         return torch.empty(0, dtype=torch.long, device=device), torch.empty(0, dtype=torch.long, device=device)

# #     vox_indices = torch.cat(all_indices)
# #     query_indices = torch.cat(all_queries)

# #     return vox_indices, query_indices
    
# @PLUGIN_LAYERS.register_module()
# class Interaction_Net(nn.Module):

#     def __init__(self,
#                  embed_dims: int,
#                  conv_cfg: dict,
#                  grid_config: dict,
#                  num_heads: int = 8,
#                  dropout: float = 0.1,
#                  conv3d_layers: int = 1,
#                  num_classes: int = 18,
#                  down_ratio: int = 4,
#                  use_edge_pos: bool = False,
#                  img_to_vox: bool = False,
#                  num_cams: int = 6,
#                  num_levels: int = 4,
#                  num_points: int = 6,
#                  num_groups: int = 8,
#                  query_to_vox: bool = True,
#                  without_occ: bool = False,
#                  vox_att_cfg: dict = None,
#                  ffn: dict = None,
#                  ) -> None:

#         super(Interaction_Net, self).__init__()
#         self.use_edge_pos=use_edge_pos
#         self.without_occ = without_occ
#         self.use_vox_atten = True if vox_att_cfg is not None else False
#         deconv_cfg = dict(type='deconv3d', bias=False)
#         conv3d_cfg=dict(type='Conv3d', bias=False)
#         gn_norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
#         out_dims = embed_dims//down_ratio
#         self.embed_dims = embed_dims
#         self.conv3d_layers = conv3d_layers
#         self.num_classes = num_classes
#         self.img_to_vox = img_to_vox
#         self.query_to_vox = query_to_vox
#         if img_to_vox:
#             self.attention_weights =  nn.Linear(embed_dims, num_groups * num_cams * num_levels * num_points)
#             self.sampling_offsets = nn.Linear(embed_dims,  num_points * 3)
#             self.out_proj = nn.Linear(embed_dims,embed_dims)
#             self.proj_drop = nn.Dropout(0.1)
#             self.num_cams = num_cams
#             self.num_levels = num_levels
#             self.num_points = num_points
#             self.num_groups = num_groups

#         self.down_ratio = down_ratio
#         self.vox_size = [int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][-1]),
#                          int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][-1]),
#                          int((grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][-1])]
#         self.vox_res = [grid_config['y'][-1],grid_config['x'][-1],grid_config['z'][-1]]
#         self.vox_lower_point = [grid_config['y'][0],
#                                  grid_config['x'][0],
#                                  grid_config['z'][0]]
#         self.norm_cross = nn.LayerNorm(embed_dims)

#         if query_to_vox:
#             self.cross_attn_Q2V = CrossAttentionLayer(embed_dim=embed_dims,
#                                                         num_heads=num_heads,
#                                                         dropout=dropout,
#                                                         use_edge_pos=use_edge_pos)
#         if vox_att_cfg is None:
#             self.conv_net = nn.ModuleList(
#                 [
#                     build_backbone(conv_cfg)
#                     for _ in range(conv3d_layers)
#                 ]
#             )
#         else:
#             self.vox_att_cfg = build_attention(vox_att_cfg)
#             self.norm = nn.LayerNorm(embed_dims)
#             if ffn is not None:
#                 self.ffn = build_from_cfg(ffn,FEEDFORWARD_NETWORK)
#             self.norm_ffn = nn.LayerNorm(embed_dims)
            
#         if not without_occ:
            
#             upsample = build_upsample_layer(deconv_cfg, embed_dims, out_dims, kernel_size=2, stride=2)
#             self.up_block = nn.Sequential(upsample,
#                             nn.BatchNorm3d(out_dims),
#                             nn.ReLU(inplace=True)
#                             )
#             self.vox_occ_net = build_conv_layer(
#                     conv3d_cfg,
#                     in_channels=out_dims,
#                     out_channels=self.num_classes,
#                     kernel_size=1,
#                     stride=1,
#                     padding=0)
            
#         self.apply(init_weights)

#     def forward(self,
#                 agent_feat: torch.Tensor = None, # [B, N, embed_dim]
#                 vox_feat:torch.Tensor = None,  #[B,H,W,Z,embed_dim]
#                 agent_pos: torch.Tensor = None, # [B,N,11]
#                 agent_pos_embed: torch.Tensor = None, # [B,N,embed_dim]
#                 indices = None,  # [B, N]       
#                 metas = None,
#                 img_feats = None,
#                 ref_3d = None,
#                 voxel_pos = None,
#                 ) -> torch.Tensor:
        

        
#         vox_indices = []
#         query_indices = []
#         vox_occ = None
#         vox_flatten_size = vox_feat.shape[1] * vox_feat.shape[2] * vox_feat.shape[3]
#         B, H, W, D, embed = vox_feat.shape
        
#         query_total = 0
#         i2v_pos = []
#         if self.img_to_vox:
#             vox_feat = vox_feat.view(B, vox_flatten_size, embed)
        
#         # batch_sizes = torch.tensor([len(num) for num in indices], device=agent_pos.device)
#         # batch_num = torch.arange(len(indices), device=agent_pos.device).repeat_interleave(batch_sizes)
#         # indices_tensor = torch.cat(indices, dim=0).to(agent_pos.device)
#         # agent_num = agent_pos.shape[1]
#         # indices_tensor += batch_num * agent_num  # [N]
#         # with torch.no_grad():
#         #     vox_indices, query_indices = compute_voxel_indices(
#         #         agent_pos.flatten(0,1), indices_tensor,
#         #         torch.tensor(self.vox_lower_point, dtype=agent_pos.dtype, device=agent_pos.device),
#         #         torch.tensor(self.vox_res, dtype=agent_pos.dtype, device=agent_pos.device),
#         #         torch.tensor(self.vox_size, dtype=agent_pos.dtype, device=agent_pos.device),
#         #         vox_flatten_size, batch_num, vox_feat.device
#         #     )
#         for b in range(B):
#             bottom_pos = box3d_to_corners(agent_pos[b][indices[b]].unsqueeze(0)).squeeze(0)  # [B, N, 8, 3]
#             vox_pos = ((bottom_pos - torch.tensor(self.vox_lower_point, dtype=agent_pos.dtype, device=agent_pos.device))
#                     / torch.tensor(self.vox_res, dtype=agent_pos.dtype, device=agent_pos.device)).round()
#             min_pos= vox_pos.min(dim=1)[0]
#             max_pos = vox_pos.max(dim=1)[0]
#             min_pos[:,0] = torch.clamp(min_pos[:,0], 0, self.vox_size[0]-1)
#             min_pos[:,1] = torch.clamp(min_pos[:,1], 0, self.vox_size[1]-1)
#             min_pos[:,2] = torch.clamp(min_pos[:,2], 0, self.vox_size[2]-1)
#             max_pos[:,0] = torch.clamp(max_pos[:,0], 0, self.vox_size[0]-1)
#             max_pos[:,1] = torch.clamp(max_pos[:,1], 0, self.vox_size[1]-1)
#             max_pos[:,2] = torch.clamp(max_pos[:,2], 0, self.vox_size[2]-1)
#             for num in range(vox_pos.shape[0]):
#                 xs = torch.arange(min_pos[num, 0].item(), max_pos[num, 0].item() + 1, device=vox_feat.device)
#                 ys = torch.arange(min_pos[num, 1].item(), max_pos[num, 1].item() + 1, device=vox_feat.device)
#                 zs = torch.arange(min_pos[num, 2].item(), max_pos[num, 2].item() + 1, device=vox_feat.device)

#                 mesh_index = torch.stack(torch.meshgrid(xs, ys, zs), dim=-1).reshape(-1, 3)
#                 if self.img_to_vox:
#                     i2v_pos.extend(torch.tensor(mesh_index))
#                 indexes = (mesh_index[..., 1] * self.vox_size[1] * self.vox_size[2] +
#                         mesh_index[..., 0] * self.vox_size[2] +
#                         mesh_index[..., 2])
#                 indexes = torch.clip(indexes, min=0, max=(self.vox_size[0] * self.vox_size[1] * self.vox_size[2] - 1))
#                 indexes += b * vox_flatten_size
#                 unique_indexes = torch.unique(indexes)
#                 vox_indices.extend(unique_indexes.tolist())
#                 query_indices.extend([query_total] * len(unique_indexes))
#                 query_total+=1
#             # if self.img_to_vox:
#             #     i2v_pos = torch.unique(torch.stack(i2v_pos),dim=0).to(torch.long)
#             #     i2v_indices = torch.tensor(i2v_pos[..., 1]*self.vox_size[1] * self.vox_size[2] +i2v_pos[...,0]*self.vox_size[2]+ i2v_pos[...,2]).to(torch.long)
#             #     occupied_vox = torch.gather(vox_feat[b].clone(), 0, i2v_indices[:,None].repeat(1,vox_feat.shape[-1])).unsqueeze(0) # [1, N, embed_dim]
#             #     bs = occupied_vox.shape[0]
#             #     sampling_offsets = self.sampling_offsets(occupied_vox)
#             #     sampling_offsets = sampling_offsets.view(bs, -1, self.num_points, 3)
#             #     attention_weights = self.attention_weights(occupied_vox) 
#             #     attention_weights = attention_weights.view(bs, -1,  self.num_points*self.num_cams*self.num_levels * self.num_groups)
#             #     attention_weights = attention_weights.softmax(-2).view(bs, -1, self.num_points, self.num_cams, self.num_levels, self.num_groups)
#             #     # offset_normalizer = torch.tensor([vox_feat.shape[1], vox_feat.shape[2],vox_feat.shape[3]]).to(vox_feat.device)
#             #     sampling_locations = i2v_pos[None, :, None, :] + sampling_offsets
#             #     sampling_locations += - torch.tensor(self.vox_lower_point, dtype=agent_pos.dtype, device=agent_pos.device)
#             #     sampling_locations = sampling_locations * torch.tensor(self.vox_res, dtype=agent_pos.dtype, device=agent_pos.device)
#             #     points_2d = (
#             #             self.project_points(
#             #                 sampling_locations, #[bs, num_anchor, num_pts, 3]
#             #                 metas["projection_mat"][b:b + 1],
#             #                 metas.get("image_wh")[b:b + 1],
#             #             )
#             #             .permute(0, 2, 3, 1, 4)
#             #             .reshape(bs, -1, self.num_points, self.num_cams, 2)
#             #     )
#             #     img_feat = [img_feats[0][b].unsqueeze(0), img_feats[1].unsqueeze(0), img_feats[2]]
#             #     features = DAF(*img_feat, points_2d, attention_weights).reshape(bs, -1, self.embed_dims)   
#             #     features = self.proj_drop(self.out_proj(features))
#             #     occupied_vox = occupied_vox + features
#             #     vox_feat[b] = torch.scatter(vox_feat[b].clone(),0,i2v_indices[:,None].repeat(1,vox_feat.shape[-1]),occupied_vox.squeeze(0))
#             #     i2v_pos = []

#         if self.img_to_vox:
#             vox_with_query = vox_feat.reshape(B,H,W,D,-1).permute(0,4,1,2,3)
        
#         if self.query_to_vox:
#             edge_index_Q2V = torch.stack([torch.tensor(query_indices, device=vox_feat.device, dtype=torch.long),
#                                         torch.tensor(vox_indices, device=vox_feat.device, dtype=torch.long)], dim=0)  
            
#             # #############   Query to Voxel    #####################
#             selected_agent_feat = torch.cat([agent_feat[b][indices[b]] for b in range(agent_feat.shape[0])],dim=0)
#             if self.use_edge_pos:
#                 selected_agent_pos = torch.cat([agent_pos_embed[b][indices[b]] for b in range(agent_pos_embed.shape[0])],dim=0).reshape(-1,self.embed_dims)
#                 vox_with_query = self.cross_attn_Q2V(x=(selected_agent_feat,vox_feat.reshape(-1,self.embed_dims)),edge_index=edge_index_Q2V,edge_attr=selected_agent_pos,voxel_pos_embedding=voxel_pos)
#             else:
#                 vox_with_query = self.cross_attn_Q2V(x=(selected_agent_feat,vox_feat.reshape(-1,self.embed_dims)),edge_index=edge_index_Q2V,voxel_pos_embedding=voxel_pos)
#             vox_with_query = self.norm_cross(vox_with_query)
#             vox_with_query = vox_with_query.reshape(-1,self.vox_size[0],self.vox_size[1],self.vox_size[2],self.embed_dims)

#             vox_with_query = vox_with_query.permute(0,4,1,2,3) # [B, C, H, W, D]
#         # #############   Voxel_Net   ################
#         if self.use_vox_atten == False:
#             for i in range(self.conv3d_layers):
#                 vox_with_query = self.conv_net[i](vox_with_query)[0]
#         else:

#             query = vox_with_query.reshape(B, embed, -1).permute(0, 2, 1) # [B, N, C]
#             query = self.vox_att_cfg(query,
#                                     value = query,
#                                     identity = query,
#                                     query_pos = voxel_pos,
#                                     reference_points=ref_3d,
#                                     spatial_shapes=torch.tensor([H, W, D],device=query.device).to(torch.long),
#                                     level_start_index = torch.tensor([0], device=vox_feat.device)
#                                     )
#             query = self.norm(query)
            
#             query = self.norm_ffn(self.ffn(query))
#             vox_with_query = query.reshape(B, H, W, D, embed).permute(0,4,1,2,3)

#         if not self.without_occ:
#             vox_occ = self.up_block(vox_with_query) # (B, C, H, W, D)
#             vox_occ = self.vox_occ_net(vox_occ).permute(0,1,3,2,4)  # (B, C, W, H, D)
#         vox_with_query = vox_with_query.permute(0,2,3,4,1) # (B, H, W, D, C)
#         if not self.without_occ:
#             return vox_with_query,vox_occ
#         else:
#             return vox_with_query,vox_occ
#     @staticmethod
#     def project_points(key_points, projection_mat, image_wh=None):
#         pts_extend = torch.cat(
#             [key_points, torch.ones_like(key_points[..., :1])], dim=-1
#         )
#         points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
#         points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
#         if image_wh is not None:
#             points_2d = points_2d / image_wh[:, :, None, None]
#         return points_2d

class CrossAttentionLayer(MessagePassing):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_edge_pos: bool = False,
                 **kwargs) -> None:
        super(CrossAttentionLayer, self).__init__(aggr='add', node_dim=0, **kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_edge_pos = use_edge_pos
        self.lin_q_node = nn.Linear(embed_dim, embed_dim)
        self.lin_k_node = nn.Linear(embed_dim, embed_dim)
        self.lin_v_node = nn.Linear(embed_dim, embed_dim)
        if self.use_edge_pos:
            self.lin_k_edge = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.lin_ih = nn.Linear(embed_dim, embed_dim)
        self.lin_hh = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout))
        self.alpha = 0
        self.index = 0

    def forward(self,
                x: torch.Tensor,
                edge_index: Adj,
                edge_attr: OptTensor = None,
                voxel_pos_embedding: OptTensor = None,
                size: Size = None,) -> torch.Tensor:
        x_source,x_target = x
        # if voxel_pos_embedding is not None:
        #     x_target_with_pos = x_target + voxel_pos_embedding.flatten(0,1)
        x_target = x_target + self._mha_block(x_target,self.norm1(x_source), edge_index,edge_attr,voxel_pos_embedding, size)
        # else:
        #     x_target = x_target + self._mha_block( x_target,self.norm1(x_source), edge_index,edge_attr, size)
        x_target = x_target + self._ff_block(self.norm3(x_target))
        return x_target

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr_j: OptTensor,
                vox_pos_embedding_i: OptTensor,
                index: torch.Tensor,
                ptr: OptTensor,
                size_i: Optional[int]) -> torch.Tensor:
        if vox_pos_embedding_i is not None:
            x_i = x_i + vox_pos_embedding_i
        query = self.lin_q_node(x_i).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        key_node = self.lin_k_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        value_node = self.lin_v_node(x_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
        if self.use_edge_pos:
            key_edge = self.lin_k_edge(edge_attr_j).view(-1, self.num_heads, self.embed_dim // self.num_heads)
            # value_edge = self.lin_v_edge(edge_attr_j).view(-1, self.num_heads, self.embed_dim // self.num_heads) 
        scale = (self.embed_dim // self.num_heads) ** 0.5

        if self.use_edge_pos:
            alpha = (query * (key_node + key_edge)).sum(dim=-1) / scale
        else:
            alpha = (query * key_node).sum(dim=-1) / scale
        alpha = softmax(alpha, index, ptr, size_i)
        self.alpha = alpha
        self.index = index
        alpha = self.attn_drop(alpha)
        # pdb.set_trace()
        return value_node * alpha.unsqueeze(-1)* x_j.sum(-1).bool()[...,None,None]

    def update(self,
               inputs: torch.Tensor,
               x: torch.Tensor) -> torch.Tensor:
        x = x[1]
        inputs = inputs.view(-1, self.embed_dim)
        gate = torch.sigmoid(self.lin_ih(inputs) + self.lin_hh(x))
        return inputs * gate

    def _mha_block(self,
                   x_target: torch.Tensor,
                   x_source: torch.Tensor,
                   edge_index: Adj,
                   edge_attr: OptTensor,
                   vox_pos_embedding: OptTensor,
                   size: Size) -> torch.Tensor:
        if vox_pos_embedding is not None:
            vox_pos_embedding = vox_pos_embedding.flatten(0,1)
        x = self.out_proj(self.propagate(edge_index=edge_index, x=(x_source,x_target), edge_attr=edge_attr,vox_pos_embedding=vox_pos_embedding,size=size))
        return self.proj_drop(x)

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)