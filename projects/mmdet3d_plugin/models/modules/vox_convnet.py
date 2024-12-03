import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import init_weights
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from mmcv.cnn import xavier_init, constant_init, build_conv_layer, build_norm_layer, build_upsample_layer
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
import pdb
from ..blocks import DAF
from mmdet.models import build_backbone
from mmcv.runner.base_module import Sequential
from mmcv.cnn import Linear
__all__ = ["Vox_Convnet"]

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

@PLUGIN_LAYERS.register_module()
class Vox_Convnet(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 conv_cfg: dict,
                 fin_conv_cfg: dict = None,
                 top_k_ratio = 0.1,
                 use_temporal: bool = False,
                 use_occ_loss: bool = False,
                 img_to_voxel: bool = False,
                 temporal_layers: int = 2,
                 current_layers: int = 1,
                 pred_occ: bool = False,
                 use_mask_token: bool = False,
                 only_background: bool = False,
                 up_sample: bool = False,
                 grid_config = None,
                 temp_cat_method: str = 'add',
                 num_groups=8,
                 num_cams=6,
                 num_levels=4,
                 num_points=6,
                 attn_drop=0.15,
                 down_ratio=4,
                 num_classes = 17,
                 interaction_range:float = 5.0,
                 use_COTR_version = False,
                 FPN_with_pred = False,
                 save_high_res_feature = False,
                 transformer=None, # IVT transformer
                 use_upsample = False,
                #  positional_encoding=None,
                 ):
        super(Vox_Convnet, self).__init__()
        self.attn_drop = attn_drop
        self.embed_dims = embed_dims
        self.use_upsample = use_upsample
        self.use_occ_loss = use_occ_loss
        self.img_to_voxel = img_to_voxel
        self.save_high_res_feature = save_high_res_feature
        self.interaction_range = interaction_range
        self.temporal_layers = temporal_layers
        self.only_background = only_background
        self.current_layers = current_layers
        self.top_k_ratio = top_k_ratio
        self.pred_occ = pred_occ
        self.FPN_with_pred = FPN_with_pred
        self.use_COTR_version = use_COTR_version
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.num_levels = num_levels
        self.use_temporal = use_temporal
        self.num_points = num_points
        self.temp_cat_method = temp_cat_method
        self.down_ratio = down_ratio
        # self.fin_conv_net = build_backbone(fin_conv_cfg)
        self.num_classes = num_classes
        self.pred_occ_net = None
        self.embed_dims = embed_dims
        self.occ_net = None
        self.up_sample = up_sample
        self.use_mask_token = use_mask_token
        conv3d_cfg=dict(type='Conv3d', bias=False)
        gn_norm_cfg=dict(type='GN', num_groups=16, requires_grad=True)
        if use_mask_token:
            self.mask_embed = nn.Embedding(1, embed_dims)
        if transformer is not None:
            self.transformer = build_transformer(transformer)
        else:
            self.transformer = None
        # self.positional_encoding = build_positional_encoding(positional_encoding) if positional_encoding is not None else None

        if use_temporal:
            self.temporal_conv_net = nn.ModuleList(
                [
                    build_backbone(conv_cfg)
                    for _ in range(temporal_layers)
                ]
            )
            if FPN_with_pred is not True:
                if self.temp_cat_method == 'cat':
                    conv = build_conv_layer(conv3d_cfg, embed_dims*2, embed_dims, kernel_size=1, stride=1)
                    self.cat_block = nn.Sequential(conv,
                                    build_norm_layer(gn_norm_cfg, embed_dims)[1],
                                    nn.ReLU(inplace=True))            
        self.camera_encoder = None

        if use_occ_loss or img_to_voxel:
            if use_temporal:
                self.occ_net = nn.ModuleList([nn.Linear(embed_dims,1) for _ in range(temporal_layers)])
            else:
                self.occ_net = nn.ModuleList([nn.Linear(embed_dims,1) for _ in range(current_layers)])
            if img_to_voxel:
                self.camera_encoder = Sequential(
                    *linear_relu_ln(embed_dims, 1, 2, 12)
                )

                if use_temporal:
                    self.attention_weights = nn.ModuleList(
                        [
                            nn.Linear(embed_dims, num_groups * num_levels * num_points) for _ in range(temporal_layers)
                        ]
                    )
                    self.sampling_offsets = nn.ModuleList(
                        [
                            nn.Linear(embed_dims,  num_points * 3) for _ in range(temporal_layers)
                        ]
                    )
                    self.out_proj = nn.ModuleList([nn.Linear(embed_dims,embed_dims) for _ in range(temporal_layers) ])
                    self.proj_drop = nn.ModuleList([nn.Dropout(0.1) for _ in range(temporal_layers)])
                else:
                    self.attention_weights = nn.ModuleList(
                        [
                            nn.Linear(embed_dims, num_groups * num_levels * num_points) for _ in range(current_layers)
                        ]
                    )
                    self.sampling_offsets = nn.ModuleList(
                        [
                            nn.Linear(embed_dims,  num_points * 3) for _ in range(current_layers)
                        ]
                    )
                    self.out_proj = nn.ModuleList([nn.Linear(embed_dims,embed_dims) for _ in range(current_layers) ])
                    self.proj_drop = nn.ModuleList([nn.Dropout(0.1) for _ in range(current_layers)])
        self.current_conv_net = nn.ModuleList(
            [
                build_backbone(conv_cfg)
                for _ in range(current_layers)
            ]
        )
        if pred_occ and self.save_high_res_feature is not True:
            if self.up_sample:
                deconv_cfg = dict(type='deconv3d', bias=False)
                out_dims = embed_dims//self.down_ratio
                if use_temporal and temporal_layers>0:
                    if self.use_COTR_version:
                        self.up0 = nn.Sequential(
                            nn.ConvTranspose3d(embed_dims,embed_dims//2,(1,3,3),padding=(0,1,1)),
                            nn.BatchNorm3d(embed_dims//2),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose3d(embed_dims//2, embed_dims//2, (1, 2, 2), stride=(1, 2, 2)),
                            nn.BatchNorm3d(embed_dims//2),
                            nn.ReLU(inplace=True),
                        )
                        self.up1 = nn.Sequential(
                            nn.ConvTranspose3d(embed_dims//2,out_dims,(1,3,3),padding=(0,1,1)),
                            nn.BatchNorm3d(out_dims),
                            nn.ReLU(inplace=True),
                            nn.ConvTranspose3d(out_dims, out_dims, (1, 2, 2), stride=(1, 2, 2)),
                            nn.BatchNorm3d(out_dims),
                            nn.ReLU(inplace=True),
                        )
                    else:
                        pred_upsample = build_upsample_layer(deconv_cfg, embed_dims, out_dims, kernel_size=2, stride=2)
                        
                        
                        self.pred_up_block = nn.Sequential(pred_upsample,
                                        nn.BatchNorm3d(out_dims),
                                        nn.ReLU(inplace=True))
                    self.pred_occ_net = build_conv_layer(
                        conv3d_cfg,
                        in_channels=out_dims,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0)
                # upsample = build_upsample_layer(deconv_cfg, embed_dims, out_dims, kernel_size=2, stride=2)
                # self.up_block = nn.Sequential(upsample,
                #                 build_norm_layer(gn_norm_cfg, out_dims)[1],
                #                 nn.ReLU(inplace=True))
                # self.vox_occ_net = build_conv_layer(
                #         conv3d_cfg,
                #         in_channels=out_dims,
                #         out_channels=self.num_classes,
                #         kernel_size=1,
                #         stride=1,
                #         padding=0)
                if FPN_with_pred is not True:
                    if only_background is not True:
                        upsample = build_upsample_layer(deconv_cfg, self.embed_dims, out_dims, kernel_size=2, stride=2)
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

            else:
                if use_temporal:
                    self.pred_occ_net = build_conv_layer(
                        conv3d_cfg,
                        in_channels=embed_dims,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0)
                self.vox_occ_net = build_conv_layer(
                        conv3d_cfg,
                        in_channels=embed_dims,
                        out_channels=self.num_classes,
                        kernel_size=1,
                        stride=1,
                        padding=0)
        if grid_config is not None:
            self.bev_origin = [grid_config['x'][0], grid_config['y'][0], grid_config['z'][0]]
            self.grid_size = [int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][2]), int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][2]), int((grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][2])]
            self.bev_resolution = [(grid_config['x'][1]-grid_config['x'][0])/self.grid_size[0], (grid_config['y'][1]-grid_config['y'][0])/self.grid_size[1], (grid_config['z'][1]-grid_config['z'][0])/self.grid_size[2]]  
            self.grid_range = [grid_config['x'][1]-grid_config['x'][0], grid_config['y'][1]-grid_config['y'][0], grid_config['z'][1]-grid_config['z'][0]]
    # self.apply(init_weights)
    # def init_weights(self):
    #     if self.use_occ_loss:
    #         for i in range(len(self.occ_net)):
    #             xavier_init(self.occ_net[i],  distribution="uniform", bias=0.0)
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        if self.transformer is not None:
            for p in self.transformer.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
        
    def forward(self,
                voxel_feat,  # [bs, embed_dims,D, H, W]
                prev_vox_feat=None,
                metas=None,
                img_feats=None,
                mlvl_feats=None,
                occ_pos = None,
                ref_3d = None,
                twice_technic = False,
                inter_feat = None,
                **kwargs,
                ) -> torch.Tensor:
        B,C,D,H,W = voxel_feat.shape
        
        if prev_vox_feat is None:
            if self.embed_dims==C:
                prev_vox_feat = voxel_feat.clone().detach()
            else:
                if self.FPN_with_pred and self.save_high_res_feature is not True:
                    prev_vox_feat = voxel_feat.new_zeros(B,self.embed_dims,self.grid_size[2],self.grid_size[0],self.grid_size[1])
                else:
                    prev_vox_feat =voxel_feat.new_zeros(B,self.embed_dims,D,H,W)
        if self.use_mask_token:
            mask_token =  self.mask_embed.weight.view(1,1,1,1,C).expand(B,D,H,W,C).to(voxel_feat.dtype).permute(0,4,1,2,3) # [1,1,1,1,C] -> [B,D,H,W,C] ->  [B,C,D,H,W]
            voxel_feat = torch.where(voxel_feat == 0, mask_token, voxel_feat)
        occ_list = []
        vox_occ_list = []
        query = None
        if self.use_temporal:
            query = prev_vox_feat
            for i in range(self.temporal_layers):
                query = self.temporal_conv_net[i](query)[0] # [B,C,D,H,W]
                bs, C, D, H, W = query.shape
                if self.use_occ_loss or self.img_to_voxel:
                    query = query.permute(0,3,4,2,1).flatten(1,3) # (bs, C, D, H, W) -> (bs, H, W, D, C) -> (bs, H*W*D, C)
                    temp_occ = self.occ_net[i](query)
                    if self.use_occ_loss:
                        occ_list.append(temp_occ)

                        
                        
                    if self.img_to_voxel:
                        
                        # occupancy_scores 계산 및 상위 k개 인덱스 선택
                        occupancy_scores = temp_occ.sigmoid()
                        num_topk = int(self.top_k_ratio * occupancy_scores.shape[1])
                        occupied_scores, occupied_indices = occupancy_scores.topk(num_topk, dim=1)

                        # occupied_indices를 이용해 배치별 인덱스 선택
                        batch_indices = occupied_indices + (torch.arange(bs, device=occupied_indices.device).view(-1, 1) * ref_3d.shape[1])[:,None,:]
                        vox_indices = ref_3d.view(-1, ref_3d.shape[-1]).index_select(0, batch_indices.flatten()).view(bs, num_topk, -1)

                        # occ_pos에서 상위 k개 인덱스를 선택해 배치별 위치 임베딩 계산
                        vox_pos_embedding = occ_pos.reshape(-1, occ_pos.shape[-1]).index_select(0, batch_indices.flatten()).view(bs, num_topk, -1)

                        # occupied_query 계산
                        occupied_query = torch.gather(query, 1, occupied_indices.repeat(1, 1, query.shape[2]))
                        
                        
                        # occupancy_scores = temp_occ.sigmoid()
                        # occupied_indices = occupancy_scores.topk(int(self.top_k_ratio*occupancy_scores.shape[1]),dim=1)[1]
                        # vox_indices = []
                        # vox_pos_embedding = []
                        # for b in range(bs):
                        #     vox_indices.append(torch.index_select(ref_3d[b].clone(),0,occupied_indices[b].squeeze(1)).unsqueeze(0).squeeze(2))
                        #     vox_pos_embedding.append(torch.index_select(occ_pos[b],0,occupied_indices[b].squeeze(1)).unsqueeze(0)) 
                        # pdb.set_trace()
                        # vox_indices = torch.cat(vox_indices,dim=0) # [b,n,3]
                        # vox_pos_embedding = torch.cat(vox_pos_embedding,dim=0)
                        # vox_z = occupied_indices % self.grid_size[2]
                        # vox_y = (occupied_indices // self.grid_size[2]) % self.grid_size[1]
                        # vox_x = (occupied_indices // self.grid_size[2] // self.grid_size[1])
                        # vox_pos = torch.stack([vox_x,vox_y,vox_z],dim=2).squeeze() #[bs, num_points, 3]
                        
                        # occupied_query = torch.gather(query,1,occupied_indices.repeat(1,1,query.shape[2]))
                        sampling_offsets = self.sampling_offsets[i](occupied_query)
                        sampling_offsets = (sampling_offsets.view(
                            bs, -1, self.num_points, 3).sigmoid()-0.5)*self.interaction_range
                        attention_weights = self._get_weights(occupied_query, vox_pos_embedding , metas,i)
                        
                        # attention_weights = self.attention_weights[i](occupied_query).view(
                        #     bs, -1,  self.num_points*self.num_cams*self.num_levels * self.num_groups)
                        # attention_weights = weights.softmax(-2).view(bs, -1, self.num_points, self.num_cams, self.num_levels, self.num_groups)
                        # offset_normalizer = torch.tensor([self.grid_size[1], self.grid_size[0],self.grid_size[2]]).to(query.device)
                        if vox_indices.dim() == 2:
                            vox_indices = vox_indices.unsqueeze(0)
                        voxel_size = torch.tensor([self.grid_range[0],self.grid_range[1],self.grid_range[2]]).to(query.device)
                        voxel_origin = torch.tensor([self.bev_origin[1], self.bev_origin[0], self.bev_origin[2]]).to(query.device)
                        vox_indices = vox_indices*voxel_size + voxel_origin
                        key_points = vox_indices[:, :, None, :] + sampling_offsets 
                        points_2d = (
                            self.project_points(
                                key_points, #[bs, num_anchor, num_pts, 3]
                                metas["projection_mat"],
                                metas.get("image_wh"),
                            )
                            .permute(0, 2, 3, 1, 4)
                            .reshape(bs, -1, self.num_points, self.num_cams, 2)
                        )
                        features = DAF(*img_feats, points_2d, attention_weights).reshape(
                            bs, -1, self.embed_dims
                        )   
                        features = self.proj_drop[i](self.out_proj[i](features))
                        if self.use_occ_loss == False:
                            occupied_scores = torch.gather(occupancy_scores,1,occupied_indices)
                            features = features * occupied_scores
                        occupied_query = occupied_query + features
                        query = torch.scatter(query,1,occupied_indices.repeat(1,1,query.shape[2]),occupied_query)
                    query = query.view(bs,H,W,D,C).permute(0,4,3,1,2) # (bs, H*W*D, C) -> (bs, C, D, H, W)

            if self.transformer is not None:
                bs,_,occ_z, occ_w, occ_h = query.shape
                dtype = query.dtype
                occ_feature = query.permute(0,1,3,4,2)
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
                query = enhanced_occ_feature.permute(0, 2, 1).view(bs, -1, occ_h, occ_w, occ_z).permute(0,1,4,2,3)

            
            if self.training and self.temporal_layers>0:

                if self.pred_occ and self.save_high_res_feature is not True:
                    if self.up_sample:
                        if self.use_COTR_version:
                            up_vox_occ = self.up0(query)
                            up_vox_occ = self.up1(up_vox_occ)
                        else:
                            up_vox_occ = self.pred_up_block(query)
                        vox_occ = self.pred_occ_net(up_vox_occ)
                    else:
                        vox_occ = self.pred_occ_net(query)
                    vox_occ_list.append(vox_occ.permute(0,1,4,3,2))
                # if twice_technic:
                #     query = query.detach()
            if self.FPN_with_pred is not True:
                if self.temp_cat_method == 'cat':
                    query = self.cat_block(torch.cat([query,voxel_feat],dim=1))
                else:
                    query = query + voxel_feat
            else:
                if self.save_high_res_feature is not True:
                    query = up_vox_occ
        else:
            query = voxel_feat
        if self.FPN_with_pred is not True:
            for i in range(self.current_layers):
                query = self.current_conv_net[i](query)[0]# [bs, embed_dims, grid_size[0], grid_size[1], grid_size[2]]
                
                bs, C, D, H, W = query.shape
                
                # if self.use_occ_loss or self.img_to_voxel:
                #     query = query.permute(0,3,4,2,1).flatten(1,3) # (bs, C, D, H, W) -> (bs, H, W, D, C) -> (bs, H*W*D, C)
                #     temp_occ = self.occ_net[self.temporal_layers + i](query) if self.use_temporal else self.occ_net[i](query)
                #     if self.use_occ_loss:
                #         occ_list.append(temp_occ)
                #     if self.img_to_voxel:
                #         occupancy_scores = temp_occ.sigmoid()
                #         occupied_indices = occupancy_scores.topk(int(self.top_k_ratio*occupancy_scores.shape[1]),dim=1)[1]
                #         vox_z = occupied_indices % self.grid_size[2]
                #         vox_y = (occupied_indices // self.grid_size[2]) % self.grid_size[1]
                #         vox_x = (occupied_indices // self.grid_size[2] // self.grid_size[1])
                #         # pdb.set_trace()
                #         vox_pos = torch.stack([vox_x,vox_y,vox_z],dim=2).squeeze() #[bs, num_points, 3]
                #         occupied_query = torch.gather(query,1,occupied_indices.repeat(1,1,query.shape[2]))
                #         sampling_offsets = self.sampling_offsets[self.temporal_layers+i](occupied_query) if self.use_temporal else self.sampling_offsets[i](occupied_query)
                #         sampling_offsets = sampling_offsets.view(
                #             bs, -1, self.num_points, 3)
                #         attention_weights = self.attention_weights[self.temporal_layers+i](occupied_query) if self.use_temporal else self.attention_weights[i](occupied_query)
                #         attention_weights = attention_weights.view(
                #             bs, -1,  self.num_points*self.num_cams*self.num_levels * self.num_groups)
                #         attention_weights = attention_weights.softmax(-2).view(bs, -1, self.num_points, self.num_cams, self.num_levels, self.num_groups)
                #         offset_normalizer = torch.tensor([self.grid_size[1], self.grid_size[0],self.grid_size[2]]).to(query.device)

                #         if vox_pos.dim() == 2:
                #             vox_pos = vox_pos.unsqueeze(0)
                #         sampling_locations = vox_pos[:, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :]
                #         bev_resolution = torch.tensor([self.bev_resolution[1], self.bev_resolution[0], self.bev_resolution[2]]).to(query.device)
                #         bev_origin = torch.tensor([self.bev_origin[1], self.bev_origin[0], self.bev_origin[2]]).to(query.device)
                #         key_points = sampling_locations * bev_resolution[None, None, None, :] + bev_origin[None, None, None, :]
                #         points_2d = (
                #             self.project_points(
                #                 key_points, #[bs, num_anchor, num_pts, 3]
                #                 metas["projection_mat"],
                #                 metas.get("image_wh"),
                #             )
                #             .permute(0, 2, 3, 1, 4)
                #             .reshape(bs, -1, self.num_points, self.num_cams, 2)
                #         )
                #         features = DAF(*img_feats, points_2d, attention_weights).reshape(
                #             bs, -1, self.embed_dims
                #         )   
                #         features = self.proj_drop[self.temporal_layers+i](self.out_proj[self.temporal_layers+i](features)) if self.use_temporal else self.proj_drop[i](self.out_proj[i](features))
                #         if self.use_occ_loss == False:
                #             occupied_scores = torch.gather(occupancy_scores,1,occupied_indices)
                #             features = features * occupied_scores
                #         occupied_query = occupied_query + features
                #         query = torch.scatter(query,1,occupied_indices.repeat(1,1,query.shape[2]),occupied_query)
                #     query = query.view(bs,H,W,D,C).permute(0,4,3,1,2) # (bs, H*W*D, C) -> (bs, C, D, H, W)
        # query = self.fin_conv_net(query)[0]
        
            if self.only_background is not True:
                if self.pred_occ:
                    if self.up_sample:
                        up_vox_occ = self.up_block(query)
                        vox_occ = self.vox_occ_net(up_vox_occ)
                    else:
                        vox_occ = self.vox_occ_net(query)
                    vox_occ_list.append(vox_occ.permute(0,1,4,3,2)) # [B, C ,W, H, D]
        return query, occ_list, vox_occ_list # 
            
        # if self.use_occ_loss:
        #     occ = self.occ_conv[i](prev_vox_feat)
        #     occ = F.sigmoid(occ)
        #     prev_vox_feat = prev_vox_feat * occ

        # past_embed = self.past_encoder([past_feature,past_pos]).unsqueeze(1)
        # curr_embed = self.curr_encoder([curr_feature,curr_pos]).unsqueeze(1)
        # embeded_feature = torch.cat([past_embed,curr_embed],dim=1)
        # embeded_feature = embeded_feature + self.temp_embed
        # embeded_feature = embeded_feature.flatten(1)
        # next_pos = self.predictor(embeded_feature)
        
        # return next_pos
    
    
    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        bs, num_anchor, num_pts = key_points.shape[:3]
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )
        points_2d = torch.matmul(projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d
    
    
    
    def _get_weights(self, instance_feature, anchor_embed, metas=None, number_of_temp=0):
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
            self.attention_weights[number_of_temp](feature).reshape(bs, num_anchor, -1, self.num_groups).softmax(dim=-2).reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_points,
                self.num_groups,
            )
        )
        if self.training and self.attn_drop > 0:
            mask = torch.rand(
                bs, num_anchor, self.num_cams, 1, self.num_points, 1
            )
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (
                1 - self.attn_drop
            )
        return weights