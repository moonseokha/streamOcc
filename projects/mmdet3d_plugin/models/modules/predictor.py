import torch
import torch.nn.functional as F
import torch.nn as nn
from .embedding import MultipleInputEmbedding,SingleInputEmbedding
from .utils import init_weights
# from mmdet.models import HEADS
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
import pdb

__all__ = ["Predictor"]


@PLUGIN_LAYERS.register_module()
class Predictor(nn.Module):

    def __init__(self,
                 embed_dims: int,
                 ) -> None:

        super(Predictor, self).__init__()
        self.embed_dims = embed_dims
        self.past_encoder = MultipleInputEmbedding(in_channels=[embed_dims,8],out_channel=embed_dims)
        self.curr_encoder = MultipleInputEmbedding(in_channels=[embed_dims,8],out_channel=embed_dims)
        self.predictor = SingleInputEmbedding(in_channel=embed_dims*2,middle_channel=embed_dims,out_channel=5)
        self.temp_embed = nn.Parameter(torch.Tensor(1, 2, embed_dims))
        nn.init.normal_(self.temp_embed, mean=0., std=.02)
    

        self.apply(init_weights)
        
        
    def forward(self,
                past_pos: torch.Tensor,
                past_feature: torch.Tensor,
                curr_pos: torch.Tensor,
                curr_feature: torch.Tensor,
                ) -> torch.Tensor:
        # print("past_pos.shape: ",past_pos.shape)
        # print("past_feature.shape: ",past_feature.shape)
        # print("curr_pos.shape: ",curr_pos.shape)
        # print("curr_feature.shape: ",curr_feature.shape)
        # pdb.set_trace()
        past_embed = self.past_encoder([past_feature,past_pos]).unsqueeze(1)
        curr_embed = self.curr_encoder([curr_feature,curr_pos]).unsqueeze(1)
        embeded_feature = torch.cat([past_embed,curr_embed],dim=1)
        embeded_feature = embeded_feature + self.temp_embed
        embeded_feature = embeded_feature.flatten(1)
        next_pos = self.predictor(embeded_feature)
        
        return next_pos