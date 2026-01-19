# This work is licensed under the Creative Commons Attribution-Non Commercial ShareAlike 4.0 International License. 
# To view a copy of this license, visit Legal Code - https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.en

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, get_norm
from detectron2.modeling.backbone import build_backbone as d2_build_backbone
import fvcore.nn.weight_init as weight_init
from models.config import CfgNode


def build_backbone2d(cfg, output_dim):
    """ Builds 2D feature extractor backbone network from Detectron2."""
    
    norm = 'nnSyncBN'
    output_stride = 4 
    
    backbone = d2_build_backbone(cfg)
    feature_extractor = FPNFeature(
        backbone.output_shape(), output_dim, output_stride, norm)

    if cfg.MODEL.BACKBONE.WEIGHTS:
        state_dict = torch.load(cfg.MODEL.BACKBONE.WEIGHTS)
        backbone.load_state_dict(state_dict, strict=False)
        print ('encoder weights loaded successfully')

    return nn.Sequential(backbone, feature_extractor), output_stride


class FPNFeature(nn.Module):
    """ Converts feature pyrimid to singe feature map (from Detectron2)"""
    
    def __init__(self, input_shape, output_dim=32, output_stride=4, norm='BN'):
        super().__init__()

        self.in_features      = ["p2", "p3", "p4", "p5"]
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(output_stride))
            )
            for k in range(head_length):
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else output_dim,
                    output_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=get_norm(norm, output_dim),
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != output_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])

    def forward(self, features):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        return x


def load_encoder(output_dim):
    cfg = {'MODEL': {'PIXEL_MEAN': [103.53, 116.28, 123.675], 'PIXEL_STD': [1.0, 1.0, 1.0], 'BACKBONE': {'NAME': 'build_resnet_fpn_backbone', 'FREEZE_AT': 2, 'WEIGHTS': 'weights/R-50.pth'}, 'RESNETS': {'DEPTH': 50, 'OUT_FEATURES': ['res2', 'res3', 'res4', 'res5'], 'NUM_GROUPS': 1, 'NORM': 'nnSyncBN', 'WIDTH_PER_GROUP': 64, 'STRIDE_IN_1X1': True, 'RES5_DILATION': 1, 'RES2_OUT_CHANNELS': 256, 'STEM_OUT_CHANNELS': 64, 'DEFORM_ON_PER_STAGE': [False, False, False, False], 'DEFORM_MODULATED': False, 'DEFORM_NUM_GROUPS': 1}, 'FPN': {'IN_FEATURES': ['res2', 'res3', 'res4', 'res5'], 'OUT_CHANNELS': 256, 'NORM': 'nnSyncBN', 'FUSE_TYPE': 'sum'}}}
    return build_backbone2d(CfgNode(cfg), output_dim = output_dim)