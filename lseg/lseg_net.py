import math
import types
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import numpy as np

from .lseg_blocks import _make_fusion_block, _make_encoder  # 경량화된 lseg_blocks.py 사용
from .lseg_vit import forward_vit  # 경량화된 lseg_vit.py 사용

# BaseModel: 단순 load 메서드만 포함
class BaseModel(nn.Module):
    def load(self, path):
        parameters = torch.load(path, map_location=torch.device("cpu"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)

# BottleneckBlock 및 DepthwiseBlock: head_block 옵션에 사용 (원본 그대로)
class BottleneckBlock(nn.Module):
    def __init__(self, activation='lrelu'):
        super(BottleneckBlock, self).__init__()
        self.depthwise = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    def forward(self, x, activate=True):
        residual = x.max(dim=1, keepdim=True)[0]
        out = self.depthwise(x)
        out = out + residual
        if activate:
            out = self.activation(out)
        return out

class DepthwiseBlock(nn.Module):
    def __init__(self, activation='lrelu'):
        super(DepthwiseBlock, self).__init__()
        self.depthwise = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
    def forward(self, x, activate=True):
        out = self.depthwise(x)
        if activate:
            out = self.activation(out)
        return out

# LSegNet: 이미지 인코딩 경로만 남김 (텍스트 인코딩 및 segmentation head는 제거)
class LSegNet(BaseModel):
    def __init__(self, labels, features=256, backbone="clip_vitl16_384", crop_size=480, 
                 arch_option=0, block_depth=0, activation='lrelu', use_bn=True):
        super(LSegNet, self).__init__()
        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }
        # _make_encoder: 백본, clip_pretrained, scratch 모듈을 생성 (경량화된 버전 사용)
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone, features, groups=1, expand=False, exportable=False,
            hooks=hooks[backbone], use_readout="project"
        )
        if hasattr(self.pretrained, "patch_embed"):
            self.pretrained.patch_embed.img_size = (crop_size, crop_size)
        # RefineNet-style fusion block 구성
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)
        # logit_scale는 텍스트 연산에 사용되나, 이미지 인코더 추출 시에는 사용하지 않음
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        # Projection layer: head1
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)
        self.arch_option = arch_option
        self.block_depth = block_depth
        if self.arch_option == 1:
            self.scratch.head_block = BottleneckBlock(activation)
        elif self.arch_option == 2:
            self.scratch.head_block = DepthwiseBlock(activation)
        else:
            self.scratch.head_block = None
        # segmentation head 제거 → Identity
        self.scratch.output_conv = nn.Identity()
        self.labels = labels

    def forward(self, x):
        # 백본의 중간 feature 추출 (lseg_vit.py의 forward_vit 사용)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.out_c).permute(0, 3, 1, 2)
        if self.arch_option in [1, 2] and self.block_depth > 0:
            for _ in range(self.block_depth - 1):
                image_features = self.scratch.head_block(image_features)
            image_features = self.scratch.head_block(image_features, activate=False)
        image_features = self.scratch.output_conv(image_features)
        return image_features
