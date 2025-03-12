import torch
import torch.nn as nn
from .lseg_vit import forward_vit

class LSegImageEncoder(nn.Module):
    def __init__(self, lseg_net):
        super(LSegImageEncoder, self).__init__()
        self.lseg_net = lseg_net

    def forward(self, x):
        # 백본을 통해 중간 feature들을 추출 (lseg_vit.py의 forward_vit 사용)
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.lseg_net.pretrained, x)
        
        # 각 feature에 대해 scratch 모듈의 RN 변환 적용
        layer_1_rn = self.lseg_net.scratch.layer1_rn(layer_1)
        layer_2_rn = self.lseg_net.scratch.layer2_rn(layer_2)
        layer_3_rn = self.lseg_net.scratch.layer3_rn(layer_3)
        layer_4_rn = self.lseg_net.scratch.layer4_rn(layer_4)
        
        # RefineNet 스타일의 fusion block 적용
        path_4 = self.lseg_net.scratch.refinenet4(layer_4_rn)
        path_3 = self.lseg_net.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.lseg_net.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.lseg_net.scratch.refinenet1(path_2, layer_1_rn)
        
        # Projection layer: CLIP 기반 head1
        image_features = self.lseg_net.scratch.head1(path_1)
        imshape = image_features.shape
        
        # 재배열 및 정규화: feature map을 (N, out_c, H, W) 형태로 변환
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.lseg_net.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.lseg_net.out_c).permute(0, 3, 1, 2)
        
        # arch_option이 1 또는 2이면 head_block 적용 (원본 그대로)
        if self.lseg_net.arch_option in [1, 2] and self.lseg_net.block_depth > 0:
            for _ in range(self.lseg_net.block_depth - 1):
                image_features = self.lseg_net.scratch.head_block(image_features)
            image_features = self.lseg_net.scratch.head_block(image_features, activate=False)
        
        # segmentation head (output_conv)는 제외합니다.
        return image_features