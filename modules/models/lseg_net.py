import torch
import torch.nn as nn
import numpy as np
import clip

# lseg_blocks.py에서 필요한 함수 및 모듈 (Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit)
from .lseg_blocks import Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class BaseModel(nn.Module):
    def load(self, path):
        """Load model from file."""
        parameters = torch.load(path, map_location=torch.device("cuda:0"))
        if "optimizer" in parameters:
            parameters = parameters["model"]
        self.load_state_dict(parameters)

class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        use_bn=False,
        arch_option=0,
        block_depth=0,
        activation="lrelu",
        **kwargs,
    ):
        super(LSeg, self).__init__()

        # Define backbone hooks for feature extraction
        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate encoder (backbone) and obtain pretrained, clip_pretrained, and scratch modules
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        # Build fusion blocks for scratch module
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Set output channels based on backbone type
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512

        # Projection layer to map features to output channels
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        # Optional head block 관련 부분은 기본적으로 사용하지 않으므로(arch_option=0) 생략합니다.
        self.arch_option = arch_option
        self.block_depth = block_depth

        # Set segmentation head (여기서는 단순 upsampling으로 대체)
        self.scratch.output_conv = head

class LSegNet(LSeg):
    """
    LSegNet: Image Encoder 기능만 추출하기 위한 네트워크.
    ONNX 추출 시 forward_image_encoder() 함수만 사용합니다.
    """
    def __init__(self, labels, path=None, crop_size=480, **kwargs):
        features = kwargs.get("features", 256)
        kwargs["use_bn"] = True  # 항상 batch norm 사용
        self.crop_size = crop_size
        self.labels = labels  # 실제 라벨 리스트 (ONNX 추출에는 사용되지 않음)
        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True)
        )
        super(LSegNet, self).__init__(head, **kwargs)
        if path is not None:
            self.load(path)

    def forward(self, x):
        """
        입력 x에 대해 ViT 백본과 scratch 모듈을 통해 이미지 특징(feature map)을 추출합니다.
        이 함수만을 ONNX로 내보내서 Image Encoder로 활용할 수 있습니다.
        """
        # ViT 백본의 각 레이어 feature 추출
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)
        # Scratch 모듈을 통한 RN 변환 적용
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # RefineNet 스타일의 fusion block 적용 (깊은 feature부터 fusion 시작)
        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # Projection layer 적용
        image_features = self.scratch.head1(path_1)
        imshape = image_features.shape
        # 특징 재배열 및 정규화: (N, out_c, H, W) 형태로 변환
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.view(imshape[0], imshape[2], imshape[3], self.out_c).permute(0, 3, 1, 2)
        # (선택적) head_block 적용: arch_option이 1 또는 2이고 block_depth > 0일 경우에만 적용
        if self.arch_option in [1, 2] and self.block_depth > 0:
            for _ in range(self.block_depth - 1):
                image_features = self.scratch.head_block(image_features)
            image_features = self.scratch.head_block(image_features, activate=False)
        # 최종적으로 output_conv를 통해 최종 Feature Map 생성
        image_features = self.scratch.output_conv(image_features)
        return image_features
