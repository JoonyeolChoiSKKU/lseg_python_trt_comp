import torch
from .lseg_net import LSegNet  # lseg_net.py는 원본 그대로 사용한다고 가정합니다.
import os
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class LSegModule:
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # ModelCheckpoint 글로벌을 허용합니다.
        torch.serialization.add_safe_globals([ModelCheckpoint])
        
        # 이미지 인코딩 추론에 필요한 최소 파라미터만 사용합니다.
        labels = ["dummy"]  # 이미지 인코딩에서는 텍스트 관련 처리가 없으므로 더미값 사용
        model = LSegNet(
            labels=labels,
            backbone=kwargs.get("backbone", "clip_vitl16_384"),
            features=kwargs.get("num_features", 256),
            crop_size=kwargs.get("crop_size", 480),
            arch_option=kwargs.get("arch_option", 0),
            block_depth=kwargs.get("block_depth", 0),
            activation=kwargs.get("activation", "lrelu")
        )
        state = torch.load(checkpoint_path, map_location="cpu")
        if "model" in state:
            state = state["model"]
        # 경량화된 구조와 불일치하는 파라미터는 무시합니다.
        model.load_state_dict(state, strict=False)
        return cls(model)
    
    def __init__(self, model):
        self.net = model
