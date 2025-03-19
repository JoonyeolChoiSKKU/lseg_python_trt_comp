import os
import torch

# ★ 필요에 따라 대체할 upsampling 파라미터를 정의하세요.
# 원래 encoding 라이브러리에서 사용한 up_kwargs의 내용을 참고하여 동일한 파라미터를 제공해야 합니다.
up_kwargs = {'mode': 'bilinear', 'align_corners': True}

from .models.lseg_net import LSegNet
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

class LSegModule:
    def __init__(self, crop_size=384,**kwargs):
        self.crop_size = crop_size
        # minimal label loading (필요한 경우, 혹은 고정된 라벨 리스트로 대체 가능)
        labels = self.get_labels('ade20k')
        self.net = LSegNet(
            labels=labels,
            backbone=kwargs["backbone"],
            features=kwargs["num_features"],
            crop_size=384,
            arch_option=kwargs["arch_option"],
            block_depth=kwargs["block_depth"],
            activation=kwargs["activation"],
            readout=kwargs["readout"],
        )
        # Patch embedding의 이미지 사이즈를 설정 (Image Encoder 추출에 필요)
        self.net.pretrained.model.patch_embed.img_size = (self.crop_size, self.crop_size)
        self._up_kwargs = up_kwargs

    def get_labels(self, dataset):
        labels = []
        path = 'data/{}_objectInfo150.txt'.format(dataset)
        if os.path.exists(path):
            with open(path, 'r') as f:
                lines = f.readlines()
            for line in lines:
                label = line.strip().split(',')[-1].split(';')[0]
                labels.append(label)
            if dataset in ['ade20k'] and labels:
                labels = labels[1:]
        return labels

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location, **kwargs):
        # 안전하게 checkpoint 내 ModelCheckpoint 글로벌을 허용
        torch.serialization.add_safe_globals([ModelCheckpoint])
        
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        # checkpoint 구조가 {"model": state_dict, ...} 또는 {"state_dict": ...}인 경우를 모두 처리합니다.
        if "model" in checkpoint:
            state = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state = checkpoint["state_dict"]
        else:
            state = checkpoint

        # 만약 checkpoint에 hyper_parameters가 있다면 이를 사용
        if "hyper_parameters" in checkpoint:
            instance = cls(**checkpoint["hyper_parameters"], map_location=map_location, **kwargs)
        else:
            instance = cls(**kwargs)

        # state_dict의 각 키에서 "net." 접두어를 제거합니다.
        new_state = {}
        for key, value in state.items():
            new_key = key
            if key.startswith("net."):
                new_key = key[len("net."):]
            new_state[new_key] = value

        # 모델에 state_dict를 로드합니다.
        instance.net.load_state_dict(new_state, strict=False)
        return instance

class LSegModuleWrapper:
    def __init__(self, model):
        self.net = model
