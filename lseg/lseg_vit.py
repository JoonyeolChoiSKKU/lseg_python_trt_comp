import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import clip

# 원본에서는 전역 activations를 사용하지만, ONNX export를 위해
# _make_vit_b16_backbone에서 pretrained.activations라는 객체 속성에 저장하도록 했으므로,
# 여기서는 그 속성을 사용합니다.
def get_activation(name):
    def hook(model, input, output):
        # 전역 변수가 아니라, 모델 객체의 activations 속성에 저장
        model.activations[name] = output
    return hook

def get_attention(name):
    def hook(module, input, output):
        x = input[0]
        B, N, C = x.shape
        qkv = (
            module.qkv(x)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        return attn
    return hook

def get_readout_oper(vit_features, features, use_readout, start_index=1):
    if use_readout == "ignore":
        readout_oper = [Slice(start_index)] * len(features)
    elif use_readout == "add":
        readout_oper = [AddReadout(start_index)] * len(features)
    elif use_readout == "project":
        readout_oper = [ProjectReadout(vit_features, start_index) for _ in features]
    else:
        assert False, "wrong operation for readout token, use_readout can be 'ignore', 'add', or 'project'"
    return readout_oper

class Slice(nn.Module):
    def __init__(self, start_index=1):
        super(Slice, self).__init__()
        self.start_index = start_index
    def forward(self, x):
        return x[:, self.start_index:]

class AddReadout(nn.Module):
    def __init__(self, start_index=1):
        super(AddReadout, self).__init__()
        self.start_index = start_index
    def forward(self, x):
        if self.start_index == 2:
            readout = (x[:, 0] + x[:, 1]) / 2
        else:
            readout = x[:, 0]
        return x[:, self.start_index:] + readout.unsqueeze(1)

class ProjectReadout(nn.Module):
    def __init__(self, in_features, start_index=1):
        super(ProjectReadout, self).__init__()
        self.start_index = start_index
        self.project = nn.Sequential(nn.Linear(2 * in_features, in_features), nn.GELU())
    def forward(self, x):
        readout = x[:, 0].unsqueeze(1).expand_as(x[:, self.start_index:])
        features = torch.cat((x[:, self.start_index:], readout), -1)
        return self.project(features)

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

def forward_vit(pretrained, x):
    b, c, h, w = x.shape
    _ = pretrained.model.forward_flex(x)
    
    layer_1 = pretrained.activations["1"]
    layer_2 = pretrained.activations["2"]
    layer_3 = pretrained.activations["3"]
    layer_4 = pretrained.activations["4"]
    
    layer_1 = pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = pretrained.act_postprocess4[0:2](layer_4)
    
    patch_size = getattr(pretrained.model, "patch_size", None)
    if patch_size is None:
        patch_size = pretrained.model.patch_embed.patch_size

    unflatten = nn.Unflatten(2, (int(h // patch_size[1]), int(w // patch_size[0])))
    
    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)
    
    layer_1 = pretrained.act_postprocess1[3:](layer_1)
    layer_2 = pretrained.act_postprocess2[3:](layer_2)
    layer_3 = pretrained.act_postprocess3[3:](layer_3)
    layer_4 = pretrained.act_postprocess4[3:](layer_4)
    
    return layer_1, layer_2, layer_3, layer_4


def _resize_pos_embed(self, posemb, gs_h, gs_w):
    posemb_tok, posemb_grid = posemb[:, : self.start_index], posemb[0, self.start_index:]
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_h, gs_w), mode="bilinear", align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_h * gs_w, -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb

def forward_flex(self, x):
    b, c, h, w = x.shape
    patch_size = getattr(self, "patch_size", None)
    if patch_size is None:
        patch_size = self.patch_embed.patch_size
    pos_embed = self._resize_pos_embed(self.pos_embed, h // patch_size[1], w // patch_size[0])
    B = x.shape[0]
    if hasattr(self.patch_embed, "backbone"):
        x = self.patch_embed.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]
    x = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
    if getattr(self, "dist_token", None) is not None:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
    else:
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
    x = x + pos_embed
    x = self.pos_drop(x)
    for blk in self.blocks:
        x = blk(x)
    x = self.norm(x)
    return x
