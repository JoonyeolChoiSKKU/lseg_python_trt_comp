import torch
import torch.nn as nn
import torch.nn.functional as F
from .lseg_vit import get_activation, get_attention, forward_vit, get_readout_oper, Transpose

def _make_encoder(backbone, features, use_pretrained=True, groups=1, expand=False, exportable=True, hooks=None, use_vit_only=False, use_readout="ignore", enable_attention_hooks=False):
    if backbone == "clip_vitl16_384": 
        import clip
        clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        import timm
        model = timm.create_model("vit_large_patch16_384", pretrained=use_pretrained)
        hooks = hooks if hooks is not None else [5, 11, 17, 23]
        pretrained = _make_vit_b16_backbone(model, features=[256, 512, 1024, 1024], size=[384,384], hooks=hooks, vit_features=1024, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)
    elif backbone == "clipRN50x16_vitl16_384":
        import clip
        clip_pretrained, _ = clip.load("RN50x16", device='cuda', jit=False)
        import timm
        model = timm.create_model("vit_large_patch16_384", pretrained=use_pretrained)
        hooks = hooks if hooks is not None else [5, 11, 17, 23]
        pretrained = _make_vit_b16_backbone(model, features=[256, 512, 1024, 1024], size=[384,384], hooks=hooks, vit_features=1024, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([256, 512, 1024, 1024], features, groups=groups, expand=expand)
    elif backbone == "clip_vitb32_384":
        import clip
        clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False)
        import timm
        model = timm.create_model("vit_base_patch32_384", pretrained=use_pretrained)
        hooks = hooks if hooks is not None else [2, 5, 8, 11]
        pretrained = _make_vit_b32_backbone(model, features=[96, 192, 384, 768], hooks=hooks, use_readout=use_readout, enable_attention_hooks=enable_attention_hooks)
        scratch = _make_scratch([96, 192, 384, 768], features, groups=groups, expand=expand)
    else:
        raise NotImplementedError(f"Backbone '{backbone}' not implemented")
    return clip_pretrained, pretrained, scratch

def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups)
    return scratch

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners=False):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super(FeatureFusionBlock, self).__init__()
        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)
    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])
        output = self.resConfUnit2(output)
        output = nn.functional.interpolate(output, scale_factor=2, mode="bilinear", align_corners=True)
        return output

class ResidualConvUnit_custom(nn.Module):
    def __init__(self, features, activation, bn):
        super().__init__()
        self.bn = bn
        self.groups = 1
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=not self.bn, groups=self.groups)
        if self.bn:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)
        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn:
            out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn:
            out = self.bn2(out)
        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        super(FeatureFusionBlock_custom, self).__init__()
        self.deconv = deconv
        self.align_corners = align_corners
        self.expand = expand
        out_features = features // 2 if expand else features
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True)
        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        self.skip_add = nn.quantized.FloatFunctional()
    def forward(self, *xs):
        output = xs[0]
        if len(xs) == 2:
            output = self.skip_add.add(output, self.resConfUnit1(xs[1]))
        output = self.resConfUnit2(output)
        output = F.interpolate(output, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def _make_vit_b16_backbone(model, features=[256,512,1024,1024], size=[384,384], hooks=[2,5,8,11], vit_features=1024, use_readout="ignore", start_index=1, enable_attention_hooks=False):
    import types
    from .lseg_vit import get_readout_oper, Transpose, forward_flex, _resize_pos_embed
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.activations = {}
    pretrained.model.blocks[hooks[0]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("1", o))
    pretrained.model.blocks[hooks[1]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("2", o))
    pretrained.model.blocks[hooks[2]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("3", o))
    pretrained.model.blocks[hooks[3]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("4", o))
    readout_oper = get_readout_oper(vit_features, features, use_readout, start_index)
    patch_size = getattr(pretrained.model, "patch_size", None)
    if patch_size is None:
        patch_size = pretrained.model.patch_embed.patch_size
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // patch_size[1], size[1] // patch_size[0]])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[0], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True)
    )
    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // patch_size[1], size[1] // patch_size[0]])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[1], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True)
    )
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // patch_size[1], size[1] // patch_size[0]])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[2], kernel_size=1, stride=1, padding=0)
    )
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([size[0] // patch_size[1], size[1] // patch_size[0]])),
        nn.Conv2d(in_channels=vit_features, out_channels=features[3], kernel_size=1, stride=1, padding=0),
        nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1)
    )
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [16, 16]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained

def _make_vit_b32_backbone(model, features=[96,192,384,768], hooks=[2,5,8,11], use_readout="ignore", start_index=1, enable_attention_hooks=False):
    import types
    from .lseg_vit import get_readout_oper, Transpose, forward_flex, _resize_pos_embed
    pretrained = nn.Module()
    pretrained.model = model
    pretrained.activations = {}
    pretrained.model.blocks[hooks[0]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("1", o))
    pretrained.model.blocks[hooks[1]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("2", o))
    pretrained.model.blocks[hooks[2]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("3", o))
    pretrained.model.blocks[hooks[3]].register_forward_hook(lambda m, i, o: pretrained.activations.__setitem__("4", o))
    readout_oper = get_readout_oper(768, features, use_readout, start_index)
    patch_size = getattr(pretrained.model, "patch_size", None)
    if patch_size is None:
        patch_size = pretrained.model.patch_embed.patch_size
    pretrained.act_postprocess1 = nn.Sequential(
        readout_oper[0],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([32 // patch_size[1], 32 // patch_size[0]])),
        nn.Conv2d(in_channels=768, out_channels=features[0], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[0], out_channels=features[0], kernel_size=4, stride=4, padding=0, bias=True)
    )
    pretrained.act_postprocess2 = nn.Sequential(
        readout_oper[1],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([32 // patch_size[1], 32 // patch_size[0]])),
        nn.Conv2d(in_channels=768, out_channels=features[1], kernel_size=1, stride=1, padding=0),
        nn.ConvTranspose2d(in_channels=features[1], out_channels=features[1], kernel_size=2, stride=2, padding=0, bias=True)
    )
    pretrained.act_postprocess3 = nn.Sequential(
        readout_oper[2],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([32 // patch_size[1], 32 // patch_size[0]])),
        nn.Conv2d(in_channels=768, out_channels=features[2], kernel_size=1, stride=1, padding=0)
    )
    pretrained.act_postprocess4 = nn.Sequential(
        readout_oper[3],
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size([32 // patch_size[1], 32 // patch_size[0]])),
        nn.Conv2d(in_channels=768, out_channels=features[3], kernel_size=1, stride=1, padding=0),
        nn.Conv2d(in_channels=features[3], out_channels=features[3], kernel_size=3, stride=2, padding=1)
    )
    pretrained.model.start_index = start_index
    pretrained.model.patch_size = [32, 32]
    pretrained.model.forward_flex = types.MethodType(forward_flex, pretrained.model)
    pretrained.model._resize_pos_embed = types.MethodType(_resize_pos_embed, pretrained.model)
    return pretrained
