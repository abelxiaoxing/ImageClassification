# --------------------------------------------------------
# TinyViT Model Architecture
# Copyright (c) 2022 Microsoft
# Adapted from LeViT and Swin Transformer
#   LeViT: (https://github.com/facebookresearch/levit)
#   Swin: (https://github.com/microsoft/swin-transformer)
# Build the TinyViT Model
# --------------------------------------------------------

import itertools
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Dict
from timm.models.layers import DropPath as TimmDropPath
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from functools import partial
from timm.layers import LayerNorm2d, NormMlpClassifierHead, DropPath,trunc_normal_, resize_rel_pos_bias_table_levit, use_fused_attn

try:
    # timm.__version__ >= "0.6"
    from timm.models._builder import build_model_with_cfg
except (ImportError, ModuleNotFoundError):
    # timm.__version__ < "0.6"
    from timm.models.helpers import build_model_with_cfg

class ConvNorm(torch.nn.Sequential):
    def __init__(self, in_chs, out_chs, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, ks, stride, pad, dilation, groups, bias=False)
        self.bn = nn.BatchNorm2d(out_chs)
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self.conv, self.bn
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.conv.groups, w.size(0), w.shape[2:],
            stride=self.conv.stride, padding=self.conv.padding, dilation=self.conv.dilation, groups=self.conv.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Conv2d_BN(torch.nn.Sequential):
    def __init__(
        self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1
    ):
        super().__init__()
        self.add_module(
            "c", torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False)
        )
        bn = torch.nn.BatchNorm2d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module("bn", bn)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(
            w.size(1) * self.c.groups,
            w.size(0),
            w.shape[2:],
            stride=self.c.stride,
            padding=self.c.padding,
            dilation=self.c.dilation,
            groups=self.c.groups,
        )
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


class DropPath(TimmDropPath):
    def __init__(self, drop_prob=None):
        super().__init__(drop_prob=drop_prob)
        self.drop_prob = drop_prob

    def __repr__(self):
        msg = super().__repr__()
        msg += f"(drop_prob={self.drop_prob})"
        return msg

class PatchEmbed(nn.Module):
    def __init__(self, in_chs, out_chs, act_layer):
        super().__init__()
        self.stride = 4
        self.conv1 = ConvNorm(in_chs, out_chs // 2, 3, 2, 1)
        self.act = act_layer()
        self.conv2 = ConvNorm(out_chs // 2, out_chs, 3, 2, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x

class MBConv(nn.Module):
    def __init__(self, in_chs, out_chs, expand_ratio, act_layer, drop_path):
        super().__init__()
        mid_chs = int(in_chs * expand_ratio)
        self.conv1 = ConvNorm(in_chs, mid_chs, ks=1)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(mid_chs, mid_chs, ks=3, stride=1, pad=1, groups=mid_chs)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(mid_chs, out_chs, ks=1, bn_weight_init=0.0)
        self.act3 = act_layer()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.drop_path(x)
        x += shortcut
        x = self.act3(x)
        return x

class PatchMerging(nn.Module):
    def __init__(self, dim, out_dim, act_layer):
        super().__init__()
        self.conv1 = ConvNorm(dim, out_dim, 1, 1, 0)
        self.act1 = act_layer()
        self.conv2 = ConvNorm(out_dim, out_dim, 3, 2, 1, groups=out_dim)
        self.act2 = act_layer()
        self.conv3 = ConvNorm(out_dim, out_dim, 1, 1, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        return x


class ConvLayer(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            act_layer,
            drop_path=0.,
            conv_expand_ratio=4.,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.blocks = nn.Sequential(*[
            MBConv(
                dim, dim, conv_expand_ratio, act_layer,
                drop_path[i] if isinstance(drop_path, list) else drop_path,
            )
            for i in range(depth)
        ])

    def forward(self, x):
        x = self.blocks(x)
        return x


class NormMlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            norm_layer=nn.LayerNorm,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = norm_layer(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x




class Attention(torch.nn.Module):
    fused_attn: torch.jit.Final[bool]
    attention_bias_cache: Dict[str, torch.Tensor]

    def __init__(
            self,
            dim,
            key_dim,
            num_heads=8,
            attn_ratio=4,
            resolution=(14, 14),
    ):
        super().__init__()
        assert isinstance(resolution, tuple) and len(resolution) == 2
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.val_dim = int(attn_ratio * key_dim)
        self.out_dim = self.val_dim * num_heads
        self.attn_ratio = attn_ratio
        self.resolution = resolution
        self.fused_attn = use_fused_attn()

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, num_heads * (self.val_dim + 2 * key_dim))
        self.proj = nn.Linear(self.out_dim, dim)

        points = list(itertools.product(range(resolution[0]), range(resolution[1])))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N), persistent=False)
        self.attention_bias_cache = {}

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and self.attention_bias_cache:
            self.attention_bias_cache = {}  # clear ab cache

    def get_attention_biases(self, device: torch.device) -> torch.Tensor:
        if torch.jit.is_tracing() or self.training:
            return self.attention_biases[:, self.attention_bias_idxs]
        else:
            device_key = str(device)
            if device_key not in self.attention_bias_cache:
                self.attention_bias_cache[device_key] = self.attention_biases[:, self.attention_bias_idxs]
            return self.attention_bias_cache[device_key]

    def forward(self, x):
        attn_bias = self.get_attention_biases(x.device)
        B, N, _ = x.shape
        # Normalization
        x = self.norm(x)
        qkv = self.qkv(x)
        # (B, N, num_heads, d)
        q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.val_dim], dim=3)
        # (B, num_heads, N, d)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn + attn_bias
            attn = attn.softmax(dim=-1)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, self.out_dim)
        x = self.proj(x)
        return x




class TinyVitBlock(nn.Module):
    """ TinyViT Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        local_conv_size (int): the kernel size of the convolution between
                               Attention and MLP. Default: 3
        act_layer: the activation function. Default: nn.GELU
    """

    def __init__(
            self,
            dim,
            num_heads,
            window_size=7,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            local_conv_size=3,
            act_layer=nn.GELU
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert window_size > 0, 'window_size must be greater than 0'
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        head_dim = dim // num_heads

        window_resolution = (window_size, window_size)
        self.attn = Attention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()


        self.mlp = NormMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        pad = local_conv_size // 2
        self.local_conv = ConvNorm(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

    def forward(self, x):
        B, H, W, C = x.shape
        L = H * W

        shortcut = x
        if H == self.window_size and W == self.window_size:
            x = x.reshape(B, L, C)
            x = self.attn(x)
            x = x.view(B, H, W, C)
        else:
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            padding = pad_b > 0 or pad_r > 0
            if padding:
                x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            # window partition
            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_size
            nW = pW // self.window_size
            x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_size * self.window_size, C
            )

            x = self.attn(x)

            # window reverse
            x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()
        x = shortcut + self.drop_path1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.local_conv(x)
        x = x.reshape(B, C, L).transpose(1, 2)

        x = x + self.drop_path2(self.mlp(x))
        return x.view(B, H, W, C)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, mlp_ratio={self.mlp_ratio}"



class TinyVitStage(nn.Module):
    """ A basic TinyViT layer for one stage.

    Args:
        dim (int): Number of input channels.
        out_dim: the output dimension of the layer
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        local_conv_size: the kernel size of the depthwise convolution between attention and MLP. Default: 3
        act_layer: the activation function. Default: nn.GELU
    """

    def __init__(
            self,
            dim,
            out_dim,
            depth,
            num_heads,
            window_size,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            downsample=None,
            local_conv_size=3,
            act_layer=nn.GELU,
    ):

        super().__init__()
        self.depth = depth
        self.out_dim =  out_dim

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                dim=dim,
                out_dim=out_dim,
                act_layer=act_layer,
            )
        else:
            self.downsample = nn.Identity()
            assert dim == out_dim

        # build blocks
        self.blocks = nn.Sequential(*[
            TinyVitBlock(
                dim=out_dim,
                num_heads=num_heads,
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                local_conv_size=local_conv_size,
                act_layer=act_layer,
            )
            for i in range(depth)])

    def forward(self, x):
        x = self.downsample(x)
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.blocks(x)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class TinyVit(nn.Module):
    def __init__(
            self,
            in_chans=3,
            num_classes=1000,
            global_pool='avg',
            embed_dims=(96, 192, 384, 768),
            depths=(2, 2, 6, 2),
            num_heads=(3, 6, 12, 24),
            window_sizes=(7, 7, 14, 7),
            mlp_ratio=4.,
            drop_rate=0.,
            drop_path_rate=0.1,
            use_checkpoint=False,
            mbconv_expand_ratio=4.0,
            local_conv_size=3,
            act_layer=nn.GELU,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = len(depths)
        self.mlp_ratio = mlp_ratio
        self.grad_checkpointing = use_checkpoint

        self.patch_embed = PatchEmbed(
            in_chs=in_chans,
            out_chs=embed_dims[0],
            act_layer=act_layer,
        )

        # stochastic depth rate rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # build stages
        self.stages = nn.Sequential()
        stride = self.patch_embed.stride
        prev_dim = embed_dims[0]
        self.feature_info = []
        for stage_idx in range(self.num_stages):
            if stage_idx == 0:
                stage = ConvLayer(
                    dim=prev_dim,
                    depth=depths[stage_idx],
                    act_layer=act_layer,
                    drop_path=dpr[:depths[stage_idx]],
                    conv_expand_ratio=mbconv_expand_ratio,
                )
            else:
                out_dim = embed_dims[stage_idx]
                drop_path_rate = dpr[sum(depths[:stage_idx]):sum(depths[:stage_idx + 1])]
                stage = TinyVitStage(
                    dim=embed_dims[stage_idx - 1],
                    out_dim=out_dim,
                    depth=depths[stage_idx],
                    num_heads=num_heads[stage_idx],
                    window_size=window_sizes[stage_idx],
                    mlp_ratio=self.mlp_ratio,
                    drop=drop_rate,
                    local_conv_size=local_conv_size,
                    drop_path=drop_path_rate,
                    downsample=PatchMerging,
                    act_layer=act_layer,
                )
                prev_dim = out_dim
                stride *= 2
            self.stages.append(stage)
            self.feature_info += [dict(num_chs=prev_dim, reduction=stride, module=f'stages.{stage_idx}')]

        # Classifier head
        self.num_features = embed_dims[-1]

        norm_layer_cf = partial(LayerNorm2d, eps=1e-5)
        self.head = NormMlpClassifierHead(
            self.num_features,
            num_classes,
            pool_type=global_pool,
            norm_layer=norm_layer_cf,
        )

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'attention_biases'}

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^patch_embed',
            blocks=r'^stages\.(\d+)' if coarse else [
                (r'^stages\.(\d+).downsample', (0,)),
                (r'^stages\.(\d+)\.\w+\.(\d+)', None),
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head.fc

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        self.head.reset(num_classes, pool_type=global_pool)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.stages, x)
        else:
            x = self.stages(x)
        return x

    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x



_checkpoint_url_format = (
    "https://github.com/wkcn/TinyViT-model-zoo/releases/download/checkpoints/{}.pth"
)


def _create_tiny_vit(variant, num_classes=1000, pretrained=False, **kwargs):
    pretrained_type = kwargs.pop("pretrained_type", "22kto1k_distill")
    assert pretrained_type in [
        "22kto1k_distill",
        "1k",
        "22k_distill",
    ], "pretrained_type should be one of 22kto1k_distill, 1k, 22k_distill"

    img_size = kwargs.get("img_size", 224)
    if img_size != 224:
        pretrained_type = pretrained_type.replace("_", f"_{img_size}_")

    num_classes_pretrained = 21841 if pretrained_type == "22k_distill" else num_classes

    variant_without_img_size = "_".join(variant.split("_")[:-1])
    cfg = dict(
        url=_checkpoint_url_format.format(
            f"{variant_without_img_size}_{pretrained_type}"
        ),
        num_classes=num_classes_pretrained,
        classifier="head",
    )

    def _pretrained_filter_fn(state_dict):
        state_dict = state_dict["model"]
        state_dict = {
            k: v for k, v in state_dict.items() if not k.endswith("attention_bias_idxs")
        }
        return state_dict

    if timm.__version__ >= "0.6":
        return build_model_with_cfg(
            TinyVit,
            variant,
            pretrained,
            pretrained_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs,
        )
    else:
        return build_model_with_cfg(
            TinyVit,
            variant,
            pretrained,
            default_cfg=cfg,
            pretrained_filter_fn=_pretrained_filter_fn,
            **kwargs,
        )


def tiny_vit_5m_224(num_classes=1000,pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.0,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit("tiny_vit_5m_224", num_classes, pretrained, **model_kwargs)


def tiny_vit_11m_224(num_classes=1000, pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[64, 128, 256, 448],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 8, 14],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit("tiny_vit_11m_224", num_classes, pretrained, **model_kwargs)


def tiny_vit_21m_224(num_classes=1000, pretrained=False, **kwargs):
    model_kwargs = dict(
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[7, 7, 14, 7],
        drop_path_rate=0.2,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit("tiny_vit_21m_224", num_classes, pretrained, **model_kwargs)


def tiny_vit_21m_384(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=384,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[12, 12, 24, 12],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit("tiny_vit_21m_384", pretrained, **model_kwargs)


def tiny_vit_21m_512(pretrained=False, **kwargs):
    model_kwargs = dict(
        img_size=512,
        embed_dims=[96, 192, 384, 576],
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 18],
        window_sizes=[16, 16, 32, 16],
        drop_path_rate=0.1,
    )
    model_kwargs.update(kwargs)
    return _create_tiny_vit("tiny_vit_21m_512", pretrained, **model_kwargs)
