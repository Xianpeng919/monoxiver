"""modified from https://github.com/whai362/PVT/tree/v2

New features:
    1) Adding Attentive Normlization (AttnBN2d and AttnLN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import math

from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger

from mmcv.runner import load_checkpoint

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from .layers import get_norm_layer, AttnLayerNorm, AttnBatchNorm2d


__all__ = ['PyramidVisionTransformerV2', 'pvt_v2_b0', 'pvt_v2_b1', 'pvt_v2_b2',
           'pvt_v2_b3', 'pvt_v2_b4', 'pvt_v2_b5', 'pvt_v2_b2_li',
           'pvt_v2_b0_cifar', 'pvt_v2_b1_cifar', 'pvt_v2_b2_cifar',
           'pvt_v2_b3_cifar', 'pvt_v2_b4_cifar', 'pvt_v2_b5_cifar',
           'pvt_v2_b2_li_cifar',
           'pvt_v2_det_b0', 'pvt_v2_det_b1', 'pvt_v2_det_b2', 'pvt_v2_det_b3',
           'pvt_v2_det_b4', 'pvt_v2_det_b5', 'pvt_v2_det_b2_li'
           ]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .875,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.,
                 linear=False,
                 norm_layer=None,
                 norm_cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.norm = get_norm_layer(hidden_features, norm_layer, norm_cfg)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (AttnLayerNorm, AttnBatchNorm2d)):
            nn.init.normal_(m.weight_, 1., 0.1)
            nn.init.normal_(m.bias_, 0., 0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        x = self.dwconv(x, H, W)
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=1,
                 linear=False,
                 linear_pool_size=7,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=None):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(
                    dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = get_norm_layer(dim, norm_layer, norm_cfg)
        else:
            self.pool = nn.AdaptiveAvgPool2d(linear_pool_size)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = get_norm_layer(dim, norm_layer, norm_cfg)
            self.act = nn.GELU()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (AttnLayerNorm, AttnBatchNorm2d)):
            nn.init.normal_(m.weight_, 1., 0.1)
            nn.init.normal_(m.bias_, 0., 0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads,
                              C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                         C // self.num_heads).permute(
                                             2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads,
                                        C // self.num_heads).permute(
                                            2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads,
                                     C // self.num_heads).permute(
                                         2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop=0.,
        attn_drop=0.,
        drop_path=0.,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        sr_ratio=1,
        linear=False,
        linear_pool_size=7,
        norm_cfg=None,
        with_norm_in_ffn=False,
        with_attn_norm_before_MHSA=True,
        with_attn_norm_in_attention=True,
        with_attn_norm_before_FFN=True,
        with_attn_norm_in_FFN=True,
    ):
        super().__init__()
        self.norm1 = get_norm_layer(
            dim, norm_layer, norm_cfg if with_attn_norm_before_MHSA else None)

        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            sr_ratio=sr_ratio,
            linear=linear,
            linear_pool_size=linear_pool_size,
            norm_layer=norm_layer,
            norm_cfg=norm_cfg if with_attn_norm_in_attention else None)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = get_norm_layer(
            dim, norm_layer, norm_cfg if with_attn_norm_before_FFN else None)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            linear=linear,
            norm_layer=norm_layer if with_norm_in_ffn else None,
            norm_cfg=norm_cfg if with_attn_norm_in_FFN else None)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (AttnLayerNorm, AttnBatchNorm2d)):
            nn.init.normal_(m.weight_, 1., 0.1)
            nn.init.normal_(m.bias_, 0., 0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 embed_dim=768,
                 with_norm=True,
                 norm_layer=nn.LayerNorm,
                 norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = get_norm_layer(embed_dim, norm_layer,
                                   norm_cfg) if with_norm else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (AttnLayerNorm, AttnBatchNorm2d)):
            nn.init.normal_(m.weight_, 1., 0.1)
            nn.init.normal_(m.bias_, 0., 0.1)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class PyramidVisionTransformerV2(nn.Module):

    def __init__(self,
                 patch_size=7,
                 stride=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8],
                 mlp_ratios=[4, 4, 4, 4],
                 downsample=[True, True, True, True],
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 norm_cfg=None,
                 norm_num_affine_trans=None,
                 depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1],
                 num_stages=4,
                 linear=False,
                 linear_pool_size=7,
                 with_norm_in_patch_embed=True,
                 with_norm_in_ffn=False,
                 with_attn_norm_in_patch_embed=False,
                 with_attn_norm_before_MHSA=False,
                 with_attn_norm_in_attention=False,
                 with_attn_norm_before_FFN=False,
                 with_attn_norm_in_FFN=False,
                 with_attn_norm_after_a_stage=False,
                 with_attn_norm_depth=0,
                 pretrained=None):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            norm_cfg_ = None
            if norm_cfg is not None:
                assert len(norm_num_affine_trans) == num_stages
                norm_cfg_ = norm_cfg.copy()
                norm_cfg_['num_affine_trans'] = norm_num_affine_trans[i]

            stride_ = stride if i == 0 else 2 if downsample[i] else 1

            patch_embed = OverlapPatchEmbed(
                patch_size=patch_size if i == 0 else 3,
                stride=stride_,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i],
                with_norm=with_norm_in_patch_embed,
                norm_layer=norm_layer,
                norm_cfg=norm_cfg_ if with_attn_norm_in_patch_embed else None)

            block = nn.ModuleList([
                Block(
                    dim=embed_dims[i],
                    num_heads=num_heads[i],
                    mlp_ratio=mlp_ratios[i],
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[cur + j],
                    norm_layer=norm_layer,
                    sr_ratio=sr_ratios[i],
                    linear=linear,
                    linear_pool_size=linear_pool_size,
                    norm_cfg=norm_cfg_,
                    with_norm_in_ffn=with_norm_in_ffn,
                    with_attn_norm_before_MHSA=with_attn_norm_before_MHSA
                    and (with_attn_norm_depth == 0 or j < with_attn_norm_depth
                         or j >= depths[i] + with_attn_norm_depth),
                    with_attn_norm_in_attention=with_attn_norm_in_attention
                    and (with_attn_norm_depth == 0 or j < with_attn_norm_depth
                         or j >= depths[i] + with_attn_norm_depth),
                    with_attn_norm_before_FFN=with_attn_norm_before_FFN
                    and (with_attn_norm_depth == 0 or j < with_attn_norm_depth
                         or j >= depths[i] + with_attn_norm_depth),
                    with_attn_norm_in_FFN=with_attn_norm_in_FFN
                    and (with_attn_norm_depth == 0 or j < with_attn_norm_depth
                         or j >= depths[i] + with_attn_norm_depth),
                ) for j in range(depths[i])
            ])
            norm = get_norm_layer(
                embed_dims[i], norm_layer,
                norm_cfg_ if with_attn_norm_after_a_stage else None)
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        # classification head
        self.head = nn.Linear(embed_dims[3],
                              num_classes) if num_classes > 0 else None

        self.apply(self._init_weights)
        self.init_weights(pretrained)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (AttnLayerNorm, AttnBatchNorm2d)):
            nn.init.normal_(m.weight_, 1., 0.1)
            nn.init.normal_(m.bias_, 0., 0.1)
            # nn.init.constant_(m.weight_, 1.0)
            # nn.init.constant_(m.bias_, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(
                self,
                pretrained,
                map_location='cpu',
                strict=False,
                logger=logger)

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'
        }  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(
            self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x, H, W)
            x = norm(x)
            if i != self.num_stages - 1 or self.head is None:
                x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                outs.append(x)

        if self.head is None:
            return outs

        return x.mean(dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        if self.head is not None:
            x = self.head(x)

        return x

    def forward_dummy(self, x):
        x = self.forward_features(x)
        if self.head is not None:
            x = self.head(x)

        return x


class DWConv(nn.Module):

    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


_model_settings = dict(
    b0=dict(
        embed_dims=[32, 64, 160, 256],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 2, 2],
    ),
    b1=dict(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 2, 2],
    ),
    b2=dict(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 6, 3],
    ),
    b3=dict(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 4, 18, 3],
    ),
    b4=dict(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[8, 8, 4, 4],
        depths=[3, 8, 27, 3],
    ),
    b5=dict(
        embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        depths=[3, 6, 40, 3],
    ),
)

# ImageNet

@register_model
def pvt_v2_b0(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(
        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b0'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b1(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b1'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b2(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b2'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b3(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b3'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b4(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b4'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b5(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b5'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

@register_model
def pvt_v2_b2_li(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=7,
        stride=4,
        in_chans=3,
        # num_classes=1000,
        **_model_settings['b2'],
        downsample=[True, True, True, True],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=True,
        linear_pool_size=7,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()

    return model

# cifar

@register_model
def pvt_v2_b0_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(
        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b0'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model

@register_model
def pvt_v2_b1_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b1'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model

@register_model
def pvt_v2_b2_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b2'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b0_sr8_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(
        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b0'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=4,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b1_sr8_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b1'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=4,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b2_sr8_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b2'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[8, 4, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=4,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b0_pure_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(
        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b0'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[1, 1, 1, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b1_pure_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b1'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[1, 1, 1, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b2_pure_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b2'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[1, 1, 1, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


@register_model
def pvt_v2_b3_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b3'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model

@register_model
def pvt_v2_b4_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b4'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model

@register_model
def pvt_v2_b5_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b5'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=False,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model

@register_model
def pvt_v2_b2_li_cifar(pretrained=False, **kwargs):

    model = PyramidVisionTransformerV2(

        patch_size=3,
        stride=1,
        in_chans=3,
        # num_classes=10,
        **_model_settings['b2'],
        downsample=[True, True, True, False],
        qkv_bias=True,
        qk_scale=None,
        # drop_rate=0.,
        attn_drop_rate=0.,
        # drop_path_rate=0.,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        norm_cfg=None,
        norm_num_affine_trans=None,
        sr_ratios=[4, 2, 2, 1],
        num_stages=4,
        linear=True,
        linear_pool_size=8,
        with_norm_in_patch_embed=True,
        with_norm_in_ffn=False,
        with_attn_norm_in_patch_embed=False,
        with_attn_norm_before_MHSA=False,
        with_attn_norm_in_attention=False,
        with_attn_norm_before_FFN=False,
        with_attn_norm_in_FFN=False,
        with_attn_norm_after_a_stage=False,
        with_attn_norm_depth=0,
        # pretrained=None,
        **kwargs)

    model.default_cfg = _cfg()
    model.default_cfg['crop_pct'] = 1.0

    return model


# Downstream tasks with ImageNet pretrained models

@BACKBONES.register_module()
class pvt_v2_det_b0(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):

        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b0'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b1(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b1'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b2(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b2'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b2_li(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b0'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=True,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b3(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b3'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b4(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b4'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes


@BACKBONES.register_module()
class pvt_v2_det_b5(PyramidVisionTransformerV2):

    def __init__(self, **kwargs):
        pretrained = kwargs.pop('pretrained', None)
        super().__init__(
            patch_size=7,
            stride=4,
            in_chans=3,
            num_classes=0,
            **_model_settings['b5'],
            downsample=[True, True, True, True],
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_cfg=None,
            norm_num_affine_trans=None,
            sr_ratios=[8, 4, 2, 1],
            num_stages=4,
            linear=False,
            linear_pool_size=7,
            with_norm_in_patch_embed=True,
            with_norm_in_ffn=False,
            with_attn_norm_in_patch_embed=False,
            with_attn_norm_before_MHSA=False,
            with_attn_norm_in_attention=False,
            with_attn_norm_before_FFN=False,
            with_attn_norm_in_FFN=False,
            with_attn_norm_after_a_stage=False,
            with_attn_norm_depth=0,
            pretrained=pretrained,)

        del self.num_classes
