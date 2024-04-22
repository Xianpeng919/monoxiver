import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from functools import partial
import numbers

from timm.models.layers import trunc_normal_

from mmcv.cnn import (NORM_LAYERS, ACTIVATION_LAYERS, constant_init,
                      kaiming_init, build_activation_layer)


#### TODO:
# 1, use PaCa to learn the mixture of means/std

def get_norm_layer(num_features, norm_layer, norm_cfg):
    if norm_layer is None:
        return nn.Identity()

    if norm_cfg is None:
        return norm_layer(num_features)

    norm_cfg_ = norm_cfg.copy()
    t = norm_cfg_.pop('type')
    if t == 'AttnBN2d':
        return AttnBatchNorm2d(num_features, **norm_cfg_)
    elif t == 'AttnLN':
        return AttnLayerNorm(num_features, **norm_cfg_)
    else:
        raise NotImplementedError


class HSigmoidv2(nn.Module):
    """ (add ref)
    """

    def __init__(self, inplace=True):
        super(HSigmoidv2, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., inplace=self.inplace) / 6.
        return out


# Interface to mmcv
if 'HSigmoidv2' not in ACTIVATION_LAYERS:
    ACTIVATION_LAYERS.register_module('HSigmoidv2', module=HSigmoidv2)
if 'Softmax' not in ACTIVATION_LAYERS:
    ACTIVATION_LAYERS.register_module('Softmax', module=nn.Softmax)
if 'Identity' not in ACTIVATION_LAYERS:
    ACTIVATION_LAYERS.register_module('Identity', module=nn.Identity)


class AttnWeights(nn.Module):
    """ Attention weights for the mixture of affine transformations
        https://arxiv.org/abs/1908.01259
    """

    def __init__(self,
                 attn_mode,
                 num_features,
                 num_affine_trans,
                 num_groups=1,
                 use_rsd=True,
                 use_maxpool=False,
                 use_bn=True,
                 eps=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super(AttnWeights, self).__init__()

        if use_rsd:
            use_maxpool = False

        self.num_affine_trans = num_affine_trans
        self.use_rsd = use_rsd
        self.use_maxpool = use_maxpool
        self.eps = eps
        if not self.use_rsd:
            self.avgpool = nn.AdaptiveAvgPool2d(1)

        layers = []
        if attn_mode == 0:
            layers = [
                nn.Conv2d(num_features, num_affine_trans, 1, bias=not use_bn),
                nn.BatchNorm2d(num_affine_trans) if use_bn else nn.Identity(),
                build_activation_layer(act_cfg),
            ]
        elif attn_mode == 1:
            if num_groups > 0:
                assert num_groups <= num_affine_trans
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.GroupNorm(
                        num_channels=num_affine_trans, num_groups=num_groups),
                    build_activation_layer(act_cfg),
                ]
            else:
                layers = [
                    nn.Conv2d(num_features, num_affine_trans, 1, bias=False),
                    nn.BatchNorm2d(num_affine_trans)
                    if use_bn else nn.Identity(),
                    build_activation_layer(act_cfg),
                ]
        else:
            raise NotImplementedError("Unknow attention weight type")

        self.attention = nn.Sequential(*layers)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)

    def forward(self, x):
        b, c, h, w = x.size()
        if self.use_rsd:
            var, mean = torch.var_mean(x, dim=(2, 3), keepdim=True)
            y = mean * (var + self.eps).rsqrt()

            # var = torch.var(x, dim=(2, 3), keepdim=True)
            # y *= (var + self.eps).rsqrt()
        else:
            y = self.avgpool(x)
            if self.use_maxpool:
                y += F.max_pool2d(x, (h, w), stride=(h, w)).view(b, c, 1, 1)
        return self.attention(y).view(b, self.num_affine_trans)


class AttnBatchNorm2d(nn.BatchNorm2d):
    """ Attentive Normalization with BatchNorm2d backbone
        https://arxiv.org/abs/1908.01259
    """

    _abbr_ = "AttnBN2d"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 attn_mode=0,
                 eps=1e-5,
                 momentum=0.1,
                 track_running_stats=True,
                 use_rsd=True,
                 use_maxpool=False,
                 use_standarized_input=False,
                 use_bn=True,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super().__init__(
            num_features,
            affine=False,
            eps=eps,
            momentum=momentum,
            track_running_stats=track_running_stats)

        self.num_affine_trans = num_affine_trans
        self.attn_mode = attn_mode
        self.use_rsd = use_rsd
        self.use_standarized_input = use_standarized_input
        self.eps_var = eps_var

        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))

        self.attn_weights = AttnWeights(
            attn_mode,
            num_features,
            num_affine_trans,
            use_rsd=use_rsd,
            use_maxpool=use_maxpool,
            use_bn=use_bn,
            eps=eps_var,
            act_cfg=act_cfg)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = super().forward(x)
        size = output.size()
        y = self.attn_weights(
            x if not self.use_standarized_input else output)  # bxk

        weight = y @ self.weight_  # bxc
        bias = y @ self.bias_  # bxc
        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


class AttnGroupNorm(nn.GroupNorm):
    """Attentive Normalization with GroupNorm backbone
        https://arxiv.org/abs/1908.01259
    """

    _abbr_ = "AttnGN"

    def __init__(self,
                 num_features,
                 num_affine_trans,
                 num_groups,
                 num_groups_attn=1,
                 attn_mode=1,
                 eps=1e-5,
                 use_rsd=True,
                 use_bn=True,
                 use_maxpool=False,
                 use_standarized_input=False,
                 eps_var=1e-3,
                 act_cfg=dict(type="HSigmoidv2")):
        super().__init__(
            num_groups=num_groups,
            num_channels=num_features,
            eps=eps,
            affine=False)

        self.num_affine_trans = num_affine_trans
        self.use_standarized_input = use_standarized_input

        self.weight_ = nn.Parameter(
            torch.Tensor(num_affine_trans, num_features))
        self.bias_ = nn.Parameter(torch.Tensor(num_affine_trans, num_features))

        self.attention_weights = AttnWeights(
            attn_mode,
            num_features,
            num_affine_trans,
            num_groups=num_groups_attn,
            use_rsd=use_rsd,
            use_bn=use_bn,
            use_maxpool=use_maxpool,
            eps=eps_var,
            act_cfg=act_cfg)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

    def forward(self, x):
        output = super().forward(x)

        size = output.size()

        y = self.attention_weights(
            x if not self.use_standarized_input else output)

        weight = y @ self.weight_
        bias = y @ self.bias_

        weight = weight.unsqueeze(-1).unsqueeze(-1).expand(size)
        bias = bias.unsqueeze(-1).unsqueeze(-1).expand(size)

        return weight * output + bias


class AttnLayerNorm(nn.LayerNorm):
    """ Attentive Normalization with LayerNorm backbone
        https://arxiv.org/abs/1908.01259
    """

    _abbr_ = "AttnLN"

    def __init__(
            self,
            normalized_shape,
            num_affine_trans,
            eps=1e-6,
            keep_spatial=False,
            use_rsd=False,
            use_standarized_input=False,
            act_layer_attn=None,
            act_layer_affine=partial(nn.Softmax, dim=-1),
            reduction_ratio=0.25,
            device=None,
            dtype=None):
        assert isinstance(
            normalized_shape, numbers.Integral
        ), f'only integral normalized shape supported {normalized_shape}'
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            normalized_shape,
            eps=eps,
            elementwise_affine=False,
            # device=device,
            # dtype=dtype
        )

        self.keep_spatial = keep_spatial
        self.use_rsd = use_rsd
        self.use_standarized_input = use_standarized_input

        # K x C
        affine_shape = tuple([num_affine_trans] + list(self.normalized_shape))

        self.weight_ = nn.Parameter(
            torch.empty(affine_shape, **factory_kwargs))
        self.bias_ = nn.Parameter(torch.empty(affine_shape, **factory_kwargs))

        if act_layer_attn is None:
            attn_weights = [
                nn.Linear(normalized_shape, num_affine_trans),
                act_layer_affine(),
            ]
        else:
            mid_dim = max(4, int(normalized_shape * reduction_ratio))
            attn_weights = [
                nn.Linear(normalized_shape, mid_dim),
                act_layer_attn(),
                nn.Linear(mid_dim, num_affine_trans),
                act_layer_affine(),
            ]

        self.attention_weights = nn.Sequential(*attn_weights)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.weight_, 1., 0.1)
        nn.init.normal_(self.bias_, 0., 0.1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: B N C, or B H W C,  channel_last
        assert x.ndim == 3 or x.ndim == 4

        output = super().forward(x)
        if self.keep_spatial:
            y = output if self.use_standarized_input else x # B N C or B H W C
        else:
            if self.use_rsd:
                var, mean = torch.var_mean(
                    output if self.use_standarized_input else x,
                    dim=1 if x.ndim == 3 else (1,2),
                    keepdim=True)
                y = mean * (var + self.eps).rsqrt()
            else:
                y = torch.mean(
                    output if self.use_standarized_input else x,
                    dim=1 if x.ndim == 3 else (1, 2),
                    keepdim=True)  # B 1 C, or B 1 1 C

        y = self.attention_weights(y)  # B * k, or B * * k

        weight = y @ self.weight_  # B * C, or B * * C
        bias = y @ self.bias_  # B * C, or B * * C

        return weight * output + bias


# Interface to mmcv
if 'AttnBN2d' not in NORM_LAYERS:
    NORM_LAYERS.register_module('AttnBN2d', module=AttnBatchNorm2d)
if 'AttnGN' not in NORM_LAYERS:
    NORM_LAYERS.register_module('AttnGN', module=AttnGroupNorm)
if 'AttnLN' not in NORM_LAYERS:
    NORM_LAYERS.register_module('AttnLN', module=AttnLayerNorm)
