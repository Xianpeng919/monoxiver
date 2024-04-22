""" RESEARCH ONLY LICENSE
Copyright (c) 2018-2019 North Carolina State University.
All rights reserved.
Redistribution and use in source and binary forms, with or without modification, are permitted provided
that the following conditions are met:
1. Redistributions and use are permitted for internal research purposes only, and commercial use
is strictly prohibited under this license. Inquiries regarding commercial use should be directed to the
Office of Research Commercialization at North Carolina State University, 919-215-7199,
https://research.ncsu.edu/commercialization/contact/, commercialization@ncsu.edu .
2. Commercial use means the sale, lease, export, transfer, conveyance or other distribution to a
third party for financial gain, income generation or other commercial purposes of any kind, whether
direct or indirect. Commercial use also means providing a service to a third party for financial gain,
income generation or other commercial purposes of any kind, whether direct or indirect.
3. Redistributions of source code must retain the above copyright notice, this list of conditions and
the following disclaimer.
4. Redistributions in binary form must reproduce the above copyright notice, this list of conditions
and the following disclaimer in the documentation and/or other materials provided with the
distribution.
5. The names “North Carolina State University”, “NCSU” and any trade-name, personal name,
trademark, trade device, service mark, symbol, image, icon, or any abbreviation, contraction or
simulation thereof owned by North Carolina State University must not be used to endorse or promote
products derived from this software without prior written permission. For written permission, please
contact trademarks@ncsu.edu.
Disclaimer: THIS SOFTWARE IS PROVIDED “AS IS” AND ANY EXPRESSED OR IMPLIED WARRANTIES,
INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NORTH CAROLINA STATE UNIVERSITY BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
# The system (AOGNet) is protected via patent (pending)
# Written by Tianfu Wu
# Contact: tianfu_wu@ncsu.edu

from .layers import (AttnBatchNorm2d, AttnGroupNorm,
                                           AttnLayerNorm, get_norm_layer,
                                           build_stem_layer)
from .aog import NodeType, build_aog
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
from timm.models.registry import register_model
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import load_checkpoint, BaseModule
from mmcv.cnn import (ConvModule, build_conv_layer, build_norm_layer,
                      build_activation_layer, build_plugin_layer,
                      constant_init, kaiming_init)
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm
import torch.nn as nn
import torch
from functools import partial
import collections
import math
import numpy as np
from os import initgroups
import sys

sys.setrecursionlimit(5000)


__all__ = ['AOGNet',
           'aognet_bn_small', 'aognet_an_small',
           'aognet_bn_base', 'aognet_an_base',
           'aognet_bn_large', 'aognet_an_large',
           'aognet_slim_small', 'aognet_slim_base', 'aognet_slim_large',
           'aognet_bn_small_cifar', 'aognet_an_small_cifar',
           'aognet_bn_base_cifar', 'aognet_an_base_cifar',
           'aognet_bn_large_cifar', 'aognet_an_large_cifar',
           'aognet_slim_small_cifar', 'aognet_slim_base_cifar',
           'aognet_slim_large_cifar',
           'aognet_bn_small_det',
           'aognet_an_small_det' ]


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'input_size': (3, 224, 224),
        'pool_size': None,
        'crop_pct': .95,
        'interpolation': 'bicubic',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


class Transition(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=2,
                 padding=0,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(Transition, self).__init__()

        if stride > 1:
            self.transition = nn.Sequential(
                ConvModule(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg),
                nn.AvgPool2d(kernel_size=(stride, stride), stride=stride))
        else:
            self.transition = nn.Identity()

    def forward(self, x, choice=None):
        return self.transition(x)


class Bottleneck_ResNet(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg1=dict(type='BN'),
                 norm_cfg2=dict(type='BN'),
                 norm_cfg3=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super().__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = [
                'after_conv1', 'after_bn1', 'after_conv2', 'after_bn2',
                'after_conv3', 'after_downsample', 'after_shortcut',
                'after_relu3'
            ]
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg1 = norm_cfg1
        self.norm_cfg2 = norm_cfg2
        self.norm_cfg3 = norm_cfg3
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_bn1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_bn1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_bn2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_bn2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]
            self.after_downsample_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_downsample'
            ]
            self.after_shortcut_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_shortcut'
            ]
            self.after_relu3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_relu3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg1, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg2, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg3, int(planes * self.expansion), postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)

        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            int(planes * self.expansion),
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_bn1_plugin_names = self.make_block_plugins(
                planes, self.after_bn1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_bn2_plugin_names = self.make_block_plugins(
                planes, self.after_bn2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                int(planes * self.expansion), self.after_conv3_plugins)
            if self.downsample is not None:
                self.after_downsample_plugin_names = self.make_block_plugins(
                    int(planes * self.expansion), self.after_downsample_plugins)
            self.after_shortcut_plugin_names = self.make_block_plugins(
                int(planes * self.expansion), self.after_shortcut_plugins)
            self.after_relu3_plugin_names = self.make_block_plugins(
                int(planes * self.expansion), self.after_relu3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn1_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn2_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)
                if self.with_plugins:
                    identity = self.forward_plugin(
                        identity, self.after_downsample_plugin_names)

            out += identity
            if self.with_plugins:
                out = self.forward_plugin(out,
                                          self.after_shortcut_plugin_names)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)
        if self.with_plugins:
            out = self.forward_plugin(out, self.after_relu3_plugin_names)

        return out


class Bottleneck_ResNeXt(Bottleneck_ResNet):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 base_width=4,
                 base_channels=64,
                 **kwargs):
        """Bottleneck block for ResNeXt.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super().__init__(inplanes, planes, **kwargs)

        if groups == 1:
            width = self.planes
        else:
            width = math.floor(self.planes *
                               (base_width / base_channels)) * groups

        self.width = width

        self.norm1_name, norm1 = build_norm_layer(
            self.norm_cfg1, width, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(
            self.norm_cfg2, width, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            self.norm_cfg3, int(self.planes * self.expansion), postfix=3)

        self.conv1 = build_conv_layer(
            self.conv_cfg,
            self.inplanes,
            width,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        self.with_modulated_dcn = False
        if self.with_dcn:
            fallback_on_stride = self.dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                self.conv_cfg,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                self.dcn,
                width,
                width,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=self.dilation,
                dilation=self.dilation,
                groups=groups,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            self.conv_cfg,
            width,
            int(self.planes * self.expansion),
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if self.with_plugins:
            self._del_block_plugins(self.after_conv1_plugin_names +
                                    self.after_bn1_plugin_names +
                                    self.after_conv2_plugin_names +
                                    self.after_bn2_plugin_names)

            self.after_conv1_plugin_names = self.make_block_plugins(
                width, self.after_conv1_plugins)
            self.after_bn1_plugin_names = self.make_block_plugins(
                width, self.after_bn1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                width, self.after_conv2_plugins)
            self.after_bn2_plugin_names = self.make_block_plugins(
                width, self.after_bn2_plugins)
            # self.after_conv3_plugin_names = self.make_block_plugins(
            #     int(planes * self.expansion), self.after_conv3_plugins)
            # if self.downsample is not None:
            #     self.after_downsample_plugin_names = self.make_block_plugins(
            #         int(planes * self.expansion), self.after_downsample_plugins)
            # self.after_shortcut_plugin_names = self.make_block_plugins(
            #     int(planes * self.expansion), self.after_shortcut_plugins)
            # self.after_relu3_plugin_names = self.make_block_plugins(
            #     int(planes * self.expansion), self.after_relu3_plugins)

    def _del_block_plugins(self, plugin_names):
        """delete plugins for block if exist.

        Args:
            plugin_names (list[str]): List of plugins name to delete.
        """
        assert isinstance(plugin_names, list)
        for plugin_name in plugin_names:
            del self._modules[plugin_name]


class Bottleneck(Bottleneck_ResNeXt):

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 drop_rate=0.,
                 drop_path=0.,
                 **kwargs):
        """Bottleneck block for AOGNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        dummy = kwargs.pop('slim_groups', 4)
        dummy = kwargs.pop('slim_ratio', 4)
        super().__init__(inplanes, planes, **kwargs)

        self.outplanes = outplanes
        self.drop_rate = drop_rate
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        _, norm3 = build_norm_layer(self.norm_cfg3, self.outplanes, postfix=3)

        self.conv3 = build_conv_layer(
            self.conv_cfg,
            self.width,
            self.outplanes,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        if self.drop_rate:
            self.drop = nn.Dropout2d(p=self.drop_rate, inplace=True)

        if self.with_plugins:
            self._del_block_plugins(self.after_conv3_plugin_names +
                                    self.after_downsample_plugin_names +
                                    self.after_shortcut_plugin_names +
                                    self.after_relu3_plugin_names)

            self.after_conv3_plugin_names = self.make_block_plugins(
                outplanes, self.after_conv3_plugins)
            self.after_downsample_plugin_names = []
            if self.downsample is not None:
                self.after_downsample_plugin_names = self.make_block_plugins(
                    outplanes, self.after_downsample_plugins)
            self.after_shortcut_plugin_names = self.make_block_plugins(
                outplanes, self.after_shortcut_plugins)
            self.after_relu3_plugin_names = self.make_block_plugins(
                outplanes, self.after_relu3_plugins)

    def forward(self, x, identity=None):

        def _inner_forward(x, identity=None):
            if identity is None:
                identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn1_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            if self.with_plugins:
                out = self.forward_plugin(out, self.after_bn2_plugin_names)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.drop_rate:
                out = self.drop(out)

            if self.downsample is not None:
                identity = self.downsample(identity)

            out = identity + self.drop_path(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, *[x, identity])
        else:
            out = _inner_forward(x, identity)

        out = self.relu(out)

        return out


class BottleneckV2(Bottleneck):

    def forward(self, x, identity=None, lateral=None):

        def _inner_forward(x, identity=None, lateral=None):
            if identity is None:
                identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            if lateral is not None:
                out -= lateral
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.drop_rate:
                out = self.drop(out)

            if self.downsample is not None:
                identity = self.downsample(identity)

            out = identity + self.drop_path(out)

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, *[x, identity, lateral])
        else:
            out = _inner_forward(x, identity, lateral)

        out = self.relu(out)

        return out


class SLIM(nn.Module):
    r""" Adapted from the Same-Layer Inception Module (SLIM) in
        'Learning Inception Attention for Image Synthesis and Image Recognition'
        https://arxiv.org/abs/2112.14804
    """

    def __init__(self, planes, groups, stride, ratio=4):
        super().__init__()
        self.groups = groups
        while planes % self.groups != 0 and self.groups >=2 :
            self.groups = self.groups // 2

        assert planes % self.groups == 0, f'{planes} not divisible by {groups}'

        self.planes = planes
        self.stride = stride

        in_channels = planes // self.groups
        in_channels_se = max(4, in_channels // ratio)

        self.query = nn.ModuleList([])
        self.key = nn.ModuleList([])
        self.value = nn.ModuleList([])
        for _ in range(self.groups):
            self.query.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels, in_channels_se, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(in_channels_se), nn.ReLU(inplace=True),
                    nn.Conv2d(in_channels_se, in_channels, 3, 1, 1)))

            self.key.append(nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(in_channels, in_channels_se, 1, bias=False),
                nn.BatchNorm2d(in_channels_se), nn.ReLU(inplace=True),
                nn.Conv2d(in_channels_se, in_channels, 1)))

            self.value.append(nn.Conv2d(in_channels, in_channels, 3, 1, 1))

        self.downsample = nn.AvgPool2d(
            (self.stride, self.stride)) if stride > 1 else nn.Identity()

    def forward(self, x):

        x = torch.split(x, self.planes // self.groups, dim=1)

        out = []
        for xs, q, k, v in zip(x, self.query, self.key, self.value):
            xq = q(xs)
            xk = k(xs)
            xv = v(xs)
            a = xq * xk
            a = a.softmax(dim=1)
            out.append(a * xv)

        out = torch.cat(out, dim=1)

        out = self.downsample(out)

        return out


class SlimBlock(Bottleneck):

    def __init__(self,
                 inplanes,
                 planes,
                 outplanes,
                 drop_rate=0.,
                 drop_path=0.,
                 **kwargs):
        """Bottleneck block for AOGNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        groups = kwargs.pop('slim_groups', 4)
        ratio = kwargs.pop('slim_ratio', 4)
        super().__init__(inplanes, planes, outplanes, drop_rate, drop_path, **kwargs)

        self.conv2 = SLIM(planes, groups, self.conv2_stride, ratio)


class AOGBlock(nn.Module):

    def __init__(
        self,
        stage_id,
        block_id,
        aog,
        op_t_node,
        op_and_node,
        op_or_node,
        inplanes,
        outplanes,
        bn_ratio=0.25,
        t_node_no_slice=False,
        t_node_handle_dblcnt=False,
        non_t_node_handle_dblcnt=True,
        or_node_reduction='sum',
        op_root_or_node=None,
        kwargs_op_root_or_node=None,
        drop_rate=0.,
        drop_path=None,
        stride=1,
        dilation=1,
        with_group_conv=0,
        base_width=4,
        style='pytorch',
        with_cp=False,
        conv_cfg=None,
        norm_cfg_ds=dict(type='BN'),
        norm_cfg1=dict(type='BN'),
        norm_cfg2=dict(type='BN'),
        norm_cfg3=dict(type='BN'),
        norm_cfg_extra=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        dcn=None,
        plugins=None,
        use_extra_norm_ac_for_block=True,
        slim_groups=4,
        slim_ratio=4,
    ):
        super().__init__()

        self.stage_id = stage_id
        self.block_id = block_id
        self.aog = aog
        self.op_t_node = op_t_node
        self.op_and_node = op_and_node
        self.op_or_node = op_or_node
        self.op_root_or_node = op_root_or_node
        self.kwargs_op_root_or_node = kwargs_op_root_or_node
        self.in_channels = inplanes
        self.out_channels = outplanes
        self.bn_ratio = bn_ratio
        self.t_node_no_slice = t_node_no_slice
        self.t_node_handle_dblcnt = t_node_handle_dblcnt
        self.non_t_node_handle_dblcnt = non_t_node_handle_dblcnt
        self.or_node_reduction = or_node_reduction
        self.drop_rate = drop_rate
        if drop_path is None:
            drop_path = [0.] * aog.param.grid_wd
        assert len(drop_path) == aog.param.grid_wd
        self.drop_path = drop_path
        self.stride = stride
        self.dilation = dilation
        self.with_group_conv = with_group_conv
        self.base_width = base_width
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg_ds = norm_cfg_ds
        self.norm_cfg1 = norm_cfg1
        self.norm_cfg2 = norm_cfg2
        self.norm_cfg3 = norm_cfg3
        self.norm_cfg_extra = norm_cfg_extra
        self.act_cfg = act_cfg
        self.dcn = dcn
        self.plugins = plugins
        self.use_extra_norm_ac_for_block = use_extra_norm_ac_for_block
        self.slim_groups = slim_groups
        self.slim_ratio = slim_ratio

        self.dim = aog.param.grid_wd
        self.in_slices = self._calculate_slices(self.dim, self.in_channels)
        self.out_slices = self._calculate_slices(self.dim, self.out_channels)

        self.node_set = aog.node_set
        self.primitive_set = aog.primitive_set
        self.BFS = aog.BFS
        self.DFS = aog.DFS

        self.hasLateral = {}
        self.hasDblCnt = {}

        self.primitiveDblCnt = None
        self._set_primitive_dbl_cnt()

        self._set_weights_attr()

        self.extra_norm_ac = None
        if self.use_extra_norm_ac_for_block:
            self.extra_norm_ac = self._extra_norm_ac(self.norm_cfg_extra,
                                                     self.out_channels)

        self.init_weights()

    def _calculate_slices(self, dim, channels):
        slices = [0] * dim
        for i in range(channels):
            slices[i % dim] += 1
        for d in range(1, dim):
            slices[d] += slices[d - 1]
        slices = [0] + slices
        return slices

    def _set_primitive_dbl_cnt(self):
        self.primitiveDblCnt = [0.0 for i in range(self.dim)]
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            if node.node_type == NodeType.TerminalNode:
                for i in range(arr.x1, arr.x2 + 1):
                    self.primitiveDblCnt[i] += node.npaths
        for i in range(self.dim):
            assert self.primitiveDblCnt[i] >= 1.0

    def _set_weights_attr(self):
        for id_ in self.DFS:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            keep_norm_base = arr.Width() < self.dim
            if keep_norm_base:
                norm_cfg2_ = self.norm_cfg1
            else:
                norm_cfg2_ = self.norm_cfg2

            if node.node_type == NodeType.TerminalNode:
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False

                inplanes = self.in_channels if self.t_node_no_slice else \
                    self.in_slices[arr.x2 + 1] - self.in_slices[arr.x1]
                outplanes = self.out_slices[arr.x2 +
                                            1] - self.out_slices[arr.x1]
                planes = math.floor(outplanes * self.bn_ratio + 0.5)
                stride = self.stride
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1:  # 32x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio / 2)
                elif self.with_group_conv == 2:  # 64x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)
                elif self.with_group_conv == 4:  # depthwise
                    groups = planes

                downsample = None
                if stride != 1 or inplanes != outplanes:
                    if stride > 1:
                        downsample = nn.Sequential(
                            nn.AvgPool2d(
                                kernel_size=(stride, stride), stride=stride),
                            ConvModule(
                                inplanes,
                                outplanes,
                                1,
                                bias=False,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg_ds,
                                act_cfg=None))
                    else:
                        downsample = ConvModule(
                            inplanes,
                            outplanes,
                            1,
                            bias=False,
                            conv_cfg=self.conv_cfg,
                            norm_cfg=self.norm_cfg_ds,
                            act_cfg=None)

                setattr(
                    self, 'stage_{}_block_{}_node_{}_op'.format(
                        self.stage_id, self.block_id, node.id),
                    self.op_t_node(
                        inplanes=inplanes,
                        planes=planes,
                        outplanes=outplanes,
                        drop_rate=self.drop_rate,
                        drop_path=self.drop_path[arr.Width()-1],
                        stride=stride,
                        dilation=self.dilation,
                        downsample=downsample,
                        groups=groups,
                        base_width=self.base_width,
                        base_channels=base_channels,
                        style=self.style,
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg1=self.norm_cfg1,
                        norm_cfg2=norm_cfg2_,
                        norm_cfg3=self.norm_cfg3,
                        dcn=self.dcn,
                        slim_groups=self.slim_groups,
                        slim_ratio=self.slim_ratio))

            elif node.node_type == NodeType.AndNode:
                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if arr.Width() == ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if self.non_t_node_handle_dblcnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break

                inplanes = self.out_slices[arr.x2 +
                                           1] - self.out_slices[arr.x1]
                outplanes = inplanes
                planes = math.floor(outplanes * self.bn_ratio + 0.5)
                stride = 1
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1:  # 32x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio / 2)
                elif self.with_group_conv == 2:  # 64x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)
                elif self.with_group_conv == 4:  # depthwise
                    groups = planes

                setattr(
                    self, 'stage_{}_block_{}_node_{}_op'.format(
                        self.stage_id, self.block_id, node.id),
                    self.op_and_node(
                        inplanes=inplanes,
                        planes=planes,
                        outplanes=outplanes,
                        drop_rate=self.drop_rate,
                        drop_path=self.drop_path[arr.Width()-1],
                        stride=stride,
                        dilation=self.dilation,
                        groups=groups,
                        base_width=self.base_width,
                        base_channels=base_channels,
                        style=self.style,
                        with_cp=self.with_cp,
                        conv_cfg=self.conv_cfg,
                        norm_cfg1=self.norm_cfg1,
                        norm_cfg2=norm_cfg2_,
                        norm_cfg3=self.norm_cfg3,
                        dcn=self.dcn,
                        slim_groups=self.slim_groups,
                        slim_ratio=self.slim_ratio))

            elif node.node_type == NodeType.OrNode:
                assert self.node_set[node.child_ids[0]].node_type != \
                    NodeType.OrNode

                self.hasLateral[node.id] = False
                self.hasDblCnt[node.id] = False
                for chid in node.child_ids:
                    ch_arr = self.primitive_set[self.node_set[chid].rect_idx]
                    if self.node_set[chid].node_type == NodeType.OrNode or \
                            arr.Width() < ch_arr.Width():
                        self.hasLateral[node.id] = True
                        break
                if self.non_t_node_handle_dblcnt:
                    for chid in node.child_ids:
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if not (self.node_set[chid].node_type
                                == NodeType.OrNode
                                or arr.Width() < ch_arr.Width()):
                            if node.npaths / self.node_set[chid].npaths != 1.0:
                                self.hasDblCnt[node.id] = True
                                break

                inplanes = self.out_slices[arr.x2 +
                                           1] - self.out_slices[arr.x1]
                outplanes = inplanes
                planes = math.floor(outplanes * self.bn_ratio + 0.5)
                stride = 1
                groups = 1
                base_channels = planes
                if self.with_group_conv == 1:  # 32x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio / 2)
                elif self.with_group_conv == 2:  # 64x4d
                    groups = math.floor(planes / self.base_width /
                                        self.bn_ratio)
                elif self.with_group_conv == 3:
                    groups = math.floor(planes / self.base_width)
                elif self.with_group_conv == 4:  # depthwise
                    groups = planes

                if node.id == self.BFS[0] and self.op_root_or_node is not None:
                    assert self.kwargs_op_root_or_node is not None
                    assert outplanes % self.kwargs_op_root_or_node['num_heads'] == 0, \
                        "{outplane} not divisible by {self.kwargs_op_root_or_node['num_heads']}"
                    self.kwargs_op_root_or_node['dim'] = outplanes
                    self.kwargs_op_root_or_node['drop_path'] = self.drop_path[arr.Width(
                    )-1]

                    setattr(
                        self, 'stage_{}_block_{}_node_{}_op'.format(
                            self.stage_id, self.block_id, node.id),
                        self.op_root_or_node(**self.kwargs_op_root_or_node))
                else:
                    setattr(
                        self, 'stage_{}_block_{}_node_{}_op'.format(
                            self.stage_id, self.block_id, node.id),
                        self.op_or_node(
                            inplanes=inplanes,
                            planes=planes,
                            outplanes=outplanes,
                            drop_rate=self.drop_rate,
                            drop_path=self.drop_path[arr.Width()-1],
                            stride=stride,
                            dilation=self.dilation,
                            groups=groups,
                            base_width=self.base_width,
                            base_channels=base_channels,
                            style=self.style,
                            with_cp=self.with_cp,
                            conv_cfg=self.conv_cfg,
                            norm_cfg1=self.norm_cfg1,
                            norm_cfg2=norm_cfg2_,
                            norm_cfg3=self.norm_cfg3,
                            dcn=self.dcn,
                            slim_groups=self.slim_groups,
                            slim_ratio=self.slim_ratio))

    def _extra_norm_ac(self, norm_cfg, num_features):
        return nn.Sequential(
            build_norm_layer(norm_cfg, num_features)[1],
            build_activation_layer(self.act_cfg))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m,
                            (AttnBatchNorm2d, AttnGroupNorm, AttnLayerNorm)):
                nn.init.normal_(m.weight_, 1., 0.1)
                nn.init.normal_(m.bias_, 0., 0.1)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m, 1)

    def forward(self, x):
        NodeIdTensorDict = {}

        # handle input x
        tnode_dblcnt = False
        if self.t_node_handle_dblcnt and self.in_channels == self.out_channels:
            x_scaled = []
            for i in range(self.dim):
                left, right = self.in_slices[i], self.in_slices[i + 1]
                x_scaled.append(x[:, left:right, :, :].div(
                    self.primitiveDblCnt[i]))
            xx = torch.cat(x_scaled, 1)
            tnode_dblcnt = True

        DFS_ = self.DFS

        # T-nodes
        for id_ in DFS_:
            node = self.node_set[id_]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(
                self.stage_id, self.block_id, node.id)

            if node.node_type == NodeType.TerminalNode:
                arr = self.primitive_set[node.rect_idx]
                right, left = self.in_slices[arr.x2 +
                                             1], self.in_slices[arr.x1]
                tnode_tensor_op = x if self.t_node_no_slice else \
                    x[:, left:right, :, :].contiguous()  # TODO: use unfold ?
                # assert tnode_tensor.requires_grad, 'slice needs to retain grad'

                if tnode_dblcnt:
                    tnode_tensor_res = xx[:, left:right, :, :].mul(node.npaths)
                    tnode_output = getattr(self, op_name)(
                        tnode_tensor_op, identity=tnode_tensor_res)
                else:
                    tnode_output = getattr(self, op_name)(tnode_tensor_op)

                NodeIdTensorDict[node.id] = tnode_output

        # AND- and OR-nodes
        node_op_idx_ = 0
        for id_ in DFS_:
            node = self.node_set[id_]
            arr = self.primitive_set[node.rect_idx]
            op_name = 'stage_{}_block_{}_node_{}_op'.format(
                self.stage_id, self.block_id, node.id)

            if node.node_type == NodeType.AndNode:
                if self.hasDblCnt[node.id]:
                    child_tensor_res = []
                    child_tensor_op = []
                    for chid in node.child_ids:
                        if chid not in DFS_:
                            continue
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            factor = node.npaths / self.node_set[chid].npaths
                            if factor == 1.0:
                                child_tensor_res.append(NodeIdTensorDict[chid])
                            else:
                                child_tensor_res.append(
                                    NodeIdTensorDict[chid].mul(factor))
                            child_tensor_op.append(NodeIdTensorDict[chid])

                    anode_tensor_res = torch.cat(child_tensor_res, 1)
                    anode_tensor_op = torch.cat(child_tensor_op, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width():
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]
                                if len(ids1.intersection(ids2)) == num_shared:
                                    anode_tensor_res = anode_tensor_res + \
                                        NodeIdTensorDict[chid]

                    anode_output = getattr(self, op_name)(
                        anode_tensor_op, identity=anode_tensor_res)

                else:
                    child_tensor = []
                    for chid in node.child_ids:
                        if chid not in DFS_:
                            continue
                        ch_arr = self.primitive_set[
                            self.node_set[chid].rect_idx]
                        if arr.Width() > ch_arr.Width():
                            child_tensor.append(NodeIdTensorDict[chid])

                    anode_tensor_op = torch.cat(child_tensor, 1)

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]

                        anode_tensor_res = anode_tensor_op

                        for chid in node.child_ids:
                            if chid not in DFS_:
                                continue
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if arr.Width() == ch_arr.Width() and \
                                    len(ids1.intersection(ids2)) > num_shared:
                                anode_tensor_op = anode_tensor_op + \
                                    NodeIdTensorDict[chid]

                        anode_output = getattr(self, op_name)(
                            anode_tensor_op, identity=anode_tensor_res)
                    else:
                        anode_output = getattr(self, op_name)(anode_tensor_op)

                NodeIdTensorDict[node.id] = anode_output

            elif node.node_type == NodeType.OrNode:
                num_op_sum = 0.
                num_res_sum = 0.
                if self.hasDblCnt[node.id]:
                    factor = node.npaths / \
                        self.node_set[node.child_ids[0]].npaths

                    if factor == 1.0:
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                    else:
                        onode_tensor_res = \
                            NodeIdTensorDict[node.child_ids[0]].mul(factor)
                    num_res_sum += 1.

                    onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                    num_op_sum += 1.
                    for chid in node.child_ids[1:]:
                        if chid not in DFS_:
                            continue
                        if self.node_set[chid].node_type != NodeType.OrNode:
                            ch_arr = self.primitive_set[
                                self.node_set[chid].rect_idx]
                            if arr.Width() == ch_arr.Width():
                                factor = node.npaths / \
                                    self.node_set[chid].npaths
                                if factor == 1.0:
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]
                                else:
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid].mul(factor)
                                num_res_sum += 1.
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                    if self.hasLateral[node.id]:
                        ids1 = set(node.parent_ids)
                        num_shared = 0
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == \
                                    NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) == num_shared:
                                onode_tensor_res = onode_tensor_res + \
                                    NodeIdTensorDict[chid]
                                num_res_sum += 1.
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            ch_arr = \
                                self.primitive_set[self.node_set[chid].rect_idx]
                            ids2 = self.node_set[chid].parent_ids
                            if self.node_set[chid].node_type == \
                                NodeType.OrNode and \
                                    len(ids1.intersection(ids2)) > num_shared:

                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.
                            elif self.node_set[chid].node_type == \
                                    NodeType.TerminalNode and \
                                    arr.Width() < ch_arr.Width():
                                ch_left = self.out_slices[arr.x1] - \
                                    self.out_slices[ch_arr.x1]
                                ch_right = self.out_slices[arr.x2 + 1] - \
                                    self.out_slices[ch_arr.x1]
                                if self.or_node_reduction == 'max':
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid]
                                        [:, ch_left:ch_right, :, :])
                                else:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid][:,
                                                               ch_left:ch_right, :, :]  # .contiguous()
                                    num_op_sum += 1.

                    if self.or_node_reduction == 'avg':
                        onode_tensor_res = onode_tensor_res / num_res_sum
                        onode_tensor_op = onode_tensor_op / num_op_sum

                    onode_output = getattr(self, op_name)(
                        onode_tensor_op, identity=onode_tensor_res)
                else:
                    if self.or_node_reduction == 'max':
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        onode_tensor_res = NodeIdTensorDict[node.child_ids[0]]
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = \
                                    self.primitive_set[self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == \
                                        NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                    onode_tensor_res = onode_tensor_res + \
                                        NodeIdTensorDict[chid]

                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ch_arr = \
                                    self.primitive_set[self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op,
                                        NodeIdTensorDict[chid])
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                        arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - \
                                        self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[
                                        arr.x2 +
                                        1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op = torch.max(
                                        onode_tensor_op, NodeIdTensorDict[chid]
                                        [:, ch_left:ch_right, :, :])

                            onode_output = getattr(self, op_name)(
                                onode_tensor_op, identity=onode_tensor_res)
                        else:
                            onode_output = getattr(self, op_name)(
                                onode_tensor_op)
                    else:
                        onode_tensor_op = NodeIdTensorDict[node.child_ids[0]]
                        num_op_sum += 1.
                        for chid in node.child_ids[1:]:
                            if chid not in DFS_:
                                continue
                            if self.node_set[chid].node_type != NodeType.OrNode:
                                ch_arr = self.primitive_set[
                                    self.node_set[chid].rect_idx]
                                if arr.Width() == ch_arr.Width():
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                        if self.hasLateral[node.id]:
                            ids1 = set(node.parent_ids)
                            num_shared = 0
                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) == num_shared:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.

                            onode_tensor_res = onode_tensor_op
                            num_res_sum = num_op_sum

                            for chid in node.child_ids[1:]:
                                if chid not in DFS_:
                                    continue
                                ch_arr = self.primitive_set[
                                    self.node_set[chid].rect_idx]
                                ids2 = self.node_set[chid].parent_ids
                                if self.node_set[chid].node_type == NodeType.OrNode and \
                                        len(ids1.intersection(ids2)) > num_shared:
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid]
                                    num_op_sum += 1.
                                elif self.node_set[chid].node_type == NodeType.TerminalNode and \
                                        arr.Width() < ch_arr.Width():
                                    ch_left = self.out_slices[arr.x1] - \
                                        self.out_slices[ch_arr.x1]
                                    ch_right = self.out_slices[
                                        arr.x2 +
                                        1] - self.out_slices[ch_arr.x1]
                                    onode_tensor_op = onode_tensor_op + \
                                        NodeIdTensorDict[chid][:, ch_left:ch_right, :, :].contiguous(
                                        )
                                    num_op_sum += 1.

                            if self.or_node_reduction == 'avg':
                                onode_tensor_op = onode_tensor_op / num_op_sum
                                onode_tensor_res = onode_tensor_res / num_res_sum

                            onode_output = getattr(self, op_name)(
                                onode_tensor_op, identity=onode_tensor_res)
                        else:
                            if self.or_node_reduction == 'avg':
                                onode_tensor_op = onode_tensor_op / num_op_sum
                            onode_output = getattr(self, op_name)(
                                onode_tensor_op)

                NodeIdTensorDict[node.id] = onode_output

        out = NodeIdTensorDict[self.aog.BFS[0]]

        if self.extra_norm_ac is not None:
            out = self.extra_norm_ac(out)

        return out


class AOGNet(nn.Module):
    """ AOGNet
    http://openaccess.thecvf.com/content_CVPR_2019/papers/Li_AOGNets_Compositional_Grammatical_Architectures_for_Deep_Learning_CVPR_2019_paper.pdf
    """

    def __init__(self,
                 aog_cfg,
                 in_channels=3,
                 stem_cfg=dict(
                     type="DeepStem",
                     kernel_size=3,
                     stride=2,
                     norm_cfg=dict(type='BN'),
                     act_cfg=dict(type='ReLU', inplace=True),
                     with_maxpool=True),
                 block='AOGBlock',
                 block_num=(2, 2, 2, 1),
                 filter_list=(32, 128, 256, 512, 824),
                 ops_t_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                             'Bottleneck'),
                 ops_and_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                               'Bottleneck'),
                 ops_or_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                              'Bottleneck'),
                 ops_root_or_node=(None, None, None, None),
                 kwargs_ops_root_or_node=(None, None, None, None),
                 bn_ratios=(0.25, 0.25, 0.25, 0.25),
                 t_node_no_slice=(False, False, False, False),
                 t_node_handle_dblcnt=(False, False, False, False),
                 non_t_node_handle_dblcnt=(True, True, True, True),
                 or_node_reduction='sum',
                 drop_rates=(0., 0., 0.1, 0.1),
                 drop_path_rate=0.,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 with_group_conv=(False, False, False, False),
                 base_width=(4, 4, 4, 4),
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 style='pytorch',
                 conv_cfg=None,
                 norm_cfg_stem=dict(type='BN', requires_grad=True),
                 norm_cfg_transition=dict(type='BN', requires_grad=True),
                 norm_cfg_ds=dict(type='BN', requires_grad=True),
                 norm_cfg1=dict(type='BN', requires_grad=True),
                 norm_cfg2=dict(type='BN', requires_grad=True),
                 norm_cfg3=dict(type='BN', requires_grad=True),
                 norm_cfg_extra=dict(type='BN', requires_grad=True),
                 num_affine_trans=(10, 10, 20, 20),
                 norm_eval=False,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=False,
                 use_extra_norm_ac_for_block=True,
                 handle_dbl_cnt_in_weight_init=False,
                 num_classes=0,
                 slim_groups=4,
                 slim_ratio=4,
                 pretrained=None):
        """
        Separate norm_cfgs for flexibilty and ablation studies of AN
        norm_cfg_stem: for stem
        norm_cfg_transition: for transition module between aog blocks
        norm_cfg_ds: for downsampling module in a block
        norm_cfg1: for the 1st conv in a block(Basic Or Bottleneck)
        norm_cfg2: for the 2nd conv in a block(Basic Or Bottleneck)
        norm_cfg3: for the 3rd conv in a block(Bottleneck)
        norm_cfg_extra: for the extra_norm_ac module
        """
        super().__init__()
        self.num_stages = len(filter_list) - 1
        assert self.num_stages == len(aog_cfg['dims'])
        self.aog_cfg = aog_cfg
        self.aogs = build_aog(aog_cfg)
        self.in_channels = in_channels
        self.stem_cfg = stem_cfg
        self.block = eval(block)
        self.block_num = block_num
        self.filter_list = filter_list
        self.ops_t_node = ops_t_node
        self.ops_and_node = ops_and_node
        self.ops_or_node = ops_or_node
        self.ops_root_or_node = ops_root_or_node
        self.kwargs_ops_root_or_node = kwargs_ops_root_or_node
        self.bn_ratios = bn_ratios
        self.t_node_no_slice = t_node_no_slice
        self.t_node_handle_dblcnt = t_node_handle_dblcnt
        self.non_t_node_handle_dblcnt = non_t_node_handle_dblcnt
        self.or_node_reduction = or_node_reduction
        self.drop_rates = drop_rates
        self.drop_path_rate = drop_path_rate
        if drop_path_rate > 0.0:  # aognet was developed using dropout only
            self.drop_rates = tuple([0.] * len(self.drop_rates))
        self.strides = strides
        self.dilations = dilations
        self.with_group_conv = with_group_conv
        self.base_width = base_width
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.style = style
        self.conv_cfg = conv_cfg
        self.norm_cfg_stem = norm_cfg_stem
        self.norm_cfg_transition = norm_cfg_transition
        self.norm_cfg_ds = norm_cfg_ds
        self.norm_cfg1 = norm_cfg1
        self.norm_cfg2 = norm_cfg2
        self.norm_cfg3 = norm_cfg3
        self.norm_cfg_extra = norm_cfg_extra
        self.num_affine_trans = num_affine_trans
        self.norm_eval = norm_eval
        self.act_cfg = act_cfg
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == self.num_stages
        self.with_cp = with_cp
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.use_extra_norm_ac_for_block = use_extra_norm_ac_for_block
        self.handle_dbl_cnt_in_weight_init = handle_dbl_cnt_in_weight_init
        self.num_classes = num_classes
        self.slim_groups = slim_groups
        self.slim_ratio = slim_ratio

        self._make_stem_layer(filter_list[0])

        self._make_stages()

        self.with_classification = num_classes > 0
        if self.with_classification:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.drop = None
            if len(self.drop_rates) > self.num_stages:
                self.drop = nn.Dropout(p=self.drop_rates[-1], inplace=True)
            self.fc = nn.Linear(filter_list[-1], num_classes)

        self._freeze_stages()

        self.feat_dim = self.filter_list[-1]

        ## initialize
        self.init_weights(pretrained)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (AttnBatchNorm2d, AttnGroupNorm, AttnLayerNorm)):
                    nn.init.normal_(m.weight_, 1., 0.1)
                    nn.init.normal_(m.bias_, 0., 0.1)
                elif isinstance(m, (_BatchNorm, nn.LayerNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.handle_dbl_cnt_in_weight_init:
                import re
                for name_, m in self.named_modules():
                    if 'node' in name_:
                        idx = re.findall(r'\d+', name_)
                        stage_id = int(idx[-3])
                        node_id = int(idx[-1])
                        npaths = self.aogs[stage_id -
                                           1].node_set[node_id].npaths
                        if npaths > 1:
                            scale = 1.0 / npaths
                            with torch.no_grad():
                                for ch in m.modules():
                                    if isinstance(ch, nn.Conv2d):
                                        ch.weight.mul_(scale)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, (Bottleneck, SlimBlock)):
                        if isinstance(m.norm3,
                                      (AttnBatchNorm2d, AttnGroupNorm)):
                            nn.init.constant_(m.norm3.weight_, 0.)
                            nn.init.constant_(m.norm3.bias_, 0.)
                        else:
                            constant_init(m.norm3, 0)
        else:
            raise TypeError('pretrained must be a str or None')

    def _make_stem_layer(self, planes):
        if 'Attn' in self.norm_cfg_stem['type']:
            self.norm_cfg_stem['num_affine_trans'] = self.num_affine_trans[0]
        self.stem_cfg['inplanes'] = self.in_channels
        self.stem_cfg['planes'] = planes
        self.stem_cfg['norm_cfg'] = self.norm_cfg_stem

        self.stem = build_stem_layer(self.stem_cfg)

    def _make_stages(self):
        depths = [aog_dim * nb for aog_dim,
                  nb in zip(self.aog_cfg['dims'], self.block_num)]
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate,
                                             sum(depths))
        ]  # stochastic depth decay rule
        cur = 0

        self.aog_layers = []
        for stage_id in range(self.num_stages):
            aog = self.aogs[stage_id]
            dim = aog.param.grid_wd
            in_channels = self.filter_list[stage_id]
            out_channels = self.filter_list[stage_id + 1]

            assert in_channels % dim == 0 and out_channels % dim == 0
            step_channels = (out_channels -
                             in_channels) // self.block_num[stage_id]
            if step_channels % dim != 0:
                low = (step_channels // dim) * dim
                high = (step_channels // dim + 1) * dim
                if (step_channels - low) <= (high - step_channels):
                    step_channels = low
                else:
                    step_channels = high

            if 'Attn' in self.norm_cfg_transition['type']:
                self.norm_cfg_transition['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg_ds['type']:
                self.norm_cfg_ds['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg1['type']:
                self.norm_cfg1['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg2['type']:
                self.norm_cfg2['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg3['type']:
                self.norm_cfg3['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]
            if 'Attn' in self.norm_cfg_extra['type']:
                self.norm_cfg_extra['num_affine_trans'] = \
                    self.num_affine_trans[stage_id]

            aog_layer = []

            for j in range(self.block_num[stage_id]):
                stride = self.strides[stage_id] if j == 0 else 1
                outchannels = (in_channels + step_channels) if \
                    j < self.block_num[stage_id]-1 else out_channels

                # transition
                aog_layer.append(
                    Transition(
                        in_channels,
                        in_channels,
                        kernel_size=1,
                        stride=stride,
                        padding=0,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg_transition,
                        act_cfg=self.act_cfg))

                # blocks
                bn_ratio = self.bn_ratios[stage_id]
                aog_layer.append(
                    self.block(stage_id=stage_id+1,
                               block_id=j+1,
                               aog=aog,
                               op_t_node=eval(self.ops_t_node[stage_id]),
                               op_and_node=eval(self.ops_and_node[stage_id]),
                               op_or_node=eval(self.ops_or_node[stage_id]),
                               inplanes=in_channels,
                               outplanes=outchannels,
                               bn_ratio=bn_ratio,
                               t_node_no_slice=self.t_node_no_slice[stage_id],
                               t_node_handle_dblcnt=self.t_node_handle_dblcnt[stage_id],
                               non_t_node_handle_dblcnt=self.non_t_node_handle_dblcnt[stage_id],
                               or_node_reduction=self.or_node_reduction,
                               op_root_or_node=eval(
                                   self.ops_root_or_node[stage_id])
                               if self.ops_root_or_node[stage_id] is not None else None,
                               kwargs_op_root_or_node=self.kwargs_ops_root_or_node[stage_id],
                               drop_rate=self.drop_rates[stage_id],
                               drop_path=dpr[cur + j * aog.param.grid_wd: cur +
                                             (j+1) * aog.param.grid_wd],
                               stride=1,
                               dilation=self.dilations[stage_id],
                               with_group_conv=self.with_group_conv[stage_id],
                               base_width=self.base_width[stage_id],
                               style=self.style,
                               with_cp=self.with_cp,
                               conv_cfg=self.conv_cfg,
                               norm_cfg_ds=self.norm_cfg_ds,
                               norm_cfg1=self.norm_cfg1,
                               norm_cfg2=self.norm_cfg2,
                               norm_cfg3=self.norm_cfg3,
                               norm_cfg_extra=self.norm_cfg_extra,
                               act_cfg=self.act_cfg,
                               dcn=self.dcn,
                               plugins=self.plugins,
                               use_extra_norm_ac_for_block=self.use_extra_norm_ac_for_block,
                               slim_groups=self.slim_groups,
                               slim_ratio=self.slim_groups))

                in_channels = outchannels

            cur += depths[stage_id]

            layer_name = f'layer{stage_id + 1}'
            self.add_module(layer_name, nn.Sequential(*aog_layer))
            self.aog_layers.append(layer_name)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.stem.eval()
            for param in self.stem.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def forward(self, x):
        y = self.stem(x)

        outs = []
        for i, layer_name in enumerate(self.aog_layers):
            aog_layer = getattr(self, layer_name)
            y = aog_layer(y)
            if i in self.out_indices:
                outs.append(y)

        if self.with_classification:
            y = self.avgpool(y)
            y = y.reshape(y.size(0), -1)
            if self.drop is not None:
                y = self.drop(y)
            y = self.fc(y)
            return y

        return tuple(outs)

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()


### interface to timm

_model_configs = dict(
    aog_cfg=dict(
        dims=(2, 2, 4, 4),
        max_splits=(2, 2, 2, 2),
        extra_node_hierarchy=('4', '4', '4', '4'),
        remove_symmetric_children_of_or_node=(1, 2, 1, 2)),
    in_channels=3,
    stem_cfg=dict(
        type="DeepStem",
        kernel_size=3,
        stride=2,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=True),
    block='AOGBlock',
    block_num=(2, 2, 2, 1),
    filter_list=(32, 128, 256, 512, 824),
    ops_t_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                'Bottleneck'),
    ops_and_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                  'Bottleneck'),
    ops_or_node=('Bottleneck', 'Bottleneck', 'Bottleneck',
                 'Bottleneck'),
    ops_root_or_node=(None, None, None, None),
    kwargs_ops_root_or_node=(None, None, None, None),
    bn_ratios=(0.25, 0.25, 0.25, 0.25),
    t_node_no_slice=(False, False, False, False),
    t_node_handle_dblcnt=(False, False, False, False),
    non_t_node_handle_dblcnt=(True, True, True, True),
    or_node_reduction='sum',
    drop_rates=(0., 0., 0.1, 0.1),
    drop_path_rate=0.,
    strides=(1, 2, 2, 2),
    dilations=(1, 1, 1, 1),
    with_group_conv=(False, False, False, False),
    base_width=(4, 4, 4, 4),
    out_indices=(0, 1, 2, 3),
    frozen_stages=-1,
    style='pytorch',
    conv_cfg=None,
    norm_cfg_stem=dict(type='BN', requires_grad=True),
    norm_cfg_transition=dict(type='BN', requires_grad=True),
    norm_cfg_ds=dict(type='BN', requires_grad=True),
    norm_cfg1=dict(type='BN', requires_grad=True),
    norm_cfg2=dict(type='BN', requires_grad=True),
    norm_cfg3=dict(type='BN', requires_grad=True),
    norm_cfg_extra=dict(type='BN', requires_grad=True),
    num_affine_trans=(10, 10, 20, 20),
    norm_eval=False,
    act_cfg=dict(type='ReLU', inplace=True),
    dcn=None,
    stage_with_dcn=(False, False, False, False),
    plugins=None,
    with_cp=False,
    zero_init_residual=False,
    use_extra_norm_ac_for_block=True,
    handle_dbl_cnt_in_weight_init=False,
    num_classes=0,
    slim_groups=4,
    slim_ratio=4,
)

# imagenet
@register_model
def aognet_bn_small(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 2, 1),
               filter_list=(32, 128, 256, 512, 824),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_small(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 2, 1),
               filter_list=(32, 128, 256, 512, 824),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               pretrained=kwargs['pretrained'],
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_bn_base(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 3, 1),
               filter_list=(56, 224, 448, 896, 1400),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_base(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 3, 1),
               filter_list=(56, 224, 448, 896, 1400),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_bn_large(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 5, 1),
               filter_list=(64, 256, 512, 1024, 1440),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_large(pretrained=False, **kwargs):

    cfg = dict(block_num=(2, 2, 5, 1),
               filter_list=(64, 256, 512, 1024, 1440),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_small(pretrained=False, **kwargs):

    cfg = dict(ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                           'SlimBlock'),
               ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                             'SlimBlock'),
               ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                            'SlimBlock'),
               block_num=(2, 2, 2, 1),
               filter_list=(32, 128, 256, 512, 824),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               slim_groups=4,
               slim_ratio=4,
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_base(pretrained=False, **kwargs):

    cfg = dict(ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                           'SlimBlock'),
               ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                             'SlimBlock'),
               ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                            'SlimBlock'),
               block_num=(2, 2, 3, 1),
               filter_list=(56, 224, 448, 896, 1400),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               slim_groups=4,
               slim_ratio=4,
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_large(pretrained=False, **kwargs):

    cfg = dict(ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                           'SlimBlock'),
               ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                             'SlimBlock'),
               ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                            'SlimBlock'),
               block_num=(2, 2, 5, 1),
               filter_list=(64, 256, 512, 1024, 1440),
               drop_path_rate=kwargs.pop('drop_path_rate', 0.),
               num_classes=kwargs['num_classes'],
               slim_groups=4,
               slim_ratio=4,
               )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


# cifar
@register_model
def aognet_bn_small_cifar(pretrained=False, **kwargs):

    cfg = dict(
        stem_cfg=dict(
            type="Stem",
            kernel_size=3,
            stride=1,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='ReLU', inplace=True),
            with_maxpool=False),
        block_num=(2, 2, 2, 1),
        filter_list=(32, 128, 256, 512, 824),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_small_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        block_num=(2, 2, 2, 1),
        filter_list=(32, 128, 256, 512, 824),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_bn_base_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        block_num=(2, 2, 3, 1),
        filter_list=(56, 224, 448, 896, 1400),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_base_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        block_num=(2, 2, 3, 1),
        filter_list=(56, 224, 448, 896, 1400),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_bn_large_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        block_num=(2, 2, 5, 1),
        filter_list=(64, 256, 512, 1024, 1440),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_an_large_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        block_num=(2, 2, 3, 1),
        filter_list=(64, 256, 512, 1024, 1440),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_small_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                    'SlimBlock'),
        ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                      'SlimBlock'),
        ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                     'SlimBlock'),
        block_num=(2, 2, 2, 1),
        filter_list=(32, 128, 256, 512, 824),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        slim_groups=4,
        slim_ratio=4,
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_base_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                    'SlimBlock'),
        ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                      'SlimBlock'),
        ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                     'SlimBlock'),
        block_num=(2, 2, 3, 1),
        filter_list=(56, 224, 448, 896, 1400),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        slim_groups=4,
        slim_ratio=4,
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model


@register_model
def aognet_slim_large_cifar(pretrained=False, **kwargs):

    cfg = dict(stem_cfg=dict(
        type="Stem",
        kernel_size=3,
        stride=1,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU', inplace=True),
        with_maxpool=False),
        ops_t_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                    'SlimBlock'),
        ops_and_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                      'SlimBlock'),
        ops_or_node=('SlimBlock', 'SlimBlock', 'SlimBlock',
                     'SlimBlock'),
        block_num=(2, 2, 5, 1),
        filter_list=(64, 256, 512, 1024, 1440),
        drop_path_rate=kwargs.pop('drop_path_rate', 0.),
        num_classes=kwargs['num_classes'],
        slim_groups=4,
        slim_ratio=4,
    )

    model_cfg = _model_configs.copy()
    model_cfg.update(cfg)

    model = AOGNet(**model_cfg)

    model.default_cfg = _cfg()

    return model



# Downstream tasks with ImageNet pretrained models

# interface to mmdet

@BACKBONES.register_module()
class aognet_bn_small_det(AOGNet):

    def __init__(self, **kwargs):

        cfg = dict(block_num=(2, 2, 2, 1),
               filter_list=(32, 128, 256, 512, 824),
               drop_path_rate=0.,
               num_classes=0,
               frozen_stages=0,
               norm_eval=True,
               pretrained=kwargs['pretrained']
               )

        model_cfg = _model_configs.copy()
        model_cfg.update(cfg)

        super().__init__(**model_cfg)

        del self.num_classes

@BACKBONES.register_module()
class aognet_an_small_det(AOGNet):

    def __init__(self, **kwargs):

        cfg = dict(block_num=(2, 2, 2, 1),
               filter_list=(32, 128, 256, 512, 824),
               drop_path_rate=0.,
               num_classes=0,
               norm_cfg2=dict(type='AttnBN2d', requires_grad=True),
               frozen_stages=0,
               norm_eval=True,
               pretrained=kwargs['pretrained']
               )

        model_cfg = _model_configs.copy()
        model_cfg.update(cfg)

        super().__init__(**model_cfg)

        del self.num_classes


# model check


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


if __name__ == '__main__':
    img = torch.randn(5, 3, 224, 224)

    # model = aognet_bn_small(num_classes=1000)
    # out = model(img)
    # print('aognet_bn_small:', out.shape, count_parameters(model))

    # model = aognet_an_small(num_classes=1000)
    # out = model(img)
    # print('aognet_an_small:', out.shape, count_parameters(model))

    # model = aognet_bn_base(num_classes=1000)
    # out = model(img)
    # print('aognet_bn_base:', out.shape, count_parameters(model))

    # model = aognet_an_base(num_classes=1000)
    # out = model(img)
    # print('aognet_an_base:', out.shape, count_parameters(model))

    # model = aognet_bn_large(num_classes=1000)
    # out = model(img)
    # print('aognet_bn_large:', out.shape, count_parameters(model))

    # model = aognet_an_large(num_classes=1000)
    # out = model(img)
    # print('aognet_an_large:', out.shape, count_parameters(model))

    model = aognet_slim_small(num_classes=1000)
    out = model(img)
    print('aognet_slim_small:', out.shape, count_parameters(model))

    model = aognet_slim_base(num_classes=1000)
    out = model(img)
    print('aognet_slim_base:', out.shape, count_parameters(model))

    model = aognet_slim_large(num_classes=1000)
    out = model(img)
    print('aognet_slim_large:', out.shape, count_parameters(model))

    #
    # print(model)
