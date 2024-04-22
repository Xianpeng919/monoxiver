import torch.nn as nn

from mmcv.cnn import ConvModule



class Stem(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=7,
                 stride=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_maxpool=True):

        super(Stem, self).__init__()

        self.with_maxpool = with_maxpool

        self.conv1 = ConvModule(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_maxpool:
            self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_maxpool:
            out = self.maxpool1(out)
        return out


class DeepStem(nn.Module):

    def __init__(self,
                 inplanes,
                 planes,
                 kernel_size=3,
                 stride=2,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 with_maxpool=True):

        super(DeepStem, self).__init__()

        self.with_maxpool = with_maxpool

        midplanes = planes // 2
        self.conv1 = nn.Sequential(
            ConvModule(
                inplanes,
                midplanes,
                kernel_size=kernel_size,
                stride=stride,
                padding=(kernel_size - 1) // 2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                midplanes,
                midplanes,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                midplanes,
                planes,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg))
        if self.with_maxpool:
            self.maxpool1 = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.conv1(x)
        if self.with_maxpool:
            out = self.maxpool1(out)
        return out


stem_cfg = {
    # layer_abbreviation: module
    'Stem': Stem,
    'DeepStem': DeepStem,
}


def build_stem_layer(cfg):
    """ Build stem layer

    Args:
        cfg (dict): cfg should contain:
            type (str): Identify activation layer type.
            layer args: args needed to instantiate a stem layer.

    Returns:
        layer (nn.Module): Created stem layer
    """
    assert isinstance(cfg, dict) and 'type' in cfg
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in stem_cfg:
        raise KeyError('Unrecognized stem type {}'.format(layer_type))
    else:
        stem = stem_cfg[layer_type]
        if stem is None:
            raise NotImplementedError

    layer = stem(**cfg_)
    return layer
