import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np
import random
import mmengine
from abc import ABCMeta, abstractmethod
from mmengine.model import BaseModule

class BaseBackbone(BaseModule, metaclass=ABCMeta):
    """Base backbone.

    This class defines the basic functions of a backbone. Any backbone that
    inherits this class should at least define its own `forward` function.
    """

    @abstractmethod
    def forward(self, x):
        """Forward function.

        Args:
            x (Tensor | tuple[Tensor]): x could be a torch.Tensor or a tuple of
                torch.Tensor, containing input data for forward computation.
        """

    def train(self, mode=True):
        """Set module status before forward computation.

        Args:
            mode (bool): Whether it is train_mode or test_mode
        """
        super(BaseBackbone, self).train(mode)

class SELayer(nn.Module):
    """Squeeze-and-Excitation Module.

    Args:
        channels (int): The input (and output) channels of the SE layer.
        ratio (int): Squeeze ratio in SELayer, the intermediate channel will be
            ``int(channels/ratio)``. Default: 16.
        conv_cfg (None or dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        act_cfg (dict or Sequence[dict]): Config dict for activation layer.
            If act_cfg is a dict, two activation layers will be configurated
            by this dict. If act_cfg is a sequence of dicts, the first
            activation layer will be configurated by the first dict and the
            second activation layer will be configurated by the second dict.
            Default: (dict(type='ReLU'), dict(type='Sigmoid'))
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmengine.is_tuple_of(act_cfg, dict)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = ConvModule(
            in_channels=channels,
            out_channels=int(channels / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(channels / ratio),
            out_channels=channels,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        out = self.global_avgpool(x)
        out = self.conv1(out)
        out = self.conv2(out)
        return x * out

class InvertedResidual_Fcat3_Fourier(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 global_feature_channels,
                 mid_channels,
                 feat_h,
                 feat_w,
                 kernel_size=3,
                 groups=None,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 with_cp=False):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        act_cfg = copy.deepcopy(act_cfg)
        super().__init__()
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv

        self.feat_h = feat_h
        self.feat_w = feat_w
        self.complex_weight = nn.Parameter(torch.randn(in_channels, feat_h, feat_h // 2 + 1, 2, dtype=torch.float32) * 0.02)
        
        if groups is None:
            groups = mid_channels

        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels
        self.global_feature_channels = global_feature_channels

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        if stride == 1:
            self.depthwise_conv = ConvModule(
                in_channels=mid_channels+global_feature_channels,
                out_channels=mid_channels+global_feature_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups+global_feature_channels,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self.depthwise_conv = ConvModule(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=groups,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)  


        if self.with_se:
            self.se = SELayer(**se_cfg)
        
        if stride == 1:
            self.linear_conv = ConvModule(
                in_channels=mid_channels+global_feature_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        else:
            self.linear_conv = ConvModule(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)            

    def forward(self, x, global_feature_map):

        def _inner_forward(x):
            out = x

            # fourier transform
            out = torch.fft.rfft2(out, s=(self.feat_h, self.feat_w), dim=(2, 3), norm='ortho')
            weight = torch.view_as_complex(self.complex_weight)
            out = out * weight
            out = torch.fft.irfft2(out, s=(self.feat_h, self.feat_w), dim=(2, 3), norm='ortho')

            out = out + x

            if self.with_expand_conv:
                out = self.expand_conv(out)

            if x.shape[2] == global_feature_map.shape[2]:
                out = torch.cat((out, global_feature_map), dim=1)
            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + out
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out   

class MGFENet(BaseBackbone):
    # Parameters to build each block:
    #     [kernel size, mid channels, out channels, with_se, act type, stride]
    arch_settings = {
        'small': [[3, 16, 16, True, 'ReLU', 2],
                  [3, 72, 24, False, 'ReLU', 2],
                  [3, 88, 24, False, 'ReLU', 1],
                  [5, 96, 40, True, 'HSwish', 2],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 240, 40, True, 'HSwish', 1],
                  [5, 120, 48, True, 'HSwish', 1],
                  [5, 144, 48, True, 'HSwish', 1],
                  [5, 288, 96, True, 'HSwish', 2],
                  [5, 576, 96, True, 'HSwish', 1],
                  [5, 576, 96, True, 'HSwish', 1]],
        'big': [[3, 16, 16, False, 'ReLU', 1, 128, 3],
                [3, 64, 24, False, 'ReLU', 2, 128, 19],
                [3, 72, 24, False, 'ReLU', 1, 64, 19],
                [5, 72, 40, True, 'ReLU', 2, 64, 43],
                [5, 120, 40, True, 'ReLU', 1, 32, 43],
                [5, 120, 40, True, 'ReLU', 1, 32, 43],
                [3, 240, 80, False, 'HSwish', 2, 32, 83],
                [3, 200, 80, False, 'HSwish', 1, 16, 83],
                [3, 184, 80, False, 'HSwish', 1, 16, 83],
                [3, 184, 80, False, 'HSwish', 1, 16, 83],
                [3, 480, 112, True, 'HSwish', 1, 16, 83],
                [3, 672, 112, True, 'HSwish', 1, 16, 83],
                [5, 672, 160, True, 'HSwish', 1, 16, 83],
                [5, 672, 160, True, 'HSwish', 2, 16, 243],
                [5, 960, 160, True, 'HSwish', 1, 8, 243]],
        'big_7x7': [[7, 16, 16, False, 'ReLU', 1, 128, 3],
                    [7, 64, 24, False, 'ReLU', 2, 128, 19],
                    [7, 72, 24, False, 'ReLU', 1, 64, 19],
                    [7, 72, 40, True, 'ReLU', 2, 64, 43],
                    [7, 120, 40, True, 'ReLU', 1, 32, 43],
                    [7, 120, 40, True, 'ReLU', 1, 32, 43],
                    [7, 240, 80, False, 'HSwish', 2, 32, 83],
                    [7, 200, 80, False, 'HSwish', 1, 16, 83],
                    [7, 184, 80, False, 'HSwish', 1, 16, 83],
                    [7, 184, 80, False, 'HSwish', 1, 16, 83],
                    [7, 480, 112, True, 'HSwish', 1, 16, 83],
                    [7, 672, 112, True, 'HSwish', 1, 16, 83],
                    [7, 672, 160, True, 'HSwish', 1, 16, 83],
                    [7, 672, 160, True, 'HSwish', 2, 16, 243],
                    [7, 960, 160, True, 'HSwish', 1, 8, 243]],
        'large_7x7': [[7, 32, 32, False, 'ReLU', 1, 128, 3],
                    [7, 128, 48, False, 'ReLU', 2, 128, 35],
                    [7, 144, 48, False, 'ReLU', 1, 64, 35],
                    [7, 144, 80, True, 'ReLU', 2, 64, 83],
                    [7, 240, 80, True, 'ReLU', 1, 32, 83],
                    [7, 240, 80, True, 'ReLU', 1, 32, 83],
                    [7, 480, 160, False, 'HSwish', 2, 32, 163],
                    [7, 400, 160, False, 'HSwish', 1, 16, 163],
                    [7, 368, 160, False, 'HSwish', 1, 16, 163],
                    [7, 368, 160, False, 'HSwish', 1, 16, 163],
                    [7, 960, 224, True, 'HSwish', 1, 16, 163],
                    [7, 1344, 224, True, 'HSwish', 1, 16, 163],
                    [7, 1344, 320, True, 'HSwish', 1, 16, 163],
                    [7, 1344, 320, True, 'HSwish', 2, 16, 483],
                    [7, 1920, 320, True, 'HSwish', 1, 8, 483]],
    }  # yapf: disable

    def __init__(self,
                 arch='small',
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 out_indices=(-1, ),
                 frozen_stages=-1,
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer=['Conv2d']),
                     dict(type='Constant', val=1, layer=['_BatchNorm'])
                 ]):
        # Protect mutable default arguments
        norm_cfg = copy.deepcopy(norm_cfg)
        super().__init__(init_cfg=init_cfg)
        assert arch in self.arch_settings
        for index in out_indices:
            if index not in range(-len(self.arch_settings[arch]),
                                  len(self.arch_settings[arch])):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, {len(self.arch_settings[arch])}). '
                                 f'But received {index}')

        if frozen_stages not in range(-1, len(self.arch_settings[arch])):
            raise ValueError('frozen_stages must be in range(-1, '
                             f'{len(self.arch_settings[arch])}). '
                             f'But received {frozen_stages}')
        self.arch = arch
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        if arch == 'big' or arch=='big_7x7' or arch == 'small':
            self.in_channels = 16
        elif arch =='large_7x7':
            self.in_channels = 32
        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='HSwish'))

        self.layers = self._make_layer()
        self.layers_len = len(self.layers)
        # self.feat_dim = self.arch_settings[arch][-1][2]
        # if arch == 'big':
        if arch == 'big' or arch=='big_7x7' or arch == 'small':
            self.feat_dim = 3 + 16 + 24 + 40 + 160 + 160
        elif arch =='large_7x7':
            self.feat_dim = 3 + 32 + 48 + 80 + 320 + 320
        self.conv2 = ConvModule(
            in_channels=self.feat_dim,
            out_channels=self.arch_settings[arch][-1][1],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )

        self.avg_pool_list = []
        for i in range(5):
            self.avg_pool_list.append(nn.AvgPool2d(3, stride=2, padding=1))
        

    def _make_layer(self):
        layers = []
        layer_setting = self.arch_settings[self.arch]
        add_feature_channels = 3
        for i, params in enumerate(layer_setting):
            (kernel_size, mid_channels, out_channels, with_se, act,
             stride, feat_size, global_feature_channels) = params
            if with_se:
                if stride == 1:
                    se_cfg = dict(
                        channels=mid_channels+global_feature_channels,
                        ratio=4,
                        act_cfg=(dict(type='ReLU'),
                                dict(type='HSigmoid', bias=1.0, divisor=2.0)))
                else:
                    se_cfg = dict(
                        channels=mid_channels,
                        ratio=4,
                        act_cfg=(dict(type='ReLU'),
                                dict(type='HSigmoid', bias=1.0, divisor=2.0)))
            else:
                se_cfg = None

            layer = InvertedResidual_Fcat3_Fourier(
                in_channels=self.in_channels,
                out_channels=out_channels,
                global_feature_channels = global_feature_channels,
                mid_channels=mid_channels,
                feat_h=feat_size,
                feat_w=feat_size,
                kernel_size=kernel_size,
                stride=stride,
                se_cfg=se_cfg,
                with_expand_conv=True,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=dict(type=act),
                with_cp=self.with_cp)
            self.in_channels = out_channels
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, layer)
            layers.append(layer_name)
        return layers

    def forward(self, x):
        # layers_map = []
        # layers_map.append(x)    # [B, 3, 256, 256]
        downsample_count = 0
        global_feature_map = self.avg_pool_list[downsample_count](x)
        downsample_count += 1

        x = self.conv1(x)
        # layers_map.append(x)    # [B, 16, 128, 128]
        # feature_map_list.append(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            # print(global_feature_map.shape)
            x = layer(x, global_feature_map)
            
            if i+1 < self.layers_len:
                if self.arch_settings[self.arch][i+1][-3] == 2:
                    global_feature_map = torch.cat((global_feature_map, x), dim=1)
                    global_feature_map = self.avg_pool_list[downsample_count](global_feature_map)
                    downsample_count +=1 

            if i in self.out_indices or \
                    i - len(self.layers) in self.out_indices:
                outs.append(x)

        global_feature_map = torch.cat((global_feature_map, x), dim=1)

        # add head
        final_output = self.conv2(global_feature_map)
        outs.append(final_output)

        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
