# SPDX-License-Identifier: AGPL-3.0-only
"""MSC-RepViT: 3/6/9/3 stages and per-scale MFG.

Four-stage feature extraction with an independent MFG at each output scale.
"""
import math
import copy
import torch
from torch import nn

__all__ = ['MSCRepViT', 'MFG']

class SqueezeExcite(nn.Module):
    """Squeeze-excitation with reduction ratio .25 and channel divisor 8."""
    def __init__(self, channels, rd_ratio=.25):
        super().__init__()
        rd_channels = max(8, int(channels * rd_ratio + 4) // 8 * 8)
        self.fc1 = nn.Conv2d(channels, rd_channels, 1, bias=True)
        self.bn = nn.Identity()
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, 1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.act(self.bn(self.fc1(x_se)))
        return x * self.gate(self.fc2(x_se))

def replace_batchnorm(net):
    raise NotImplementedError('replace_batchnorm is scheduled for a later release; see RELEASE_SCOPE.md.')

def _make_divisible(v, divisor, min_value=None):
    """
    It ensures that all layers have a channel number that is divisible by 8
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse_self(self):
        raise NotImplementedError('Conv2d_BN.fuse_self is scheduled for a later release; see RELEASE_SCOPE.md.')

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse_self(self):
        if self.training:
            raise ValueError('Call eval() before deployment conversion')
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse_self()
        elif isinstance(self.m, nn.Conv2d):
            m = copy.deepcopy(self.m)
        else:
            return self
        kh, kw = m.kernel_size
        if (m.in_channels != m.out_channels or m.stride != (1, 1)
                or kh % 2 == 0 or kw % 2 == 0
                or m.padding != (m.dilation[0] * (kh // 2), m.dilation[1] * (kw // 2))):
            return self
        identity = torch.zeros_like(m.weight)
        output = torch.arange(m.out_channels, device=m.weight.device)
        identity[output, output % (m.in_channels // m.groups), kh // 2, kw // 2] = 1
        m.weight.add_(identity)
        return m.eval()

class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse_self(self):
        conv = self.conv.fuse_self()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device, dtype=conv1_w.dtype), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv.eval()

class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

class RepViT(nn.Module):
    def __init__(self, cfgs):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        # Infer stage boundaries from the graph, without an allocation or BN update.
        stride = 4
        endpoints = {}
        channels = {}
        for index, (k, t, c, use_se, use_hs, step) in enumerate(self.cfgs, 1):
            stride *= step
            endpoints[stride] = index
            channels[stride] = _make_divisible(c, 8)
        if set(endpoints) != {4, 8, 16, 32}:
            raise ValueError('Expected backbone stages at strides 4, 8, 16 and 32')
        self.output_indices = tuple(endpoints[s] for s in (4, 8, 16, 32))
        self.channel = [channels[s] for s in (4, 8, 16, 32)]
        
    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 3 or min(x.shape[-2:]) < 1:
            raise ValueError('Expected a nonempty Bx3xHxW image tensor')
        features = []
        for index, layer in enumerate(self.features):
            x = layer(x)
            if index in self.output_indices:
                features.append(x)
        return features
    
    def switch_to_deploy(self):
        replace_batchnorm(self)


class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

class MFG(nn.Module):
    """MFG channel gating with pooled descriptors and learnable mixing."""
    def __init__(self, channel, b=1, gamma=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global spatial pooling.
        # Shared one-dimensional convolution along the channel descriptor.
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k / 2), bias=False)
        self.fc = nn.Conv2d(channel, channel, 1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.mix = Mix()

    def forward(self, input):
        x = self.avg_pool(input)
        x1 = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2)#(1,64,1)
        x2 = self.fc(x).squeeze(-1).transpose(-1, -2)#(1,1,64)
        out1 = torch.sum(torch.matmul(x1,x2),dim=1).unsqueeze(-1).unsqueeze(-1)#(1,64,1,1)
        #x1 = x1.transpose(-1, -2).unsqueeze(-1)
        out1 = self.sigmoid(out1)
        out2 = torch.sum(torch.matmul(x2.transpose(-1, -2),x1.transpose(-1, -2)),dim=1).unsqueeze(-1).unsqueeze(-1)

        #out2 = self.fc(x)
        out2 = self.sigmoid(out2)
        out = self.mix(out1,out2)
        out = self.conv1(out.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        out = self.sigmoid(out)

        return input*out

class MSCRepViT(nn.Module):
    """Configurable stage backbone followed by one MFG at each output scale.

    Returns four features in stride order (4, 8, 16, 32). The
    graph cache reserves a leading None slot; that is an integration detail,
    not a fifth feature generated by this backbone. Stage channels are inferred from the graph without a random forward pass.
    """
    def __init__(self, depths=(3, 6, 9, 3)):
        super().__init__()
        if len(depths) != 4 or any(type(d) is not int or d < 1 for d in depths):
            raise ValueError('depths must contain four positive integers')
        self.depths = tuple(depths)
        cfg = []
        for stage, count in enumerate(depths):
            for i in range(count):
                se = int(i == 0) if stage == 0 else int(i > 0 and i % 2 == 1)
                cfg.append([3, 2, (48,96,192,384)[stage], se, int(stage >= 2), 2 if stage > 0 and i == 0 else 1])
        self.backbone = RepViT(cfg)
        self.mfg = nn.ModuleList(MFG(c) for c in self.backbone.channel)
        self.channel = list(self.backbone.channel)

    def forward(self, x):
        features = self.backbone(x)
        return [gate(feature) for gate, feature in zip(self.mfg, features)]

    def switch_to_deploy(self):
        self.backbone.switch_to_deploy()

