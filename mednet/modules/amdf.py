# SPDX-License-Identifier: AGPL-3.0-only
"""AMDF downsampling and fusion modules for MED-Net.

"""

import torch
from torch import nn
from einops import rearrange

__all__ = ["AMDFDownsample", "AMDFFusion"]


def autopad(k, p=None, d=1):
    raise NotImplementedError('autopad is scheduled for a later release; see RELEASE_SCOPE.md.')


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        raise NotImplementedError('Conv.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x):
        raise NotImplementedError('Conv.forward is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        raise NotImplementedError('Bottleneck.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x):
        raise NotImplementedError('Bottleneck.forward is scheduled for a later release; see RELEASE_SCOPE.md.')




class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        raise NotImplementedError('C3k.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')




class EffectiveSEModule(nn.Module):
    def __init__(self, channels, add_maxpool=False):
        super().__init__()
        self.add_maxpool = add_maxpool
        self.fc = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
        self.gate = nn.Hardsigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc(x_se)
        return x * self.gate(x_se)


class MBConv(nn.Module):
    def __init__(self, inc, ouc, shortcut=True, e=4, dropout=0.1):
        super().__init__()
        midc = inc * e
        self.conv_pw_1 = Conv(inc, midc, 1)
        self.conv_dw_1 = Conv(midc, midc, 3, g=midc)
        self.effective_se = EffectiveSEModule(midc)
        self.conv1 = Conv(midc, ouc, 1, act=False)
        self.dropout = nn.Dropout2d(p=dropout)
        self.add = shortcut and inc == ouc

    def forward(self, x):
        return x + self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x))))) if self.add else self.dropout(self.conv1(self.effective_se(self.conv_dw_1(self.conv_pw_1(x)))))


class C3k_EMBC(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(MBConv(c_, c_, shortcut) for _ in range(n)))


class AMDFFusion(nn.Module):
    """AMDF fusion: split, fixed path, additive Wise-Joint.

    Code 0 uses Efficient Modules; code 1 uses Inverted Modules.
    """
    def __init__(self, c1, c2, n=1, judgment_code=0, e=0.5, g=1, shortcut=True):
        super().__init__()
        if judgment_code not in (0, 1):
            raise ValueError('Judgment Code must be 0 or 1')
        self.judgment_code = int(judgment_code)
        self.c = int(c2 * e)
        self.split_projection = Conv(c1, 2 * self.c, 1)
        self.path = nn.Sequential(*(C3k_EMBC(self.c, self.c, 2, shortcut, g)
            if judgment_code == 0 else MBConv(self.c, self.c, shortcut) for _ in range(n)))
        self.output_projection = Conv(self.c, c2, 1)

    def forward(self, x):
        direct, processed = self.split_projection(x).chunk(2, dim=1)
        return self.output_projection(direct + self.path(processed))


class AMDFDownsample(nn.Module):
    """AMDF weighted downsampling: preserve channels, halve even spatial dimensions.

    The grouped convolution requires ``ch // group`` to be positive
    and divide both ``ch`` and ``4 * ch``. The release keeps these constraints.
    """

    def __init__(self, ch, group=16):
        super().__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.attention = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1), Conv(ch, ch, k=1))
        self.ds_conv = Conv(ch, ch * 4, k=3, s=2, g=(ch // group))

    def forward(self, x):
        att = rearrange(self.attention(x), 'bs ch (s1 h) (s2 w) -> bs ch h w (s1 s2)', s1=2, s2=2)
        att = self.softmax(att)
        x = rearrange(self.ds_conv(x), 'bs (s ch) h w -> bs ch h w s', s=4)
        return torch.sum(x * att, dim=-1)
