"""MED-Net model graph and multiscale detection forward pass.

This is a model-only runtime, not a replacement for the training framework.
"""
import math
from pathlib import Path

import torch
from torch import nn
import yaml

from .modules.msc_repvit import MSCRepViT
from .modules.amdf import AMDFFusion, AMDFDownsample
from .modules.udeh import UDEH
from .modules.common import SPPF, C2PSA
from .frj import select_codes


class Concat(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, xs):
        return torch.cat(xs, self.dim)


class MEDNet(nn.Module):
    def __init__(self, nc=None, scale=None, config=None, occupancy=None, route_mode='on'):
        super().__init__()
        config = Path(config) if config else Path(__file__).parent/'configs'/'mednet.yaml'
        self.config = yaml.safe_load(config.read_text(encoding='utf-8'))
        self.nc = self.config['nc'] if nc is None else int(nc)
        if self.nc < 1:
            raise ValueError('nc must be positive')
        self.scale = scale or self.config['scale']
        depth, width, cap = self.config['scales'][self.scale]
        scaled = lambda c: math.ceil(min(c, cap)*width/8)*8
        positions = [6+i for i, row in enumerate(self.config['layers']) if row[2] == 'AMDFFusion']
        if route_mode == 'on':
            if occupancy is None:
                raise ValueError('FRJ ON requires an initialization occupancy ratio')
            codes = select_codes(occupancy, self.config['frj_priority'], positions)
        elif route_mode in ('wide', 'deep'):
            codes = {p: int(route_mode == 'deep') for p in positions}
        else:
            raise ValueError('route_mode must be on, wide, or deep')
        self.route_codes = tuple((p, codes[p]) for p in positions)
        self.backbone = MSCRepViT(self.config['stage_depths'])
        channels = [0, *self.backbone.channel, self.backbone.channel[-1]]
        self.layers = nn.ModuleList()
        self.sources = []
        for index, (source, repeats, name, args) in enumerate(self.config['layers'], 6):
            n = max(round(repeats*depth), 1) if repeats > 1 else repeats
            c1 = channels[source] if isinstance(source, int) else sum(channels[i] for i in source)
            if name == 'SPPF':
                c2 = scaled(args[0])
                module = SPPF(c1, c2, args[1])
            elif name == 'C2PSA':
                c2 = scaled(args[0])
                module = C2PSA(c1, c2, n)
            elif name == 'AMDFFusion':
                c2 = scaled(args[0])
                module = AMDFFusion(c1, c2, n, judgment_code=codes[index])
            elif name == 'AMDFDownsample':
                c2 = c1
                module = AMDFDownsample(c1)
            elif name == 'Upsample':
                c2 = c1
                module = nn.Upsample(scale_factor=args[0], mode=args[1])
            elif name == 'Concat':
                c2 = c1
                module = Concat(args[0])
            elif name == 'UDEH':
                c2 = c1
                module = UDEH(self.nc, scaled(args[0]), [channels[i] for i in source])
                module.stride = torch.tensor([8., 16., 32.])
                module.bias_init()
            else:
                raise ValueError(f'Unsupported module: {name}')
            self.layers.append(module)
            self.sources.append(source)
            channels.append(c2)
        # Initialize normalization and activation settings.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps, module.momentum = 1e-3, 0.03
            elif isinstance(module, (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU)):
                module.inplace = True

    def _apply(self, fn):
        raise NotImplementedError('MEDNet._apply is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x):
        raise NotImplementedError('MEDNet.forward is scheduled for a later release; see RELEASE_SCOPE.md.')
