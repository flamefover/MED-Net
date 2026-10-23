# SPDX-License-Identifier: AGPL-3.0-only
"""UDEH: unified detail enhancement detection head.

"""
import math
import torch
from torch import nn

__all__ = ["UDEH"]



def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Converts per-side discrete distributions into expected distances.
    """

    def __init__(self, c1=16):
        raise NotImplementedError('DFL.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x):
        raise NotImplementedError('DFL.forward is scheduled for a later release; see RELEASE_SCOPE.md.')

def make_anchors(feats, strides, grid_cell_offset=0.5):
    raise NotImplementedError('make_anchors is scheduled for a later release; see RELEASE_SCOPE.md.')

def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    raise NotImplementedError('dist2bbox is scheduled for a later release; see RELEASE_SCOPE.md.')

class Scale(nn.Module):
    """A learnable scale parameter.

    This layer scales the input by a learnable factor. It multiplies a
    learnable scale parameter of shape (1,) with input of any shape.

    Args:
        scale (float): Initial value of scale factor. Default: 1.0
    """

    def __init__(self, scale: float = 1.0):
        raise NotImplementedError('Scale.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError('Scale.forward is scheduled for a later release; see RELEASE_SCOPE.md.')

class Conv_GN(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        raise NotImplementedError('Conv_GN.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.gn(self.conv(x)))

class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        raise NotImplementedError('Conv2d_cd.__init__ is scheduled for a later release; see RELEASE_SCOPE.md.')
    
    def get_weight(self):
        raise NotImplementedError('Conv2d_cd.get_weight is scheduled for a later release; see RELEASE_SCOPE.md.')


class Conv2d_hd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=1.0):

        if kernel_size != 3:
            raise ValueError('Detail difference convolutions require kernel_size=3')
        super(Conv2d_hd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def get_weight(self):
        raise NotImplementedError('Conv2d_hd.get_weight is scheduled for a later release; see RELEASE_SCOPE.md.')

class Conv2d_vd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        if kernel_size != 3:
            raise ValueError('Detail difference convolutions require kernel_size=3')
        super(Conv2d_vd, self).__init__() 
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    
    def get_weight(self):
        conv_weight = self.conv.weight
        conv_shape = conv_weight.shape
        conv_weight_vd = conv_weight.new_zeros(conv_shape[0], conv_shape[1], 9)
        conv_weight_vd[:, :, [0, 1, 2]] = conv_weight[:, :, :]
        conv_weight_vd[:, :, [6, 7, 8]] = -conv_weight[:, :, :]
        conv_weight_vd = conv_weight_vd.reshape(conv_shape[0], conv_shape[1], conv_shape[2], conv_shape[2])
        return conv_weight_vd, self.conv.bias

class DEConv(nn.Module):
    def __init__(self, dim):
        super(DEConv, self).__init__()
        self.conv1_1 = Conv2d_cd(dim, dim, 3, bias=True)
        self.conv1_2 = Conv2d_hd(dim, dim, 3, bias=True)
        self.conv1_3 = Conv2d_vd(dim, dim, 3, bias=True)
        self.conv1_5 = nn.Conv2d(dim, dim, 3, padding=1, bias=True)
        
        self.bn = nn.BatchNorm2d(dim)
        self.act = nn.SiLU()

    def forward(self, x):
        if hasattr(self, 'conv1_1'):
            w1, b1 = self.conv1_1.get_weight()
            w2, b2 = self.conv1_2.get_weight()
            w3, b3 = self.conv1_3.get_weight()
            w5, b5 = self.conv1_5.weight, self.conv1_5.bias

            w = w1 + w2 + w3 + w5
            b = b1 + b2 + b3 + b5
            res = nn.functional.conv2d(input=x, weight=w, bias=b, stride=1, padding=1, groups=1)
        else:
            res = self.conv1_5(x)
            
        if hasattr(self, 'bn'):
            res = self.bn(res)
        
        return self.act(res)
    
    def switch_to_deploy(self):
        if self.training:
            raise ValueError('Call eval() before deployment conversion')
        if not hasattr(self, 'conv1_1'):
            return self
        w1, b1 = self.conv1_1.get_weight()
        w2, b2 = self.conv1_2.get_weight()
        w3, b3 = self.conv1_3.get_weight()
        w5, b5 = self.conv1_5.weight, self.conv1_5.bias

        self.conv1_5.weight = torch.nn.Parameter(w1 + w2 + w3 + w5)
        self.conv1_5.bias = torch.nn.Parameter(b1 + b2 + b3 + b5)
        
        del self.conv1_1
        del self.conv1_2
        del self.conv1_3
        return self

class DEConv_GN(DEConv):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    def __init__(self, dim):
        super().__init__(dim)
        
        self.bn = nn.GroupNorm(16, dim)

class UDEH(nn.Module):
    """Unified Detail Enhancement Head (UDEH).

    Args:
        nc: Number of classes.
        hidc: Shared feature width; must be divisible by 16 for GroupNorm.
        ch: Input channel widths, in increasing detection-stride order.

    Forward updates the supplied feature list in place. In training it returns raw tensors [B, 64 + nc, H_i, W_i]. In
    evaluation it returns (predictions, raw), where predictions are
    [B, 4 + nc, sum(H_i * W_i)] with xywh boxes and class probabilities.
    With export=True it returns predictions alone (no NMS).
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = "onnx"  # explicit export default
    shape = None

    def __init__(self, nc=80, hidc=256, ch=()):
        """Initialize the UDEH head with depthwise input projections."""
        super().__init__()
        if type(nc) is not int or nc < 1 or not ch or hidc < 16 or hidc % 16:
            raise ValueError('Positive nc, nonempty channels, and hidc divisible by 16 are required')
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.register_buffer('stride', torch.zeros(self.nl), persistent=False)
        self.register_buffer('anchors', torch.empty(0), persistent=False)
        self.register_buffer('strides', torch.empty(0), persistent=False)
        self.conv = nn.ModuleList(nn.Sequential(nn.Conv2d(x, x, 3, padding=1, groups=x, bias=False), Conv_GN(x, hidc, 1)) for x in ch)
        self.share_conv = nn.Sequential(DEConv_GN(hidc), DEConv_GN(hidc))
        self.cv2 = nn.Conv2d(hidc, 4 * self.reg_max, 1)
        self.cv3 = nn.Conv2d(hidc, self.nc, 1)
        self.scale = nn.ModuleList(Scale(1.0) for x in ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = self.conv[i](x[i])
            x[i] = self.share_conv(x[i])
            x[i] = torch.cat((self.scale[i](self.cv2(x[i])), self.cv3(x[i])), 1)
        if self.training:  # Training path
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in ("saved_model", "pb", "tflite", "edgetpu", "tfjs"):  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
        dbox = self.decode_bboxes(box)

        if self.export and self.format in ("tflite", "edgetpu"):
            # Precompute normalization factor to increase numerical stability
            img_h = shape[2]
            img_w = shape[3]
            img_size = torch.tensor([img_w, img_h, img_w, img_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * img_size)
            dbox = dist2bbox(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2], xywh=True, dim=1)

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        # for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
        m.cv2.bias.data[:] = 1.0  # box
        m.cv3.bias.data[: m.nc] = math.log(5 / m.nc / (640 / 16) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes):
        """Decode bounding boxes."""
        return dist2bbox(self.dfl(bboxes), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
