import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _pair
from torch.nn import init
import math
import numpy as np
import time

import ste

OPS = {
  'none' : lambda C, stride, affine: Zero(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C, stride, affine: Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
  'sep_conv_3x3' : lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
}

class LinearShift(nn.Module):
    def __init__(self, in_features, out_features, bias=True, check_grad=False, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
        super(LinearShift, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_kernel = use_kernel
        self.check_grad = check_grad
        self.use_cuda = use_cuda
        self.conc_weight = None
        self.rounding = rounding
        self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits
        
        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        self.shift = nn.Parameter(tensor_constructor(out_features, in_features))
        self.sign = nn.Parameter(tensor_constructor(out_features, in_features), requires_grad = (freeze_sign == False))

        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.shift.data.uniform_(*self.shift_range) 
        self.sign.data.uniform_(-1, 1) 
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        self.shift.data = ste.clamp(self.shift.data, *self.shift_range)
        shift_rounded = ste.round(self.shift, rounding=self.rounding)
        sign_rounded_signed = ste.sign(ste.round(self.sign, rounding=self.rounding))
        weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)

        return torch.nn.functional.linear(input, weight_ps, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class _ConvNdShift(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode, 
                 check_grad=False, freeze_sign=False, rounding='deterministic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
        super(_ConvNdShift, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.rounding=rounding
        self.shift_range = (-1 * (2**(weight_bits - 1) - 2), 0) # we use ternary weights to represent sign
        self.act_integer_bits, self.act_fraction_bits = act_integer_bits, act_fraction_bits

        if check_grad:
            tensor_constructor = torch.DoubleTensor # double precision required to check grad
        else:
            tensor_constructor = torch.Tensor # In PyTorch torch.Tensor is alias torch.FloatTensor

        if transposed:
            self.shift = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size))
            self.sign = nn.Parameter(tensor_constructor(
                in_channels, out_channels // groups, *kernel_size), 
                requires_grad = (freeze_sign == False))
        else:
            self.shift = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size))
            self.sign = nn.Parameter(tensor_constructor(
                out_channels, in_channels // groups, *kernel_size),
                requires_grad = (freeze_sign == False))
        if bias:
            self.bias = nn.Parameter(tensor_constructor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        self.shift.data.uniform_(-10, -1) # (-0.1, 0.1)
        self.sign.data.uniform_(-1, 1) # (-0.1, 0.1)
        
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.shift)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        return s.format(**self.__dict__)

class Conv2dShift(_ConvNdShift):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', 
                 check_grad=False, freeze_sign=False, use_kernel=False, use_cuda=True, rounding='deterministic', 
                 weight_bits=5, act_integer_bits=16, act_fraction_bits=16):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.use_kernel = use_kernel
        self.use_cuda = use_cuda
        self.conc_weight = None
        super(Conv2dShift, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode, 
            check_grad, freeze_sign, rounding,
            weight_bits, act_integer_bits, act_fraction_bits)

    #@weak_script_method
    def forward(self, input):
        self.shift.data = ste.clamp(self.shift.data, *self.shift_range)
        shift_rounded = ste.round(self.shift, self.rounding)
        sign_rounded_signed = ste.sign(ste.round(self.sign, self.rounding))
        weight_ps = ste.unsym_grad_mul(2**shift_rounded, sign_rounded_signed)
        input_fixed_point = ste.round_fixed_point(input, self.act_integer_bits, self.act_fraction_bits)
        if self.bias is not None:
            bias_fixed_point = ste.round_fixed_point(self.bias, self.act_integer_bits, self.act_fraction_bits)
        else:
            bias_fixed_point = None

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)

            input_padded = F.pad(input_fixed_point, expanded_padding, mode='circular')
            padding =  _pair(0)
        else:
            input_padded = input_fixed_point
            padding = self.padding

        return torch.nn.functional.conv2d(input_padded, weight_ps, bias_fixed_point, 
                                              self.stride, padding, self.dilation, self.groups)



class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      Conv2dShift(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
    #   nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      Conv2dShift(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      Conv2dShift(C_in, C_out, kernel_size=1, padding=0, bias=False),
    #   nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
    #   nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      Conv2dShift(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      Conv2dShift(C_in, C_in, kernel_size=1, padding=0, bias=False),
    #   nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
    #   nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      Conv2dShift(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      Conv2dShift(C_in, C_out, kernel_size=1, padding=0, bias=False),
    #   nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
    #   nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x


class Zero(nn.Module):

  def __init__(self, stride):
    super(Zero, self).__init__()
    self.stride = stride

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = Conv2dShift(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = Conv2dShift(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    # self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    # self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

