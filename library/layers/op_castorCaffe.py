import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
from itertools import repeat

def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)

class ClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, bound, nbit):
        boundpos=bound*(2.0**(nbit-1)-1)/(2.0**(nbit-1))
        temp_loc1 = data>boundpos
        temp_loc2 = data<-bound
        ctx.save_for_backward(temp_loc1, temp_loc2)
        return data.clamp(-bound, boundpos)
    @staticmethod
    def backward(ctx, grad_output):
        temp_loc1, temp_loc2 = ctx.saved_tensors
        grad_output[temp_loc1] = 0.01
        grad_output[temp_loc2] = -0.01
        return grad_output, None, None

class QuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, abs_max, nbit):
        if nbit==32:
            return data
        bound = 2.0**(nbit-1)
        scale = abs_max / bound
        # interger_data = (data / scale).round()#.clamp(-bound, bound)
        interger_data = torch.floor(data / scale + 0.5)
        data = interger_data * scale
        return data
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class QuantConv2dCaffe(nn.Conv2d):
    '''
        casual_dim=2, 针对数据(N,C,T,F)，T为时间维度
        casual_dim=3, 针对数据(N,C,F,T)，T为时间维度
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type

        casual: bool = False,
        casual_dim: int = 2,
        right_context: int = 0,

        clamp: bool = False,
        quant: bool = False,
        weight_clamp_val: float = 2.0,
        bias_clamp_val: float = 32.0,
        input_clamp_val: float = 8.0,
        # output_clamp_val: float = 8.0,
        weight_quant_bit: int = 8,
        bias_quant_bit: int = 16,
        input_quant_bit: int = 8,
        # output_quant_bit: int = 8
    ):

        super(QuantConv2dCaffe, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)

        self.casual = casual
        self.pad_h, self.pad_w = self.padding
        self.casual_dim = casual_dim
        if self.casual:
            if self.casual_dim == 2:
                self.padding = (0,self.pad_w)
            elif self.casual_dim == 3:
                self.padding = (self.pad_h,0)
            else:
                raise ValueError('Casual_dim must be 2 or 3 (for an input of {}).'.format(self.casual_dim))
        self.right_context = right_context

        self.clamp = clamp
        self.quant = quant
        self.weight_clamp_val = weight_clamp_val
        self.bias_clamp_val = bias_clamp_val
        self.input_clamp_val = input_clamp_val
        # self.output_clamp_val = output_clamp_val

        self.weight_quant_bit = weight_quant_bit
        self.bias_quant_bit = bias_quant_bit
        self.input_quant_bit = input_quant_bit
        # self.output_quant_bit = output_quant_bit
    
    def forward(self, x):

        if self.casual:
            if self.casual_dim == 2:
                x_left_pad = F.pad(x, (0, 0, 2*self.pad_h-self.right_context, self.right_context), "constant")
            else:
                x_left_pad = F.pad(x, (self.pad_w*2-self.right_context, self.right_context, 0, 0), "constant")
        else:
            x_left_pad = x

        weight = self.weight
        bias = self.bias
        if self.clamp:
            weight = ClampFunction.apply(weight, self.weight_clamp_val, self.weight_quant_bit)
            if bias is not None:
                bias = ClampFunction.apply(bias, self.bias_clamp_val, self.bias_quant_bit)
            x_left_pad = ClampFunction.apply(x_left_pad, self.input_clamp_val, self.input_quant_bit)

        if self.quant:
            weight = QuantFunction.apply(weight, self.weight_clamp_val, self.weight_quant_bit)
            if bias is not None:
                bias = QuantFunction.apply(bias, self.bias_clamp_val, self.bias_quant_bit)
            x_left_pad = QuantFunction.apply(x_left_pad, self.input_clamp_val, self.input_quant_bit)

        if self.padding_mode != 'zeros':
            out = F.conv2d(F.pad(x_left_pad, self._reversed_padding_repeated_twice, mode=self.padding_mode), weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        else:
            out = F.conv2d(x_left_pad, weight, bias, self.stride, self.padding, self.dilation, self.groups)

        # if self.clamp:
        #     out = ClampFunction.apply(out, self.output_clamp_val, self.output_quant_bit)
        # if self.quant:
        #     out = QuantFunction.apply(out, self.output_clamp_val, self.output_quant_bit)

        return out

class QuantConvTranspose2dCaffe(nn.ConvTranspose2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = 'zeros',

        clamp: bool = False,
        quant: bool = False,
        weight_clamp_val: float = 2.0,
        bias_clamp_val: float = 32.0,
        input_clamp_val: float = 8.0,
        # output_clamp_val: float = 8.0,
        weight_quant_bit: int = 8,
        bias_quant_bit: int = 16,
        input_quant_bit: int = 8,
        # output_quant_bit: int = 8
    ):
        super(QuantConvTranspose2dCaffe, self).__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)

        self.clamp = clamp
        self.quant = quant

        self.weight_clamp_val = weight_clamp_val
        self.bias_clamp_val = bias_clamp_val
        self.input_clamp_val = input_clamp_val
        # self.output_clamp_val = output_clamp_val

        self.weight_quant_bit = weight_quant_bit
        self.bias_quant_bit = bias_quant_bit
        self.input_quant_bit = input_quant_bit
        # self.output_quant_bit = output_quant_bit

    def forward(self, input, output_size=None):

        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(
            input, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        weight = self.weight
        bias = self.bias

        if self.clamp:
            weight = ClampFunction.apply(weight, self.weight_clamp_val, self.weight_quant_bit)
            if bias is not None:
                bias = ClampFunction.apply(bias, self.bias_clamp_val, self.bias_quant_bit)
            input = ClampFunction.apply(input, self.input_clamp_val, self.input_quant_bit)

        if self.quant:
            weight = QuantFunction.apply(weight, self.weight_clamp_val, self.weight_quant_bit)
            if bias is not None:
                bias = QuantFunction.apply(bias, self.bias_clamp_val, self.bias_quant_bit)
            input = QuantFunction.apply(input, self.input_clamp_val, self.input_quant_bit)

        out = F.conv_transpose2d(input, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation)

        # if self.clamp:
        #     out = ClampFunction.apply(out, self.output_clamp_val, self.output_quant_bit)
        # if self.quant:
        #     out = QuantFunction.apply(out, self.output_clamp_val, self.output_quant_bit)

        return out
		