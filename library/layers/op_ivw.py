import torch
import torch.nn as nn 
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
TORCH_MAJOR = int(torch.__version__.split('.')[0])
TORCH_MINOR = int(torch.__version__.split('.')[1])
if TORCH_MAJOR == 1 and TORCH_MINOR < 8:
    from torch._six import container_abcs
else:
    import collections.abc as container_abcs
from itertools import repeat

class ClampFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, bound, nbit):
        boundpos  = bound*(2.0**(nbit-1)-1)/(2.0**(nbit-1))
        temp_loc1 = data>boundpos
        temp_loc2 = data<-bound
        ctx.save_for_backward(temp_loc1, temp_loc2)
        return data.clamp(-bound, boundpos)

    @staticmethod
    def backward(ctx, grad_output):
        temp_loc1, temp_loc2   = ctx.saved_tensors
        grad_output[temp_loc1] = 0.01
        grad_output[temp_loc2] = -0.01

        return grad_output, None, None

class QuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, data, nbit=4, abs_max=1):
        bound         = 2.0**(nbit-1)
        scale         = abs_max / bound
        interger_data = (data / scale).floor()#.clamp(-bound, bound)
        data          = interger_data * scale

        return data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

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

class _ConvNd(nn.Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        super(_ConvNd, self).__init__()
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
        if transposed:
            self.weight = Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight = Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
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
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class _ConvTransposeMixin(object):
    def _output_padding(self, input, output_size, stride, padding, kernel_size):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                    .format(k, k + 2, len(output_size)))

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(k):
                dim_size = ((input.size(d + 2) - 1) * stride[d] -
                            2 * padding[d] + kernel_size[d])
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                            output_size, min_sizes, max_sizes, input.size()[2:]))

            res = torch.jit.annotate(List[int], [])
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

class IvwConv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                clamp=False, quant=False,
                w_clamp_val=0.25, b_clamp_val=8, a_clamp_val=8, 
                w_quant_bit=4, b_quant_bit=16, a_quant_bit=8,name='Null'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IvwConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.is_bias = bias
        self.clamp   = clamp
        self.quant   = quant

        self.w_clamp_val = w_clamp_val
        self.b_clamp_val = b_clamp_val
        self.a_clamp_val = a_clamp_val

        self.w_quant_bit = w_quant_bit
        self.b_quant_bit = b_quant_bit
        self.a_quant_bit = a_quant_bit

        self.name        = name

    def conv2d_forward(self, input, weight):
        bias = self.bias

        if self.clamp:
            weight   = ClampFunction.apply(weight, self.w_clamp_val, self.w_quant_bit)
            if self.is_bias:
                bias = ClampFunction.apply(bias, self.b_clamp_val, self.b_quant_bit)
            input    = input.clamp(-self.a_clamp_val, self.a_clamp_val*(2.0**(self.a_quant_bit-1)-1.0)/(2.0**(self.a_quant_bit-1)))

        if self.quant:
            weight   = QuantFunction.apply(weight, self.w_quant_bit, self.w_clamp_val)
            if self.is_bias:
                bias = QuantFunction.apply(bias, self.b_quant_bit, self.b_clamp_val)
            input    = QuantFunction.apply(input, self.a_quant_bit, self.a_clamp_val) 

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)

        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

class IvwConvTranspose2d(_ConvTransposeMixin, _ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', 
                clamp=False, quant=False,
                w_clamp_val=0.25, b_clamp_val=8, a_clamp_val=8, w_quant_bit=4, b_quant_bit=16, a_quant_bit=8,):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        output_padding = _pair(output_padding)
        super(IvwConvTranspose2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

        self.clamp = clamp
        self.quant = quant

        self.w_clamp_val = w_clamp_val
        self.b_clamp_val = b_clamp_val
        self.a_clamp_val = a_clamp_val

        self.w_quant_bit = w_quant_bit
        self.b_quant_bit = b_quant_bit
        self.a_quant_bit = a_quant_bit

    def forward(self, input, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self._output_padding(input, output_size, self.stride, self.padding, self.kernel_size)

        weight         = self.weight
        bias           = self.bias

        if self.clamp:
            weight = ClampFunction.apply(weight, self.w_clamp_val, self.w_quant_bit)
            bias   = ClampFunction.apply(bias, self.b_clamp_val, self.b_quant_bit)
            input  = input.clamp(-self.a_clamp_val, self.a_clamp_val*(2.0**(self.a_quant_bit-1)-1.0)/(2.0**(self.a_quant_bit-1)))

        if self.quant:
            weight = QuantFunction.apply(weight, self.w_quant_bit, self.w_clamp_val)
            bias   = QuantFunction.apply(bias, self.b_quant_bit, self.b_clamp_val)
            input  = QuantFunction.apply(input, self.a_quant_bit, self.a_clamp_val) 

        return F.conv_transpose2d(
            input, weight, bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)

class IvwConv2dFixout(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', 
                clamp=False, quant=False,
                w_clamp_val=0.25, b_clamp_val=8, a_clamp_val=8, o_clamp_val=32, 
                w_quant_bit=4, b_quant_bit=16, a_quant_bit=8, o_quant_bit=16, name='Null'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(IvwConv2dFixout, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self.is_bias = bias
        self.clamp   = clamp
        self.quant   = quant

        self.w_clamp_val = w_clamp_val
        self.b_clamp_val = b_clamp_val
        self.a_clamp_val = a_clamp_val
        self.o_clamp_val = o_clamp_val

        self.w_quant_bit = w_quant_bit
        self.b_quant_bit = b_quant_bit
        self.a_quant_bit = a_quant_bit
        self.o_quant_bit = o_quant_bit

        self.name        = name

    def conv2d_forward(self, input, weight):
        bias = self.bias

        if self.clamp:
            weight   = ClampFunction.apply(weight, self.w_clamp_val, self.w_quant_bit)
            if self.is_bias:
                bias = ClampFunction.apply(bias, self.b_clamp_val, self.b_quant_bit)
            input    = input.clamp(-self.a_clamp_val, self.a_clamp_val*(2.0**(self.a_quant_bit-1)-1.0)/(2.0**(self.a_quant_bit-1)))

        if self.quant:
            weight   = QuantFunction.apply(weight, self.w_quant_bit, self.w_clamp_val)
            if self.is_bias:
                bias = QuantFunction.apply(bias, self.b_quant_bit, self.b_clamp_val)
            input    = QuantFunction.apply(input, self.a_quant_bit, self.a_clamp_val) 

        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            out = F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                        weight, bias, self.stride,
                        _pair(0), self.dilation, self.groups)
        else:
            out = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

        if self.clamp:
            out = out.clamp(-self.o_clamp_val, self.o_clamp_val*(2.0**(self.o_quant_bit-1)-1.0)/(2.0**(self.o_quant_bit-1)))
        if self.quant:
            out = QuantFunction.apply(out, self.o_quant_bit, self.o_clamp_val) 

        return out

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

