import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adc import _adc
from .dac import _quantize_dac
from .w2g import w2g
from .crossbarlayer import CrossbarLayer

quantize_input = _quantize_dac.apply
# quantize_weight = _quantize_dac.apply
adc = _adc.apply


class crxb_Conv2d(nn.Conv2d):
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, crxb_size=64, scaler_dw=1, **crxb_cfg):
        super(crxb_Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)

        assert self.groups == 1, "currently not support grouped convolution for custom conv"

        # self.ir_drop = ir_drop
        # self.device = device

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        # self.enable_ec_SAF = enable_ec_SAF

        # self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad=False)
        self.register_buffer('nchout_index', torch.arange(self.out_channels))

        weight_flatten_rows = self.in_channels * torch.cumprod(torch.tensor(self.kernel_size), 0)[-1].item()
        weight_flatten_cols = self.out_channels
        self.crxb_row, self.crxb_row_pads = self.num_pad(weight_flatten_rows, self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(weight_flatten_cols, self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, 0, 0, self.crxb_row_pads]
        weight_crxb_shape = torch.Size((self.crxb_col, self.crxb_row,
                                        self.crxb_size, self.crxb_size))

        ################# Hardware conversion ##############################
        # weight and input levels
        # Q(x) = (2^{k−1} − 1)* round((2^{k−1} − 1) * x)
        # self.n_lvl = 2 ** quantize
        # self.h_lvl = (self.n_lvl - 2) / 2
        # ReRAM cells
        # 7-bit precisionis achievable on state-of-the-art RRAM device [9]
        # [9] High precision tuning of state for memristive devices by
        # adaptable variation-tolerant algorithm
        # self.Gmax = gmax  # max conductance
        # self.Gmin = gmin  # min conductance
        # self.delta_g = (self.Gmax - self.Gmin) / (2 ** 7)  # conductance step
        # self.crxb = crxb_layer(ir_drop, device, gmax, gmin, gwire, gload, scaler_dw, vdd, enable_noise,
        #                        freq, temp , crxb_size, quantize, enable_SAF, enable_ec_SAF)
        self.crxb = CrossbarLayer(crxb_size=crxb_size, **crxb_cfg)
        # self.Gwire = gwire
        # self.Gload = gload
        # DAC
        # self.Vdd = vdd  # unit: volt
        # self.delta_v = self.Vdd / (self.n_lvl - 1)
        # self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.counter = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))
        self.scaler_dw = scaler_dw
        self.delta_w = 0  # self.weight.abs().max() / self.h_lvl * self.scaler_dw
        self.delta_x = 0  # self.delta_in_sum.data / self.counter.data

        self.h_out = None
        self.w_out = None

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    # Mapping the weights to the crossbar array
    def mapping(self, input):
        # 1. input data and weight quantization
        # delta_x delta_w
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.crxb.h_lvl * self.scaler_dw

            # trainable delta_x
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.crxb.h_lvl
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data
        input_quan, weight_quan = self.crxb.quantize(input, self.delta_x, self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance
        # compute output feature size
        if self.h_out is None and self.w_out is None:
            self.h_out = int(
                (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
            self.w_out = int(
                (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)

        num_block = self.h_out * self.w_out
        block_size = input.shape[1] * torch.cumprod(torch.tensor(self.kernel_size), 0)[-1]
        pad_block_size = block_size + self.input_pad[2] + self.input_pad[3]

        # 2.1 flatten and unfold the weight and input
        input_unfold = F.unfold(input_quan, kernel_size=self.kernel_size[0],
                                dilation=self.dilation, padding=self.padding,
                                stride=self.stride)
        weight_flatten = weight_quan.view(self.out_channels, -1)

        # 2.2. add paddings
        input_padded = F.pad(input_unfold, self.input_pad,
                             mode='constant', value=0)
        weight_padded = F.pad(weight_flatten, self.w_pad,
                              mode='constant', value=0)

        # 2.3. reshape to crxb size
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, num_block)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.crxb.w2g(weight_crxb)
        return input_crxb, G_crxb

    def forward(self, input):
        assert input.dim() == 4
        # 1. input data and weight quantization
        input_crxb, G_crxb = self.mapping(input)

        # 2. Perform the computation between input voltage and weight conductance
        # 2.1-2.3 Mapping
        # 2.4. compute matrix multiplication
        G_crxb = self.crxb.noise_injection(input_crxb, G_crxb)

        # this block is to calculate the ir drop of the crossbar
        output_crxb = self.crxb.solve_crxb(input_crxb, G_crxb)

        # 3. perform ADC operation (i.e., current to digital conversion)
        # input.shape[0], self.crxb_col, self.crxb_row,
        # self.crxb_size, self.num_block
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / (self.crxb.h_lvl)
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * \
                           self.delta_i / (self.crxb.delta_v * self.crxb.delta_g)
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)

        output_adc = self.crxb.output_convet(output_crxb, self.delta_i, self.delta_y)

        if self.crxb.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y / self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb)
                               - torch.matmul(G_neg_diff, input_crxb)) * ec_scale

        output_sum = torch.sum(output_adc, dim=2)
        # input.shape[0], self.crxb_col,
        # self.crxb_size, self.num_block
        output = output_sum.view(input.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2],
                                 self.h_out,
                                 self.w_out).index_select(dim=1, index=self.nchout_index)  # remove the padded columns

        if self.bias is not None:
            output += self.bias.unsqueeze(1).unsqueeze(1)

        return output

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0


class crxb_Linear(nn.Linear):
    """
    This is the custom linear layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        scaler_dw(float): weight quantization scaler to reduce the influence of the ir drop.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
    """

    def __init__(self, in_features, out_features,
                 bias=True, crxb_size=64, scaler_dw=1, **crxb_cfg):
        super(crxb_Linear, self).__init__(in_features, out_features, bias)

        # self.ir_drop = ir_drop
        # self.device = device
        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        # self.enable_ec_SAF = enable_ec_SAF

        # self.out_index = nn.Parameter(torch.arange(out_features), requires_grad=False)
        self.register_buffer('out_index', torch.arange(out_features))

        self.crxb_row, self.crxb_row_pads = self.num_pad(
            self.weight.shape[1], self.crxb_size)
        self.crxb_col, self.crxb_col_pads = self.num_pad(
            self.weight.shape[0], self.crxb_size)
        # p3d = (0, 1, 2, 1, 3, 3) # pad by (0, 1), (2, 1), and (3, 3)
        self.w_pad = [0, self.crxb_row_pads, 0, self.crxb_col_pads]
        self.input_pad = [0, self.crxb_row_pads]
        weight_crxb_shape = torch.Size((self.crxb_col, self.crxb_row,
                                        self.crxb_size, self.crxb_size))

        ################# Hardware conversion ##############################
        # weight and input levels
        # Q(x) = (2^{k−1} − 1)* round((2^{k−1} − 1) * x)
        # self.n_lvl = 2 ** quantize
        # self.h_lvl = (self.n_lvl - 2) / 2
        # ReRAM cells
        # 7-bit precisionis achievable on state-of-the-art RRAM device [9]
        # [9] High precision tuning of state for memristive devices by
        # adaptable variation-tolerant algorithm
        # self.Gmax = gmax  # max conductance
        # self.Gmin = gmin  # min conductance
        # self.delta_g = (self.Gmax - self.Gmin) / (2 ** 7)  # conductance step
        # self.crxb = crxb_layer(ir_drop, device, gmax, gmin, gwire, gload, scaler_dw, vdd, enable_noise,
        #                        freq, temp , crxb_size, quantize, enable_SAF, enable_ec_SAF)
        self.crxb = CrossbarLayer(crxb_size=crxb_size, **crxb_cfg)
        # self.Gwire = gwire
        # self.Gload = gload
        # DAC
        # self.Vdd = vdd  # unit: volt
        # self.delta_v = self.Vdd / (self.n_lvl - 1)
        # self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.counter = nn.Parameter(torch.Tensor(1), requires_grad=False)
        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))
        self.scaler_dw = scaler_dw
        self.delta_w = 0  # self.weight.abs().max() / self.h_lvl * self.scaler_dw
        self.delta_x = 0  # self.delta_in_sum.data / self.counter.data

    def num_pad(self, source, target):
        crxb_index = math.ceil(source / target)
        num_padding = crxb_index * target - source
        return crxb_index, num_padding

    # Mapping the weights to the crossbar array
    def mapping(self, input):
        # 1. input data and weight quantization
        # delta_x delta_w
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.crxb.h_lvl * self.scaler_dw

            # trainable delta_x
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.crxb.h_lvl
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        input_quan, weight_quan = self.crxb.quantize(input, self.delta_x, self.weight, self.delta_w)

        # 2. Perform the computation between input voltage and weight conductance

        # 2.1. skip the input unfold and weight flatten for fully-connected layers
        # 2.2. add padding
        input_padded = F.pad(input_quan, self.input_pad,
                             mode='constant', value=0)
        weight_padded = F.pad(weight_quan, self.w_pad,
                              mode='constant', value=0)
        # 2.3. reshape
        input_crxb = input_padded.view(input.shape[0], 1, self.crxb_row,
                                       self.crxb_size, 1)
        weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
                                         self.crxb_row, self.crxb_size).transpose(1, 2)
        # convert the floating point weight into conductance pair values
        G_crxb = self.crxb.w2g(weight_crxb)
        return input_crxb, G_crxb

    def forward(self, input):
        assert input.dim() == 2
        # 1. input data and weight quantization
        input_crxb, G_crxb = self.mapping(input)

        # 2. Perform the computation between input voltage and weight conductance
        # 2.1-2.3 Mapping
        # 2.4. compute matrix multiplication
        G_crxb = self.crxb.noise_injection(input_crxb, G_crxb)

        # this block is to calculate the ir drop of the crossbar
        output_crxb = self.crxb.solve_crxb(input_crxb, G_crxb)

        # 3. perform ADC operation (i.e., current to digital conversion)
        # input.shape[0], self.crxb_col, self.crxb_row,
        # self.crxb_size, self.num_block
        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / (self.crxb.h_lvl)
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * \
                           self.delta_i / (self.crxb.delta_v * self.crxb.delta_g)
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)

        output_adc = self.crxb.output_convet(output_crxb, self.delta_i, self.delta_y)

        if self.crxb.enable_SAF:
            if self.enable_ec_SAF:
                G_pos_diff, G_neg_diff = self.w2g.error_compensation()
                ec_scale = self.delta_y / self.delta_i
                output_adc += (torch.matmul(G_pos_diff, input_crxb)
                               - torch.matmul(G_neg_diff, input_crxb)) * ec_scale

        output_sum = torch.sum(output_adc, dim=2).squeeze(dim=3)
        # input.shape[0], self.crxb_col,
        # self.crxb_size, self.num_block
        output = output_sum.view(input.shape[0],
                                 output_sum.shape[1] * output_sum.shape[2]
                                 ).index_select(dim=1, index=self.out_index)  # remove padded columns

        if self.bias is not None:
            output += self.bias

        return output

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0
