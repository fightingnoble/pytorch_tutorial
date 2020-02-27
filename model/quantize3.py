import torch
import torch.nn as nn
import torch.nn.functional as F

from .torx.adc import _adc
from .torx.dac import _quantize_dac

quantize_input = _quantize_dac.apply
quantize_weight = _quantize_dac.apply
adc = _adc.apply


class QConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dw=1):
        super(QConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                      stride, padding, dilation, groups, bias)
        self.input_qbit = input_qbit
        self.weight_qbit = weight_qbit
        self.activation_qbit = activation_qbit
        self.scaler_dw = scaler_dw

        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))

        self.delta_x = self.delta_w = None
        self.delta_i = self.delta_y = None
        self.h_lvl_i = 2 ** (input_qbit - 1) - 1
        self.h_lvl_w = 2 ** (weight_qbit - 1) - 1
        self.h_lvl_a = 2 ** (activation_qbit - 1) - 1

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl_w * self.scaler_dw
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.h_lvl_i
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        input_clip = F.hardtanh(input, min_val=-self.h_lvl_i * self.delta_x.item(),
                                max_val=self.h_lvl_i * self.delta_x.item())
        input_quan = quantize_input(input_clip, self.delta_x)  # * self.delta_x  # convert to voltage
        weight_quan = quantize_weight(self.weight, self.delta_w)  # * self.delta_w
        if self.bias is not None:
            bias_quan = quantize_weight(self.bias, self.delta_x)
        else:
            bias_quan = None

        output_crxb = F.conv2d(input, weight_quan, bias_quan, self.stride,
                               self.padding, self.dilation, self.groups)

        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / self.h_lvl_a
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * self.delta_i
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_a * self.delta_i.item(),
                                 max_val=self.h_lvl_a * self.delta_i.item())
        output_adc = adc(output_clip, self.delta_i, 1.)

        return output_adc

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0


class QLinear(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, input_qbit=32, weight_qbit=32, activation_qbit=32, scaler_dw=1):
        super(QLinear, self).__init__(in_features, out_features, bias)
        self.input_qbit = input_qbit
        self.weight_qbit = weight_qbit
        self.activation_qbit = activation_qbit
        self.scaler_dw = scaler_dw

        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('delta_out_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))

        self.delta_x = self.delta_w = None
        self.delta_i = self.delta_y = None
        self.h_lvl_i = 2 ** (input_qbit - 1) - 1
        self.h_lvl_w = 2 ** (weight_qbit - 1) - 1
        self.h_lvl_a = 2 ** (activation_qbit - 1) - 1

    def forward(self, input):
        # 1. input data and weight quantization
        with torch.no_grad():
            self.delta_w = self.weight.abs().max() / self.h_lvl_w * self.scaler_dw
            if self.training:
                self.counter.data += 1
                self.delta_x = input.abs().max() / self.h_lvl_i
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        input_clip = F.hardtanh(input, min_val=-self.h_lvl_i * self.delta_x.item(),
                                max_val=self.h_lvl_i * self.delta_x.item())
        input_quan = quantize_input(input_clip, self.delta_x)  # * self.delta_x  # convert to voltage
        weight_quan = quantize_weight(self.weight, self.delta_w)  # * self.delta_w
        if self.bias is not None:
            bias_quan = quantize_weight(self.bias, self.delta_x)
        else:
            bias_quan = None

        output_crxb = F.linear(input, weight_quan, bias_quan)

        with torch.no_grad():
            if self.training:
                self.delta_i = output_crxb.abs().max() / self.h_lvl_a
                self.delta_out_sum.data += self.delta_i
            else:
                self.delta_i = self.delta_out_sum.data / self.counter.data
            self.delta_y = self.delta_w * self.delta_x * self.delta_i
        #         print('adc LSB ration:', self.delta_i/self.max_i_LSB)
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl_a * self.delta_i.item(),
                                 max_val=self.h_lvl_a * self.delta_i.item())
        output_adc = adc(output_clip, self.delta_i, 1.)

        return output_adc

    def _reset_delta(self):
        self.delta_in_sum.data[0] = 0
        self.delta_out_sum.data[0] = 0
        self.counter.data[0] = 0


class QBN2d(nn.Module):
    def __init__(self, num_features, qbit, **kwargs):
        super(QBN2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features, **kwargs)
        self.activation_qbit = qbit

        self.register_buffer('delta_in_sum', torch.zeros(1))
        self.register_buffer('counter', torch.zeros(1))

        self.delta_x = None
        self.h_lvl = 2 ** (qbit - 1) - 1

    def forward(self, x):
        x = self.bn(x)
        with torch.no_grad():
            if self.training:
                self.counter.data += 1
                self.delta_x = x.abs().max() / self.h_lvl
                self.delta_in_sum.data += self.delta_x
            else:
                self.delta_x = self.delta_in_sum.data / self.counter.data

        x = F.hardtanh(x, min_val=-self.h_lvl * self.delta_x.item(),
                       max_val=self.h_lvl * self.delta_x.item())
        x = quantize_input(x, self.delta_x) # * self.delta_x
        return x
