import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adc import _adc
from .dac import _quantize_dac
from .SAF import Inject_SAF

quantize_input = _quantize_dac.apply
quantize_weight = _quantize_dac.apply
adc = _adc.apply


# def __init__(self,ir_drop, device, gmax, gmin, gwire, gload, scaler_dw=1, vdd=3.3, enable_noise=True,
#              freq=10e6, temp=300 , crxb_size=64, quantize=8, enable_SAF=False,
#              enable_ec_SAF=False):


class CrossbarLayer():
    """
    This is the custom conv layer that takes non-ideal effects of ReRAM crossbar into account. It has three functions.
    1) emulate the DAC at the input of the crossbar and qnantize the input and weight tensors.
    2) map the quantized tensor to the ReRAM crossbar arrays and include non-ideal effects such as noise, ir drop, and
        SAF.
    3) emulate the ADC at the output of he crossbar and convert the current back to digital number
        to the input of next layers

    Args:
        ir_drop(bool): switch that enables the ir drop calculation.
        device(torch.device): device index to select. It’s a no-op if this argument is a negative integer or None.
        gmax(float): maximum conductance of the ReRAM.
        gmin(float): minimun conductance of the ReRAM.
        gwire(float): conductance of the metal wire.
        gload(float): load conductance of the ADC and DAC.
        vdd(float): supply voltage.
        enable_noise(bool): switch to enable stochastic_noise.
        freq(float): operating frequency of the ReRAM crossbar.
        temp(float): operating temperature of ReRAM crossbar.
        crxb_size(int): size of the crossbar.
        quantize(int): quantization resolution of the crossbar.
        enable_SAF(bool): switch to enable SAF
        enable_ec_SAF(bool): switch to enable SAF error correction.
    """

    def __init__(self, ir_drop, device, gmax, gmin, gwire, gload, vdd=3.3, enable_noise=True,
                 freq=10e6, temp=300, crxb_size=64, quantize=8, enable_SAF=False, enable_ec_SAF=False):

        self.ir_drop = ir_drop
        self.device = device

        ################## Crossbar conversion #############################
        self.crxb_size = crxb_size
        self.enable_ec_SAF = enable_ec_SAF
        self.enable_SAF = enable_SAF

        # self.nchout_index = nn.Parameter(torch.arange(self.out_channels), requires_grad=False)
        # weight_flatten = self.weight.view(self.out_channels, -1)
        # self.crxb_row, self.crxb_row_pads = self.num_pad(
        #     weight_flatten.shape[1], self.crxb_size)
        # self.crxb_col, self.crxb_col_pads = self.num_pad(
        #     weight_flatten.shape[0], self.crxb_size)
        # self.w_pad = (0, self.crxb_row_pads, 0, self.crxb_col_pads)
        # self.input_pad = (0, 0, 0, self.crxb_row_pads)
        # weight_padded = F.pad(weight_flatten, self.w_pad,
        #                       mode='constant', value=0)
        # weight_crxb = weight_padded.view(self.crxb_col, self.crxb_size,
        #                                  self.crxb_row, self.crxb_size).transpose(1, 2)

        ################# Hardware conversion ##############################
        # weight and input levels
        # Q(x) = (2^{k−1} − 1)* round((2^{k−1} − 1) * x)
        self.n_lvl = 2 ** quantize
        self.h_lvl = (self.n_lvl - 2) / 2
        # ReRAM cells
        # 7-bit precisionis achievable on state-of-the-art RRAM device [9]
        # [9] High precision tuning of state for memristive devices by
        # adaptable variation-tolerant algorithm
        self.Gmax = gmax  # max conductance
        self.Gmin = gmin  # min conductance
        self.delta_g = (self.Gmax - self.Gmin) / (2 ** 7)  # conductance step
        # self.w2g = w2g(self.delta_g, Gmin=self.Gmin, G_SA0=self.Gmax,
        #                G_SA1=self.Gmin, weight_shape=weight_crxb.shape, enable_SAF=enable_SAF)
        self.Gwire = gwire
        self.Gload = gload
        # DAC
        self.Vdd = vdd  # unit: volt
        self.delta_v = self.Vdd / (self.n_lvl - 1)
        # self.delta_in_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.delta_out_sum = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.counter = nn.Parameter(torch.Tensor(1), requires_grad=False)
        # self.scaler_dw = scaler_dw
        # self.delta_w = 0  # self.weight.abs().max() / self.h_lvl * self.scaler_dw
        # self.delta_x = 0  # self.delta_in_sum.data / self.counter.data

        ################ Stochastic Conductance Noise setup #########################
        # parameters setup
        self.enable_stochastic_noise = enable_noise
        self.freq = freq  # operating frequency
        self.kb = 1.38e-23  # Boltzmann const
        self.temp = temp  # temperature in kelvin
        self.q = 1.6e-19  # electron charge

        self.tau = 0.5  # Probability of RTN
        self.a = 1.662e-7  # RTN fitting parameter
        self.b = 0.0015  # RTN fitting parameter

        self.SAF_dist=[1.75, 9.04]
        self.variation=0.05/3

        # # compute output feature size
        # if self.h_out is None and self.w_out is None:
        #     self.h_out = int(
        #         (input.shape[2] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        #     self.w_out = int(
        #         (input.shape[3] - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1)
        #
        # self.num_block = self.h_out*self.w_out
        # block_size = input.shape[1]*torch.cumprod(torch.tensor(self.kernel_size.shape),0)[-1]
        # self.pad_block_size = block_size + self.input_pad[2]+self.input_pad[3]

    def quantize(self, input, delta_x, weight=None, delta_w=None):
        # quantization
        input_clip = F.hardtanh(input, min_val=-self.h_lvl * delta_x.item(),
                                max_val=self.h_lvl * delta_x.item())
        input_quan = quantize_input(input_clip, delta_x) * self.delta_v  # convert to voltage
        if weight is not None:
            weight_quan = quantize_weight(weight, delta_w)
            return input_quan, weight_quan
        else:
            return input_quan

    def noise_injection(self, input_crxb, G_crxb):
        # this block is for introducing stochastic noise into ReRAM conductance
        if self.enable_stochastic_noise:
            # nn.Module.register_buffer('rand_p',torch.empty(G_crxb.shape))
            # nn.Module.register_buffer('rand_g',torch.empty(G_crxb.shape))
            rand_p = nn.Parameter(torch.empty(G_crxb.shape), requires_grad=False)
            rand_g = nn.Parameter(torch.empty(G_crxb.shape), requires_grad=False)
            if self.device.type == "cuda":
                rand_p = rand_p.cuda()
                rand_g = rand_g.cuda()
            with torch.no_grad():
                input_reduced = (input_crxb.norm(p=2, dim=0).norm(p=2, dim=3).unsqueeze(dim=3)) / \
                                (input_crxb.shape[0] * input_crxb.shape[3])
                # equivalent standard deviation of ReRAM conductance
                # Thermal noise + shot noise + programming variation
                grms = torch.sqrt(G_crxb * self.freq * (4 * self.kb * self.temp + 2 * self.q * input_reduced) / (
                            input_reduced ** 2) + (self.delta_g / 3) ** 2)
                grms[torch.isnan(grms)] = 0
                grms[grms.eq(float('inf'))] = 0

                # Random Telegraph Noise
                rand_p.uniform_()
                rand_g.normal_(0, 1)
                G_p = G_crxb * (self.b * G_crxb + self.a) / (G_crxb - (self.b * G_crxb + self.a))
                G_p[rand_p.ge(self.tau)] = 0
                G_g = grms * rand_g

            G_crxb += (G_g + G_p)
        return G_crxb

    def inject_noise_s(self, G_crxb, SAF = None):
        temp_weight = torch.tensor(G_crxb.data)
        if self.enable_stochastic_noise:
            temp_var = torch.empty_like(G_crxb).normal_(0,self.variation)
            temp_weight = temp_weight * (1+temp_var)
        if self.enable_SAF:
            Inject_SAF(temp_weight, SAF, self.SAF_dist[0] / 100, self.SAF_dist[-1] / 100, self.Gmax, self.Gmin)
        return temp_weight

    def w2g(self, input):
        # x_relu() function is Critical
        G_pos = self.Gmin + F.relu(input) * self.delta_g
        G_neg = self.Gmin + F.relu(-input) * self.delta_g
        output = torch.cat((G_pos.unsqueeze(0), G_neg.unsqueeze(0)), 0)
        return output

    # this block is to calculate the ir drop of the crossbar
    def solve_crxb(self, input_crxb, G_crxb):
        if self.ir_drop:
            from .IR_solver import IrSolver

            crxb_pos = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[0].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_pos.resetcoo()
            crxb_neg = IrSolver(Rsize=self.crxb_size,
                                Csize=self.crxb_size,
                                Gwire=self.Gwire,
                                Gload=self.Gload,
                                input_x=input_crxb.permute(3, 0, 1, 2, 4),
                                Gmat=G_crxb[1].permute(3, 2, 0, 1),
                                device=self.device)
            crxb_neg.resetcoo()

            output_crxb = (crxb_pos.caliout() - crxb_neg.caliout())
            crxb_col = G_crxb.shape[0]
            crxb_row = G_crxb.shape[1]
            output_crxb = output_crxb.contiguous().view(crxb_col,
                                                        crxb_row,
                                                        self.crxb_size,
                                                        input_crxb.shape[0],
                                                        input_crxb.shape[-1])

            output_crxb = output_crxb.permute(3, 0, 1, 2, 4)

        else:
            # print(G_crxb[0].is_cuda,input_crxb.is_cuda)
            #
            output_crxb = torch.matmul(G_crxb[0], input_crxb) - \
                          torch.matmul(G_crxb[1], input_crxb)
        return output_crxb

    # 3. perform ADC operation (i.e., current to digital conversion)
    def output_convet(self, output_crxb, delta_i, delta_y):
        output_clip = F.hardtanh(output_crxb, min_val=-self.h_lvl * delta_i.item(),
                                 max_val=self.h_lvl * delta_i.item())
        output_adc = adc(output_clip, delta_i, delta_y)
        return output_adc

def inject_noise_s(G_crxb, variation=None, SAF_dist=None, Gmin=None, Gmax=None):
    temp_weight = torch.tensor(G_crxb.data)
    if variation is not None:
        temp_var = torch.empty_like(G_crxb).normal_(0,variation)
        temp_weight = temp_weight * (1+temp_var)
    if SAF_dist is not None:
        SAF = torch.empty_like(G_crxb).normal_(0,variation)
        temp_weight = torch.where(SAF < float(SAF_dist[0] / 100), torch.full_like(G_crxb,Gmin), temp_weight)
        temp_weight = torch.where(SAF > 1 - float(SAF_dist[-1] / 100), torch.full_like(G_crxb,Gmax), temp_weight)
    return temp_weight
