import torch
import torch.nn as nn

from ..utils import fft2c, ifft2c, combine_all_coils, project_all_coils
from .complex import Conv2plus1d, Conv3d

from einops import rearrange, parse_shape

class MC_dataConsistencyTerm(nn.Module):
    """
    Inputs:
    1. Coil Combined Image (x) : Slices, Contrast, XRes, YRes, (real, imag)
    2. Undersampled Kspace (k0) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    3. Mask (mask) : Slices, Coils, Contrast, XRes, YRes, (real, imag)
    4. Sensitivity maps (sensitivity): Slices, Coils, Contrast, XRes, YRes, (real, imag)

    Outputs:
    coil combined (out): Slices, Contrast, XRes, YRes, (real,imag)
    """

    def __init__(self, num_echos, noise_lvl=None):
        super(MC_dataConsistencyTerm, self).__init__()
        self.noise_lvl = noise_lvl
        if noise_lvl is not None:
            noise_lvl_tensor = torch.Tensor([noise_lvl]) * torch.ones(
                num_echos
            )  # Different lvl for each contrast & Channels
            self.noise_lvl = torch.nn.Parameter(noise_lvl_tensor)

    def perform(self, x, k0, mask, sensitivity, coil_dim=1):
        k = fft2c(project_all_coils(x, sensitivity, coil_dim))
        if self.noise_lvl is not None:  # noisy case
            v = torch.sigmoid(self.noise_lvl)  # Normalize to 0~1
            v = self.noise_lvl.unsqueeze(0).unsqueeze(1).unsqueeze(3).unsqueeze(4).unsqueeze(5)

            k = (1 - mask) * k + mask * (v * k + (1 - v) * k0)

        else:  # noiseless case
            k = (1 - mask) * k + mask * k0
        return combine_all_coils(ifft2c(k), sensitivity, coil_dim)


class weightedAverageTerm(nn.Module):
    def __init__(self, num_echos, para=None):
        super(weightedAverageTerm, self).__init__()
        self.para = para
        if para is not None:
            para = torch.Tensor([para]) * torch.ones(
                num_echos
            )  # Different lvl for each contrast
            
            self.para = torch.nn.Parameter(para)

    def perform(self, cnn, Sx):
        para = torch.sigmoid(self.para)
        para = self.para.unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        return para * cnn + (1 - para) * Sx

class MC_cnn_layer(nn.Module):
    """
    Inputs: Slices, Contrast, XRes, YRes, (real, imag) tensor

    Outputs: Slices, Contrast, XRes, YRes, (real, imag) Tensor (Denoised)
    """

    def __init__(self, num_echos, nfeatures=64):
        super(MC_cnn_layer, self).__init__()
        conf = dict(ch1=64, ch2=64, ksize=9, pad=4, st=3)

        self.conv1 = Conv2plus1d(1,           conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv2 = Conv2plus1d(conf["ch1"], conf["ch2"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), stride=(conf["st"], 1, 1), bias=True)
        self.conv3 = Conv2plus1d(conf["ch2"], conf["ch1"], kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)
        self.conv4 = Conv2plus1d(conf["ch1"],           1, kernel_size=(conf["ksize"], 3, 3), padding=(conf["pad"], 1, 1), bias=True)

    def forward(self, x):
        size1 = x.size()[1:-1]
        x = self.conv1(x)
        x = nn.functional.relu(x, inplace=True)

        size2 = x.size()[2:-1]
        x = self.conv2(x)
        x = nn.functional.relu(x, inplace=True)
       
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size2, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size2, mode='nearest')
        x = torch.stack([real, imag], -1)

        x = self.conv3(x)
        x = nn.functional.relu(x, inplace=True)

        
        real, imag = x.unbind(-1)
        real = nn.functional.interpolate(real, size=size1, mode='nearest')
        imag = nn.functional.interpolate(imag, size=size1, mode='nearest')
        x = torch.stack([real, imag], -1)

        x = self.conv4(x)
        return x

class JVSNet(nn.Module):
    def __init__(self, num_echos, alfa=None, beta=1, cascades=5):
        super(JVSNet, self).__init__()

        self.cascades = cascades
        conv_blocks = []
        dc_blocks = []
        wa_blocks = []

        for _ in range(cascades):
            conv_blocks.append(MC_cnn_layer(num_echos))
            dc_blocks.append(MC_dataConsistencyTerm(num_echos, noise_lvl=alfa))
            wa_blocks.append(weightedAverageTerm(num_echos, para=beta))

        self.conv_blocks = nn.ModuleList(conv_blocks)
        self.dc_blocks = nn.ModuleList(dc_blocks)
        self.wa_blocks = nn.ModuleList(wa_blocks)

    def forward(self, x, k, m, c):

        for i in range(self.cascades):
            Sx = self.dc_blocks[i].perform(x, k, m, c)
            x = self.conv_blocks[i](x) + x
            x = self.wa_blocks[i].perform(x, Sx)
        return x
