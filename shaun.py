from torch import tanh
import torch.nn as nn
from crisPy2.neural_network import ConvBlock, ResBlock

class Shaun(nn.Module):
    def __init__(in_channels, out_channels, nef):
        super(Shaun, self).__init__()

        self.C01 = ConvBlock(in_channels, nef, kernel=7)

        self.C11 = ConvBlock(nef, 2*nef, stride=2)

        self.C21 = ConvBlock(2*nef, 4*nef, stride=2)

        self.R1 = ResBlock(4*nef, 4*nef)
        self.R2 = ResBlock(4*nef, 4*nef)
        self.R3 = ResBlock(4*nef, 4*nef)
        self.R4 = ResBlock(4*nef, 4*nef)
        self.R5 = ResBlock(4*nef, 4*nef)
        self.R6 = ResBlock(4*nef, 4*nef)
        self.R7 = ResBlock(4*nef, 4*nef)
        self.R8 = ResBlock(4*nef, 4*nef)
        self.R9 = ResBlock(4*nef, 4*nef)

        self.C31 = ConvBlock(4*nef, 2*nef, upsample=True)

        self.C41 = ConvBlock(2*nef, nef, upsample=True)

        self.C51 = ConvBlock(nef, out_channels, kernel=7)

    def forward(self, inp):
        C01 = self.C01(inp)

        C11 = self.C11(C01)

        C21 = self.C21(C11)

        R1 = self.R1(C21)
        R2 = self.R2(R1)
        R3 = self.R3(R2)
        R4 = self.R4(R3)
        R5 = self.R5(R4)
        R6 = self.R6(R5)
        R7 = self.R7(R6)
        R8 = self.R8(R7)
        R9 = self.R9(R8)

        C31 = self.C31(R9)

        C41 = self.C41(C31)

        C51 = self.C51(C41)

        return tanh(C51) + inp