import torch.nn as nn
from crisPy2.neural_network import ConvBlock, ResBlock

class Shaun(nn.Module):
    def __init__(in_channels, out_channels, nef):
        super(Shaun, self).__init__()

        self.init = ConvBlock(in_chnanels, nef)

        #downsample to N/2 where N is size of image
        self.R11 = ResBlock(nef, 2*nef, stride=2)
        self.R12 = ResBlock(2*nef, 2*nef)

        #downsample to N/4
        self.R21 = ResBlock(2*nef, 4*nef, stride=2)
        self.R22 = ResBlock(4*nef, 4*nef)

        #downsample to N/8
        self.R31 = ResBlock(4*nef, 8*nef, stride=2)
        self.R32 = ResBlock(8*nef, 8*nef)

        #convolutional latent space
        self.L1 = ConvBlock(8*nef, 16*nef)
        self.L2 = ConvBlock(16*nef, 8*nef)

        #upsample to N/4
        self.R51 = ResBlock(8*nef, 4*nef, upsample=True)
        self.R52 = ResBlock(4*nef, 4*nef)

        #upsample to N/2
        self.R61 = ResBlock(4*nef, 2*nef, upsample=True)
        self.R62 = ResBlock(2*nef, 2*nef)

        #upsample to N
        self.R71 = ResBlock(2*nef, nef, upsample=True)
        self.R72 = ResBlock(nef, nef)

        self.end = nn.Conv2d(nef, out_channels, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.end.weight)

    def forward(self, inp):
        init = self.init(inp)

        R11 = self.R11(init)
        R12 = self.R12(R11)
        R12 = R12 + R11

        R21 = self.R21(R12)
        R22 = self.R22(R21)
        R22 = R22 + R21

        R31 = self.R31(R22)
        R32 = self.R32(R31)
        R32 = R32 + R31

        L1 = self.L1(R32)
        L2 = self.L2(L1)

        R51 = self.R51(L2)
        R51 = R51 + R22
        R52 = self.R52(R51)
        R52 = R52 + R51

        R61 = self.R61(R52)
        R61 = R61 + R11
        R62 = self.R62(R61)
        R62 = R62 + R61

        R71 = self.R71(R62)
        R72 = self.R72(R71)

        end = self.end(R72)

        return end