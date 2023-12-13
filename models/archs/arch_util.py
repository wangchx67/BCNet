import torch
import torch.nn.functional as F
import numpy as np
from kornia.filters import sobel
from models.archs.Fourier import *

class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels * chan_factor), 1, stride=1, padding=0, bias=bias),
            # nn.BatchNorm2d(int(in_channels * chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels // chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
            # nn.BatchNorm2d(int(in_channels // chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class En_unit(nn.Module):
    def __init__(self, n_feat, chan_factor,bias=False, n=2,groups=1):
        super(En_unit, self).__init__()

        self.down=DownSample(n_feat, 2, chan_factor)
        self.conv=nn.Sequential(
            ProcessBlock(int(n_feat * chan_factor )),
            # nn.Conv2d(int(n_feat * chan_factor), int(n_feat * chan_factor), 3, 1, 1, bias=bias),
            # SpaBlock_RCB(int(n_feat * chan_factor ))
            # nn.BatchNorm2d(int(n_feat * chan_factor)),
            # # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x_in):
        x=self.down(x_in)
        x=self.conv(x)
        return x

class CEM(nn.Module):
    def __init__(self, ch=128):
        super(CEM,self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(ch,ch,3,1,1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1,groups=4),
            # nn.BatchNorm2d(ch),
            nn.LeakyReLU(0.1),
            # nn.SiLU(inplace=True)
        )

        self.sigm= nn.Sigmoid()
        self.tanh=nn.Tanh()

    def forward(self, fs):

        f1=fs[0]
        f2=fs[1]

        f12=f1+f2

        attention1 = self.sigm(self.conv1(f12))
        attention2 = self.sigm(self.conv2(f12))

        x = f1*attention1 + f2*attention2

        return x

class De_unit_concat2(nn.Module):
    def __init__(self, n_feat, chan_factor,bias=False, n=2,groups=1):
        super(De_unit_concat2, self).__init__()
        self.up=UpSample(n_feat, 2, chan_factor)
        self.conv0 = nn.Conv2d(int(n_feat / chan_factor),int(n_feat / chan_factor),3,1,1,bias=bias)
        # self.conv0 = SpaBlock(int(n_feat / chan_factor))

        self.fusion= nn.Sequential(
            nn.Conv2d(2*int(n_feat / chan_factor),int(n_feat / chan_factor),3,1,1,bias=bias),
            # nn.BatchNorm2d(int(n_feat / chan_factor)),
            # nn.LeakyReLU(0.1, inplace=True),
            # nn.SiLU(inplace=True)
        )

    def forward(self, x_in, x_en):
        x=self.conv0(self.up(x_in))
        x=self.fusion(torch.cat((x,x_en),dim=1))

        return x

class lightnessNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=32,
                 chan_factor=2,
                 bias=False,
                 ):

        super(lightnessNet, self).__init__()
        self.inp_channels=inp_channels
        self.out_channels=out_channels

        self.conv_in = nn.Sequential(
            nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias),
            # SpaBlock(int(n_feat * chan_factor ** 0)),
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
        )

        self.down1=En_unit(int((chan_factor ** 0) * n_feat),chan_factor,bias)
        self.down2=En_unit(int((chan_factor ** 1) * n_feat),chan_factor,bias)
        self.down3=En_unit(int((chan_factor ** 2) * n_feat),chan_factor,bias)
        self.down4=En_unit(int((chan_factor ** 3) * n_feat),chan_factor,bias)

        self.up1 = De_unit_concat2(int((chan_factor ** 4) * n_feat), chan_factor, bias)
        self.up2 = De_unit_concat2(int((chan_factor ** 3) * n_feat), chan_factor, bias)
        self.up3 = De_unit_concat2(int((chan_factor ** 2) * n_feat), chan_factor, bias)
        self.up4 = De_unit_concat2(int((chan_factor ** 1) * n_feat), chan_factor, bias)

        self.conv_out = nn.Sequential(
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
            # SpaBlock(int(n_feat * chan_factor ** 0)),
            nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, inp_img):

        inv_map = 1-inp_img
        edg_map = sobel(inp_img)

        shallow_feats = self.conv_in(inp_img)

        down1=self.down1(shallow_feats)
        down2=self.down2(down1)
        down3=self.down3(down2)
        down4=self.down4(down3)

        up1 = self.up1(down4,down3*inv_map[:,:,::8,::8]*(1+edg_map[:,:,::8,::8]))
        up2 = self.up2(up1,down2*inv_map[:,:,::4,::4]*(1+edg_map[:,:,::4,::4]))
        up3 = self.up3(up2,down1*inv_map[:,:,::2,::2]*(1+edg_map[:,:,::2,::2]))
        up4 = self.up4(up3,shallow_feats*inv_map*(1+edg_map))

        # up1 = self.up1(down4, down3)
        # up2 = self.up2(up1, down2)
        # up3 = self.up3(up2, down1)
        # up4 = self.up4(up3, shallow_feats)

        out_img = self.conv_out(up4)+inp_img

        return out_img,[shallow_feats,down1,down2,down3,down4]

class ChromNet(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 n_feat=32,
                 chan_factor=2,
                 bias=False,
                 color_aug=False
                 ):
        super(ChromNet, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.color_aug = color_aug

        self.conv_in = nn.Sequential(
            nn.Conv2d(inp_channels, int(n_feat * chan_factor ** 0), kernel_size=3, padding=1, bias=bias),
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
        )

        self.down1 = En_unit(int((chan_factor ** 0) * n_feat), chan_factor, bias, groups=1)
        self.down2 = En_unit(int((chan_factor ** 1) * n_feat), chan_factor, bias, groups=2)

        self.fusion2 = CEM(ch=int((chan_factor ** 2) * n_feat))
        # self.fusion2 = nn.Conv2d(int((chan_factor ** 2) * n_feat * 2),int((chan_factor ** 2) * n_feat),3,1,1)

        self.up1 = De_unit_concat2(int((chan_factor ** 4) * n_feat), chan_factor, bias, 2,groups=8)
        self.up2 = De_unit_concat2(int((chan_factor ** 3) * n_feat), chan_factor, bias, 2,groups=4)
        self.up3 = De_unit_concat2(int((chan_factor ** 2) * n_feat), chan_factor, bias, 2,groups=2)
        self.up4 = De_unit_concat2(int((chan_factor ** 1) * n_feat), chan_factor, bias, 2,groups=1)

        self.encode_q = nn.Conv2d(int((chan_factor ** 2) * n_feat), 313, kernel_size=1, padding=0, dilation=1, stride=1, bias=True)

        self.conv_out = nn.Sequential(
            # ProcessBlock(int(n_feat * chan_factor ** 0)),
            nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias),
        )

    def forward(self, inp_img, mid=None, ref_map=None):

        if ref_map is not None:
            shallow_feats_ = self.conv_in(ref_map)
            shallow_feats = self.conv_in(inp_img)
            down1_ = self.down1(shallow_feats_)
            down1 = self.down1(shallow_feats)
            down2_ = self.down2(down1_)
            down2 = 0.7 * self.down2(down1)  + 0.3 * down2_

        else:
            if self.color_aug:
                random_number1 = torch.rand(1)
                if random_number1>0.5:
                    random_number2 = 0.5 + 0.5 * torch.rand(1)
                    random_number2 = random_number2.to(inp_img.device)
                    inp_img = inp_img*random_number2
            shallow_feats = self.conv_in(inp_img)
            down1 = self.down1(shallow_feats)
            down2 = self.down2(down1)

        up1 = self.up1(mid[4], mid[3])
        up2 = self.up2(up1, mid[2])

        up2 = self.fusion2([up2, down2])
        # up2 = self.fusion2(torch.cat((up2, down2),dim=1))
        q = self.encode_q(up2)
        up3 = self.up3(up2,mid[1])
        up4 = self.up4(up3,mid[0])

        out_img = self.conv_out(up4)

        return out_img, q
