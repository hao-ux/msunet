import torch
import torch.nn as nn
import torch.nn.functional as F
# from nets.odconv import ODConv2d



class PyramidConv(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.div_inchannel = in_channels // 4
        self.conv1x1 = nn.Conv2d(self.div_inchannel, self.div_inchannel, 1,padding=0, groups=self.div_inchannel, bias=False)
        self.conv3x3 = nn.Conv2d(self.div_inchannel, self.div_inchannel, 3,padding=1, groups=self.div_inchannel, bias=False)
        self.conv5x5 = nn.Conv2d(self.div_inchannel, self.div_inchannel, 5,padding=2, groups=self.div_inchannel, bias=False)
        self.conv7x7 = nn.Conv2d(self.div_inchannel, self.div_inchannel, 7,padding=3, groups=self.div_inchannel, bias=False)
        self.gn = nn.GroupNorm(4, in_channels)
    
    
    def forward(self, x):
        x = self.gn(x)
        x0, x1, x2, x3 = torch.split(x, [
                self.div_inchannel, self.div_inchannel, self.div_inchannel, self.div_inchannel
            ], dim=1)
        x0 = self.conv1x1(x0)
        x1 = self.conv3x3(x1)
        x2 = self.conv5x5(x2)
        x3 = self.conv7x7(x3)
        x = torch.cat([x0, x1, x2, x3], 1) + x
        
        return x

class Agg(nn.Module):
    def __init__(self, in_channels, k=3):
        super().__init__()
        self.maxpool = nn.MaxPool2d(k, 1, k//2)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, groups=1, bias=False),
            nn.GroupNorm(4, in_channels),
            nn.GELU()
        )
        
    def forward(self, x):
        inputx = x
        x = self.maxpool(x)
        x = self.fc(x + inputx)
        return x
    

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.rpf = PyramidConv(in_channels)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, groups=1, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0, groups=1, bias=False)
        )
        
    def forward(self, x):
        x = self.rpf(x)
        x = self.fc(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0,groups=1,bias=False)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 5), padding=(0, 2), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=(1, 7), padding=(0, 3), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels),
            nn.Conv2d(out_channels, out_channels, kernel_size=(7, 1), padding=(3, 0), groups=out_channels, bias=False),
            nn.GroupNorm(4, out_channels)
        )
        self.act = nn.GELU()
        self.bn = nn.GroupNorm(4, out_channels)
        self.conv5 = nn.Conv2d(out_channels*3, out_channels, 1, 1, 0,groups=out_channels,bias=False)
        self.conv6 = nn.Conv2d(out_channels, out_channels, 1, 1, 0,groups=1,bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.conv2(x)
        x2 = self.conv3(x + x1)
        x3 = self.conv4(x + x2)
        x = torch.cat([x1, x2, x3], 1)
        x = self.conv5(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv6(x)
        return x

class MSUNet(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
        self.encoder1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, 1,1, bias=False),
            Encoder(8, 8)
        )
        self.encoder2 = Encoder(8, 16)
        self.encoder3 = Encoder(16, 24)
        self.encoder4 = Encoder(24, 32)
        self.encoder5 = Encoder(32, 40)
        self.encoder6 = nn.Sequential(
            nn.Conv2d(40, 40, 3, 1, 1, groups=40, bias=False),
            nn.GroupNorm(4, 40),
            nn.GELU(),
            nn.Conv2d(40, 40, 1, 1, 0, groups=1, bias=False)
        )
        self.max_pool = nn.MaxPool2d(3, 2, 1)
        self.decoder5 = Decoder(40, 32)
        self.decoder4 = Decoder(32, 24)
        self.decoder3 = Decoder(24, 16)
        
        self.decoder2 = Decoder(16, 8)
        self.decoder1 = nn.Sequential(
            nn.Conv2d(8, num_classes, 1, 1, bias=False)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(72, 32, 1, bias=False),
            nn.GroupNorm(4, 32),
            Agg(32, 7)
        )
        self.conv22 = nn.Sequential(
            nn.Conv2d(56, 24, 1, bias=False),
            nn.GroupNorm(4, 24),
            Agg(24, 7)
        )
        self.conv33 = nn.Sequential(
            nn.Conv2d(40, 16, 1, bias=False),
            nn.GroupNorm(4, 16),
            Agg(16, 3)
        )
        self.conv44 = nn.Sequential(
            nn.Conv2d(24, 8, 1, bias=False),
            nn.GroupNorm(4, 8),
            Agg(8, 3)
        )

    def forward(self, x):
        # print(F.max_pool2d(x, 32, 32).shape)
        x1 = self.max_pool(self.encoder1(x))
        x2 = self.max_pool(self.encoder2(x1))
        x3 = self.max_pool(self.encoder3(x2))
        x4 = self.max_pool(self.encoder4(x3))
        x5 = self.max_pool(self.encoder5(x4))
        x6 = self.encoder6(x5)
        x44 = F.interpolate(x6, scale_factor=2,mode ='bilinear', align_corners=True)
        x61 = self.conv11(torch.cat([x44, x4], 1))
        x33 = F.interpolate(x61, scale_factor=2,mode ='bilinear', align_corners=True)
        x62 = self.conv22(torch.cat([x33, x3], 1))
        x22 = F.interpolate(x62, scale_factor=2,mode ='bilinear', align_corners=True)
        x63 = self.conv33(torch.cat([x22, x2], 1))
        x11 = F.interpolate(x63, scale_factor=2,mode ='bilinear', align_corners=True)
        x64 = self.conv44(torch.cat([x11, x1], 1))
        
        x7 = self.decoder5(F.interpolate(x6+x5, scale_factor=2,mode ='bilinear', align_corners=True)) + x61
        x8 = self.decoder4(F.interpolate(x7, scale_factor=2,mode ='bilinear', align_corners=True)) + x62
        x9 = self.decoder3(F.interpolate(x8, scale_factor=2,mode ='bilinear', align_corners=True)) + x63
        x10 = self.decoder2(F.interpolate(x9, scale_factor=2,mode ='bilinear', align_corners=True)) + x64
        x = F.interpolate(x10, scale_factor=2,mode ='bilinear', align_corners=True)
        x11 = self.decoder1(x)
        # print(self.conv(x6).shape)
        # print(x11.shape)
        return x11

