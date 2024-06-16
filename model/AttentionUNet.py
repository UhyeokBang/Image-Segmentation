import torch
import torch.nn as nn


class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(AttentionUNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
        def ConvLayer2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
            return nn.Sequential(
                CBR2d(in_channels, out_channels, kernel_size, padding, stride, bias),
                CBR2d(out_channels, out_channels, kernel_size, padding, stride, bias)
            )
        
        def PoolLayer2d(kernel_size=2, stride=2):
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        def UpConvLayer2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.encoder1 = ConvLayer2d(in_channels, 64)
        self.pool1 = PoolLayer2d(2, 2)

        self.encoder2 = ConvLayer2d(64, 128)
        self.pool2 = PoolLayer2d(2, 2)

        self.encoder3 = ConvLayer2d(128, 256)
        self.pool3 = PoolLayer2d(2, 2)

        self.encoder4 = ConvLayer2d(256, 512)
        self.pool4 = PoolLayer2d(2, 2)

        self.bridge = ConvLayer2d(512, 1024)

        self.upconv4 = UpConvLayer2d(1024, 512)
        self.decoder4 = ConvLayer2d(1024, 512)
        self.attention4 = AttentionBlock(F_g=512, F_l=512, F_int=256)

        self.upconv3 = UpConvLayer2d(512, 256)
        self.decoder3 = ConvLayer2d(512, 256)
        self.attention3 = AttentionBlock(F_g=256, F_l=256, F_int=128)

        self.upconv2 = UpConvLayer2d(256, 128)
        self.decoder2 = ConvLayer2d(256, 128)
        self.attention2 = AttentionBlock(F_g=128, F_l=128, F_int=64)

        self.upconv1 = UpConvLayer2d(128, 64)
        self.decoder1 = ConvLayer2d(128, 64)
        self.attention1 = AttentionBlock(F_g=64, F_l=64, F_int=32)

        self.output_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)

        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)

        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)

        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        bridge = self.bridge(p4)

        u4 = self.upconv4(bridge)
        e4 = self.attention4(g=u4, x=e4)
        c4 = torch.cat((u4, e4), dim=1)
        d4 = self.decoder4(c4)

        u3 = self.upconv3(d4)
        e3 = self.attention3(g=u3, x=e3)
        c3 = torch.cat((u3, e3), dim=1)
        d3 = self.decoder3(c3)

        u2 = self.upconv2(d3)
        e2 = self.attention2(g=u2, x=e2)
        c2 = torch.cat((u2, e2), dim=1)
        d2 = self.decoder2(c2)

        u1 = self.upconv1(d2)
        e1 = self.attention1(g=u1, x=e1)
        c1 = torch.cat((u1, e1), dim=1)
        d1 = self.decoder1(c1)

        output = self.output_conv(d1)

        return output
