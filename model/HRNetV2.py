import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_bias=False):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x

def fuse_layer1(x):
    x0_0 = x[0]
    x0_1 = nn.Conv2d(x[1].size(1), 32, kernel_size=1, bias=False)(x[1])
    x0_1 = nn.BatchNorm2d(32)(x0_1)
    x0_1 = F.interpolate(x0_1, scale_factor=2, mode='nearest')
    x0 = x0_0 + x0_1

    x1_0 = nn.Conv2d(x[0].size(1), 64, kernel_size=3, stride=2, padding=1, bias=False)(x[0])
    x1_0 = nn.BatchNorm2d(64)(x1_0)
    x1_1 = x[1]
    x1 = x1_0 + x1_1
    return [x0, x1]

def fuse_layer2(x):
    x0_0 = x[0]
    x0_1 = nn.Conv2d(x[1].size(1), 32, kernel_size=1, bias=False)(x[1])
    x0_1 = nn.BatchNorm2d(32)(x0_1)
    x0_1 = F.interpolate(x0_1, scale_factor=2, mode='nearest')
    x0_2 = nn.Conv2d(x[2].size(1), 32, kernel_size=1, bias=False)(x[2])
    x0_2 = nn.BatchNorm2d(32)(x0_2)
    x0_2 = F.interpolate(x0_2, scale_factor=4, mode='nearest')
    x0 = x0_0 + x0_1 + x0_2

    x1_0 = nn.Conv2d(x[0].size(1), 64, kernel_size=3, stride=2, padding=1, bias=False)(x[0])
    x1_0 = nn.BatchNorm2d(64)(x1_0)
    x1_1 = x[1]
    x1_2 = nn.Conv2d(x[2].size(1), 64, kernel_size=1, bias=False)(x[2])
    x1_2 = nn.BatchNorm2d(64)(x1_2)
    x1_2 = F.interpolate(x1_2, scale_factor=2, mode='nearest')
    x1 = x1_0 + x1_1 + x1_2

    x2_0 = nn.Conv2d(x[0].size(1), 32, kernel_size=3, stride=2, padding=1, bias=False)(x[0])
    x2_0 = nn.BatchNorm2d(32)(x2_0)
    x2_0 = Mish()(x2_0)
    x2_0 = nn.Conv2d(32, 128, kernel_size=3, stride=2, padding=1, bias=False)(x2_0)
    x2_0 = nn.BatchNorm2d(128)(x2_0)
    x2_1 = nn.Conv2d(x[1].size(1), 128, kernel_size=3, stride=2, padding=1, bias=False)(x[1])
    x2_1 = nn.BatchNorm2d(128)(x2_1)
    x2_2 = x[2]
    x2 = x2_0 + x2_1 + x2_2
    return [x0, x1, x2]

def fuse_layer3(x):
    x0_0 = x[0]
    x0_1 = nn.Conv2d(x[1].size(1), 32, kernel_size=1, bias=False)(x[1])
    x0_1 = nn.BatchNorm2d(32)(x0_1)
    x0_1 = F.interpolate(x0_1, scale_factor=2, mode='nearest')
    x0_2 = nn.Conv2d(x[2].size(1), 32, kernel_size=1, bias=False)(x[2])
    x0_2 = nn.BatchNorm2d(32)(x0_2)
    x0_2 = F.interpolate(x0_2, scale_factor=4, mode='nearest')
    x0_3 = nn.Conv2d(x[3].size(1), 32, kernel_size=1, bias=False)(x[3])
    x0_3 = nn.BatchNorm2d(32)(x0_3)
    x0_3 = F.interpolate(x0_3, scale_factor=8, mode='nearest')
    x0 = torch.cat([x0_0, x0_1, x0_2, x0_3], dim=1)
    return x0


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class ConvBnMish(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBnMish, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.mish = Mish()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        return x

def transition_layer1(x, out_channels_list=[32, 64]):
    x0 = ConvBnMish(x.size(1), out_channels_list[0])(x)
    x1 = ConvBnMish(x.size(1), out_channels_list[1], stride=2)(x)
    return [x0, x1]

def transition_layer2(x, out_channels_list=[32, 64, 128]):
    x0 = ConvBnMish(x[0].size(1), out_channels_list[0])(x[0])
    x1 = ConvBnMish(x[1].size(1), out_channels_list[1])(x[1])
    x2 = ConvBnMish(x[1].size(1), out_channels_list[2], stride=2)(x[1])
    return [x0, x1, x2]

def transition_layer3(x, out_channels_list=[32, 64, 128, 256]):
    x0 = ConvBnMish(x[0].size(1), out_channels_list[0])(x[0])
    x1 = ConvBnMish(x[1].size(1), out_channels_list[1])(x[1])
    x2 = ConvBnMish(x[2].size(1), out_channels_list[2])(x[2])
    x3 = ConvBnMish(x[2].size(1), out_channels_list[3], stride=2)(x[2])
    return [x0, x1, x2, x3]



# Basic 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_conv_shortcut=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.mish1 = Mish()

        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.with_conv_shortcut = with_conv_shortcut
        if self.with_conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.mish2 = Mish()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.with_conv_shortcut:
            residual = self.shortcut(x)
        
        out += residual
        out = self.mish2(out)
        return out

# Bottleneck Block
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_conv_shortcut=False):
        super(BottleneckBlock, self).__init__()
        expansion = 4
        de_channels = out_channels // expansion

        self.conv1 = nn.Conv2d(in_channels, de_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(de_channels)
        self.mish1 = Mish()

        self.conv2 = nn.Conv2d(de_channels, de_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(de_channels)
        self.mish2 = Mish()

        self.conv3 = nn.Conv2d(de_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.with_conv_shortcut = with_conv_shortcut
        if self.with_conv_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.mish3 = Mish()

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.mish1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.mish2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.with_conv_shortcut:
            residual = self.shortcut(x)
        
        out += residual
        out = self.mish3(out)
        return out

# Stem network
class StemNet(nn.Module):
    def __init__(self, in_channels):
        super(StemNet, self).__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.mish = Mish()

        self.block1 = BottleneckBlock(64, 256, with_conv_shortcut=True)
        self.block2 = BottleneckBlock(256, 256, with_conv_shortcut=False)
        self.block3 = BottleneckBlock(256, 256, with_conv_shortcut=False)
        self.block4 = BottleneckBlock(256, 256, with_conv_shortcut=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mish(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x

# Function to create branches
def make_branch(block, in_channels, out_channels):
    layers = []
    for _ in range(4):
        layers.append(block(in_channels, out_channels, with_conv_shortcut=False))
    return nn.Sequential(*layers)

# Final layer
class FinalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FinalLayer, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.sigmoid(x)
        return x

# HRNet Segmentation Model
class HRNetSeg(nn.Module):
    def __init__(self, in_channels = 3, num_classes= 1000):
        super(HRNetSeg, self).__init__()
        self.stem_net = StemNet(in_channels)

        # Assume fuse_layer and transition_layer functions are defined elsewhere
        self.fuse_layer1 = fuse_layer1
        self.fuse_layer2 = fuse_layer2
        self.fuse_layer3 = fuse_layer3
        self.transition_layer1 = transition_layer1
        self.transition_layer2 = transition_layer2
        self.transition_layer3 = transition_layer3

        self.branch1_0 = make_branch(BasicBlock, 256, 32)
        self.branch1_1 = make_branch(BasicBlock, 256, 64)

        self.branch2_0 = make_branch(BasicBlock, 256, 32)
        self.branch2_1 = make_branch(BasicBlock, 256, 64)
        self.branch2_2 = make_branch(BasicBlock, 256, 128)

        self.branch3_0 = make_branch(BasicBlock, 256, 32)
        self.branch3_1 = make_branch(BasicBlock, 256, 64)
        self.branch3_2 = make_branch(BasicBlock, 256, 128)
        self.branch3_3 = make_branch(BasicBlock, 256, 256)

        self.final_layer = FinalLayer(256, num_classes)

    def forward(self, x):
        x = self.stem_net(x)

        # Placeholder for transition and fusion layers
        x = self.transition_layer1(x)
        x0 = self.branch1_0(x)
        x1 = self.branch1_1(x)
        x = self.fuse_layer1([x0, x1])

        x = self.transition_layer2(x)
        x0 = self.branch2_0(x)
        x1 = self.branch2_1(x)
        x2 = self.branch2_2(x)
        x = self.fuse_layer2([x0, x1, x2])

        x = self.transition_layer3(x)
        x0 = self.branch3_0(x)
        x1 = self.branch3_1(x)
        x2 = self.branch3_2(x)
        x3 = self.branch3_3(x)
        x = self.fuse_layer3([x0, x1, x2, x3])

        x = self.final_layer(x)
        return x