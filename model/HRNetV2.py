import torch
import torch.nn as nn
import torch.nn.functional as F

def CBR2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=True):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias), nn.BatchNorm2d(num_features=out_channels), nn.ReLU())
     
def ConvLayer2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=True):
    return nn.Sequential(
        CBR2d(in_channels, out_channels, kernel_size, padding, stride, bias), 
        CBR2d(out_channels, out_channels, kernel_size, padding, stride, bias)
    )

class Interpolate(nn.Module):
    def __init__(self, size, mode='bilinear', align_corners=True):
        super(Interpolate, self).__init__()
        self.size = size
        self.mode = mode
        self.align_corners = align_corners

        self.func = nn.functional.interpolate
        
    def forward(self, x):
        x = self.func(x, size=self.size, mode=self.mode, align_corners=self.align_corners)

        return x

# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.block1 = ConvLayer2d(in_channels, in_channels, padding=1, bias=False)
        self.block2 = ConvLayer2d(in_channels, in_channels, padding=1, bias=False)
        self.block3 = ConvLayer2d(in_channels, in_channels, padding=1, bias=False)
        self.block4 = ConvLayer2d(in_channels, in_channels, padding=1, bias=False)

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(x)
        out = self.block3(x)
        out = self.block4(x)

        return out

# Bottleneck Block
class BottleneckBlock(nn.Module):
    def __init__(self):
        super(BottleneckBlock, self).__init__()

        def BottleneckConvLayer2d(in_channels, b_channels, out_channels, bias=True):
            return nn.Sequential(
                CBR2d(in_channels, b_channels, kernel_size=1, stride=1, bias=bias), 
                CBR2d(b_channels, b_channels, kernel_size=3, stride=1, padding=1, bias=bias), 
                CBR2d(b_channels, out_channels, kernel_size=1, stride=1, bias=bias)
            )
        
        self.block1 = BottleneckConvLayer2d(256, 64, 256, bias=False)
        self.block2 = BottleneckConvLayer2d(256, 64, 256, bias=False)
        self.block3 = BottleneckConvLayer2d(256, 64, 256, bias=False)

        self.first = nn.Sequential(
            CBR2d(64, 64, kernel_size=1, stride=1, bias=False), 
            CBR2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), 
            CBR2d(64, 256, kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        out = self.first(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        return out

# Stem network
class StemNet(nn.Module):
    def __init__(self):
        super(StemNet, self).__init__()

        self.block = ConvLayer2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x):
        return self.block(x)

# High(HI) - Medium(MD) - Small(SM) - Tiny(TN)
def high_to_medium():
    return nn.Sequential(
        nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(96)
    )

def medium_to_high():
    return nn.Sequential(
        nn.Conv2d(96, 48, kernel_size=1, bias=False), 
        nn.BatchNorm2d(48)
    )

def medium_to_small():
    return nn.Sequential(
        nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(192)
    )

def small_to_medium():
    return nn.Sequential(
        nn.Conv2d(192, 96, kernel_size=1, bias=False), 
        nn.BatchNorm2d(96)
    )

def small_to_tiny():
    return nn.Sequential(
        nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1, bias=False), 
        nn.BatchNorm2d(384)
    )

def tiny_to_small():
    return nn.Sequential(
        nn.Conv2d(384, 192, kernel_size=1, bias=False), 
        nn.BatchNorm2d(192)
    )

# For Stage
class Stage01StreamGenerateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.h_res_block = CBR2d(256, 48, kernel_size=3, padding=1, bias=False)
        self.m_res_block = CBR2d(256, 96, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, inputs):
        out_h = self.h_res_block(inputs)
        out_m = self.m_res_block(inputs)

        return out_h, out_m

class Stage02Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = high_to_medium()
        
        self.medium_to_high = medium_to_high()
        
        self.medium_to_small1 = medium_to_small()
        self.medium_to_small2 = medium_to_small()

        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])

        high2med = self.high_to_medium(inputs_high)
        high2small = self.medium_to_small1(high2med)

        med2high = self.medium_to_high(inputs_medium)
        med2high = F.interpolate(med2high, size = high_size, mode = "bilinear", align_corners=True)

        med2small = self.medium_to_small2(inputs_medium)

        out_high = inputs_high + med2high
        out_med = inputs_medium + high2med
        out_small = med2small + high2small

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)

        return out_high, out_med, out_small
    
class Stage03Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = high_to_medium()
        
        self.medium_to_high1 = medium_to_high()
        self.medium_to_high2 = medium_to_high()
        
        self.medium_to_small1 = medium_to_small()
        self.medium_to_small2 = medium_to_small()

        self.small_to_medium = small_to_medium()

        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium, inputs_small):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])

        med2high = self.medium_to_high1(inputs_medium)
        med2high = F.interpolate(med2high, size = high_size, mode = "bilinear", align_corners=True)

        small2med = self.small_to_medium(inputs_small)
        small2med = F.interpolate(small2med, size = med_size, mode = "bilinear", align_corners=True)

        small2high = self.medium_to_high2(small2med)
        small2high = F.interpolate(small2high, size = high_size, mode = "bilinear", align_corners=True)

        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small1(inputs_medium)
        high2med2small = self.medium_to_small2(high2med)

        out_high = inputs_high + med2high + small2high
        out_med = inputs_medium + high2med + small2med
        out_small = med2small + high2med2small + inputs_small

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)

        return out_high, out_med, out_small
    
class Stage03Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = high_to_medium()
        
        self.medium_to_high1 = medium_to_high()
        self.medium_to_high2 = medium_to_high()

        self.medium_to_small1 = medium_to_small()
        self.medium_to_small2 = medium_to_small()
        
        self.small_to_medium = small_to_medium()
        
        self.small_to_tiny1 = small_to_tiny()
        self.small_to_tiny2 = small_to_tiny()
        self.small_to_tiny3 = small_to_tiny()

        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium, inputs_small):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])

        med2high = self.medium_to_high1(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear", align_corners=True
        )
        small2med = self.small_to_medium(inputs_small)
        small2med = F.interpolate(
            small2med, size = med_size, mode = "bilinear", align_corners=True
        )

        small2med2high = self.medium_to_high2(small2med)
        small2med2high = F.interpolate(
            small2med2high, size = high_size, mode = "bilinear", align_corners=True
        )

        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small1(inputs_medium)
        high2med2small = self.medium_to_small2(high2med)

        high2tiny = self.small_to_tiny1(high2med2small)
        med2tiny = self.small_to_tiny2(med2small)
        small2tiny = self.small_to_tiny3(inputs_small)

        out_high = inputs_high + med2high + small2med2high
        out_med = inputs_medium + high2med + small2med
        out_small = med2small + high2med2small + inputs_small
        out_tiny = high2tiny + med2tiny + small2tiny

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)
        out_tiny = self.relu(out_tiny)

        return out_high, out_med, out_small, out_tiny
    
class Stage04Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = high_to_medium()

        self.medium_to_high1 = medium_to_high()
        self.medium_to_high2 = medium_to_high()
        self.medium_to_high3 = medium_to_high()

        self.medium_to_small1 = medium_to_small()
        self.medium_to_small2 = medium_to_small()

        self.small_to_medium1 = small_to_medium()
        self.small_to_medium2 = small_to_medium()

        self.small_to_tiny1 = small_to_tiny()
        self.small_to_tiny2 = small_to_tiny()
        self.small_to_tiny3 = small_to_tiny()

        self.tiny_to_small = tiny_to_small()

        self.relu = nn.ReLU()

    def forward(self, inputs_high, inputs_medium, inputs_small, inputs_tiny):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])
        small_size = (inputs_small.shape[-1], inputs_small.shape[-2])

        med2high = self.medium_to_high1(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear", align_corners=True
        )
        small2med = self.small_to_medium1(inputs_small)
        small2med = F.interpolate(
            small2med, size = med_size, mode = "bilinear", align_corners=True
        )
        tiny2small = self.tiny_to_small(inputs_tiny)
        tiny2small = F.interpolate(
            tiny2small, size = small_size, mode = "bilinear", align_corners=True
        )

        small2med2high = self.medium_to_high2(small2med)
        small2med2high = F.interpolate(
            small2med2high, size = high_size, mode = "bilinear", align_corners=True
        )

        tiny2med = self.small_to_medium2(tiny2small)
        tiny2med = F.interpolate(
            tiny2med, size = med_size, mode = "bilinear", align_corners=True
        )
        tiny2high = self.medium_to_high3(tiny2med)
        tiny2high = F.interpolate(
            tiny2high, size = high_size, mode = "bilinear", align_corners=True
        )

        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small1(inputs_medium)
        high2med2small = self.medium_to_small2(high2med)
        high2tiny = self.small_to_tiny1(high2med2small)
        med2tiny = self.small_to_tiny2(med2small)
        small2tiny = self.small_to_tiny3(inputs_small)

        out_high = inputs_high + med2high + small2med2high + tiny2high
        out_med = inputs_medium + high2med + small2med + tiny2med
        out_small = med2small + high2med2small + inputs_small + tiny2small
        out_tiny = high2tiny + med2tiny + small2tiny + inputs_tiny

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)
        out_tiny = self.relu(out_tiny)

        return out_high, out_med, out_small, out_tiny

class LastBlock(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        total_channel = 48+96+192+384
        self.block = nn.Sequential(
            CBR2d(total_channel, total_channel, kernel_size=1, bias=False), 
            nn.Conv2d(total_channel, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, inputs_high, inputs_med, inputs_small, inputs_tiny):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        origin_size = (high_size[0]*4, high_size[1]*4)

        med2high = F.interpolate(inputs_med, size = high_size, mode="bilinear", align_corners=True)
        small2high = F.interpolate(inputs_small, size = high_size, mode="bilinear", align_corners=True)
        tiny2high = F.interpolate(inputs_tiny, size = high_size, mode="bilinear", align_corners=True)

        out = torch.cat([inputs_high, med2high, small2high, tiny2high], dim=1)
        out = self.block(out)

        out = F.interpolate(out, size=origin_size, mode = "bilinear", align_corners=True)

        return out

class Stage03Block(nn.Module):
    def __init__(self):
        super(Stage03Block, self).__init__()

        self.highbasic = BasicBlock(48)
        self.medbasic = BasicBlock(96)
        self.smallbasic = BasicBlock(192)

        self.reinforce = Stage03Reinforce()
 
    def forward(self, high, med, small): 
        
        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)

        return self.reinforce(high, med, small)

class Stage04Block(nn.Module):
    def __init__(self):
        super(Stage04Block, self).__init__()

        self.highbasic = BasicBlock(48)
        self.medbasic = BasicBlock(96)
        self.smallbasic = BasicBlock(192)
        self.tinybasic = BasicBlock(384)

        self.reinforce = Stage04Reinforce()
 
    def forward(self, high, med, small, tiny): 
        
        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)
        tiny = self.tinybasic(tiny)

        return self.reinforce(high, med, small, tiny)

# HRNet Segmentation Model
class HRNetV2(nn.Module):
    def __init__(self, num_classes):
        super(HRNetV2, self).__init__()
        self.stem_net = StemNet()

        self.bottle = BottleneckBlock()

        self.highbasic2 = BasicBlock(48)
        self.medbasic2 = BasicBlock(96)

        self.highbasic3 = BasicBlock(48)
        self.medbasic3 = BasicBlock(96)
        self.smallbasic3 = BasicBlock(192)

        self.firstGenBlock = Stage01StreamGenerateBlock()

        self.secondFusion = Stage02Fusion()

        self.thirdBlock1 = Stage03Block()
        self.thirdBlock2 = Stage03Block()
        self.thirdBlock3 = Stage03Block()

        self.thirdFusion = Stage03Fusion()

        self.fourthBlock1 = Stage04Block()
        self.fourthBlock2 = Stage04Block()
        self.fourthBlock3 = Stage04Block()

        self.lastBlock = LastBlock(num_classes)

    def forward(self, x):
        out = self.stem_net(x)
        #stage 1
        out = self.bottle(out)
        high, med = self.firstGenBlock(out)
        #stage 2
        high = self.highbasic2(high)
        med = self.medbasic2(med)

        high, med, small = self.secondFusion(high, med)

        #stage 3
        high, med, small = self.thirdBlock1(high, med, small)
        high, med, small = self.thirdBlock2(high, med, small)
        high, med, small = self.thirdBlock3(high, med, small)

        high = self.highbasic3(high)
        med = self.medbasic3(med)        
        small = self.smallbasic3(small)

        high, med, small, tiny = self.thirdFusion(high, med, small)

        #stage 4
        high, med, small, tiny = self.fourthBlock1(high, med, small, tiny)
        high, med, small, tiny = self.fourthBlock2(high, med, small, tiny)
        high, med, small, tiny = self.fourthBlock3(high, med, small, tiny)

        x = self.lastBlock(high, med, small, tiny)
        
        return x