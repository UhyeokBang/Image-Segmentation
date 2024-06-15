import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic Block
class BasicBlock(nn.Module):
    def __init__(self, in_channels):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),

            nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.block(x)
        out = self.block(x)
        out = self.block(x)
        out = self.block(x)
        return out

# Bottleneck Block
class BottleneckBlock(nn.Module):
    def __init__(self):
        super(BottleneckBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1, stride=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=1, stride=1,  bias=False),
            nn.BatchNorm2d(256),
        )

        self.firstblock = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,  bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(256),
        )

        self.ReLU = nn.ReLU()

    def forward(self, x):
        out = self.firstblock(x)
        out = self.ReLU(out)

        out = self.block(out)
        out = self.ReLU(out)

        out = self.block(out)
        out = self.ReLU(out)

        out = self.block(out)
        out = self.ReLU(out)

        return out

# Stem network
class StemNet(nn.Module):
    def __init__(self):
        super(StemNet, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,64,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
        

class Stage01StreamGenerateBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_res_block = nn.Sequential(
            nn.Conv2d(256,48,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.medium_res_block = nn.Sequential(
            nn.Conv2d(256,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
    def forward(self,inputs):
        out_high = self.high_res_block(inputs)
        out_med = self.medium_res_block(inputs)
        return out_high, out_med
    

class Stage02Fuse(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1,bias=False),
            nn.BatchNorm2d(48)
        )
        self.medium_to_small = nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high,inputs_medium):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])

        med2high = self.medium_to_high(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear",align_corners=True
        )
        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small(inputs_medium)
        high2med2small = self.medium_to_small(high2med)

        out_high = inputs_high + med2high
        out_med = inputs_medium + high2med
        out_small = med2small + high2med2small

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)

        return out_high, out_med, out_small
    
class Stage03Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1,bias=False),
            nn.BatchNorm2d(48)
        )
        self.medium_to_small = nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.small_to_medium = nn.Sequential(
            nn.Conv2d(192,96,kernel_size=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high,inputs_medium, inputs_small):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])

        med2high = self.medium_to_high(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear",align_corners=True
        )
        small2med = self.small_to_medium(inputs_small)
        small2med = F.interpolate(
            small2med, size = med_size, mode = "bilinear",align_corners=True
        )
        small2med2high = self.medium_to_high(small2med)
        small2med2high = F.interpolate(
            small2med2high, size = high_size,mode = "bilinear",align_corners=True
        )

        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small(inputs_medium)
        high2med2small = self.medium_to_small(high2med)

        out_high = inputs_high + med2high + small2med2high
        out_med = inputs_medium + high2med + small2med
        out_small = med2small + high2med2small + inputs_small

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)

        return out_high, out_med, out_small
    
class Stage03fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1,bias=False),
            nn.BatchNorm2d(48)
        )
        self.medium_to_small = nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.small_to_medium = nn.Sequential(
            nn.Conv2d(192,96,kernel_size=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.small_to_tiny = nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(384)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high,inputs_medium, inputs_small):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])

        med2high = self.medium_to_high(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear",align_corners=True
        )
        small2med = self.small_to_medium(inputs_small)
        small2med = F.interpolate(
            small2med, size = med_size, mode = "bilinear",align_corners=True
        )

        small2med2high = self.medium_to_high(small2med)
        small2med2high = F.interpolate(
            small2med2high, size = high_size,mode = "bilinear",align_corners=True
        )


        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small(inputs_medium)
        high2med2small = self.medium_to_small(high2med)
        high2tiny = self.small_to_tiny(high2med2small)
        med2tiny = self.small_to_tiny(med2small)
        small2tiny = self.small_to_tiny(inputs_small)

        out_high = inputs_high + med2high + small2med2high
        out_med = inputs_medium + high2med + small2med
        out_small = med2small + high2med2small + inputs_small
        out_tiny = high2tiny + med2tiny + small2tiny

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)
        out_tiny = self.relu(out_tiny)

        return out_high, out_med, out_small,out_tiny
    
class Stage04Reinforce(nn.Module):
    def __init__(self):
        super().__init__()
        self.high_to_medium = nn.Sequential(
            nn.Conv2d(48,96,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.medium_to_high = nn.Sequential(
            nn.Conv2d(96,48,kernel_size=1,bias=False),
            nn.BatchNorm2d(48)
        )
        self.medium_to_small = nn.Sequential(
            nn.Conv2d(96,192,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.small_to_medium = nn.Sequential(
            nn.Conv2d(192,96,kernel_size=1,bias=False),
            nn.BatchNorm2d(96)
        )
        self.small_to_tiny = nn.Sequential(
            nn.Conv2d(192,384,kernel_size=3,stride=2,padding=1,bias=False),
            nn.BatchNorm2d(384)
        )
        self.tiny_to_small = nn.Sequential(
            nn.Conv2d(384,192,kernel_size=1,bias=False),
            nn.BatchNorm2d(192)
        )
        self.relu = nn.ReLU()

    def forward(self, inputs_high,inputs_medium, inputs_small, inputs_tiny):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        med_size = (inputs_medium.shape[-1], inputs_medium.shape[-2])
        small_size = (inputs_small.shape[-1], inputs_small.shape[-2])

        med2high = self.medium_to_high(inputs_medium)
        med2high = F.interpolate(
            med2high, size = high_size, mode = "bilinear",align_corners=True
        )
        small2med = self.small_to_medium(inputs_small)
        small2med = F.interpolate(
            small2med, size = med_size, mode = "bilinear",align_corners=True
        )
        tiny2small = self.tiny_to_small(inputs_tiny)
        tiny2small = F.interpolate(
            tiny2small, size = small_size, mode = "bilinear",align_corners=True
        )

        small2med2high = self.medium_to_high(small2med)
        small2med2high = F.interpolate(
            small2med2high, size = high_size,mode = "bilinear",align_corners=True
        )

        tiny2med = self.small_to_medium(tiny2small)
        tiny2med = F.interpolate(
            tiny2med, size = med_size,mode = "bilinear",align_corners=True
        )
        tiny2high = self.medium_to_high(tiny2med)
        tiny2high = F.interpolate(
            tiny2high, size = high_size,mode = "bilinear",align_corners=True
        )

        high2med = self.high_to_medium(inputs_high)
        med2small = self.medium_to_small(inputs_medium)
        high2med2small = self.medium_to_small(high2med)
        high2tiny = self.small_to_tiny(high2med2small)
        med2tiny = self.small_to_tiny(med2small)
        small2tiny = self.small_to_tiny(inputs_small)


        out_high = inputs_high + med2high + small2med2high + tiny2high
        out_med = inputs_medium + high2med + small2med + tiny2med
        out_small = med2small + high2med2small + inputs_small + tiny2small
        out_tiny = high2tiny + med2tiny + small2tiny + inputs_tiny

        out_high = self.relu(out_high)
        out_med = self.relu(out_med)
        out_small = self.relu(out_small)
        out_tiny = self.relu(out_tiny)

        return out_high, out_med, out_small,out_tiny

class LastBlock(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        total_channel = 48+96+192+384
        self.block = nn.Sequential(
            nn.Conv2d(total_channel,total_channel,kernel_size=1,bias=False),
            nn.BatchNorm2d(total_channel),
            nn.ReLU(),
            nn.Conv2d(total_channel,num_classes,kernel_size=1,bias=False)
        )
    def forward(self,inputs_high, inputs_med, inputs_small, inputs_tiny):
        high_size = (inputs_high.shape[-1], inputs_high.shape[-2])
        origin_size = (high_size[0]*4,high_size[1]*4)

        med2high = F.interpolate(inputs_med, size = high_size, mode="bilinear", align_corners=True)
        small2high = F.interpolate(inputs_small, size = high_size, mode="bilinear", align_corners=True)
        tiny2high = F.interpolate(inputs_tiny, size = high_size, mode="bilinear", align_corners=True)

        out = torch.cat([inputs_high, med2high,small2high,tiny2high], dim=1)
        out = self.block(out)

        out = F.interpolate(out,size=origin_size,mode = "bilinear",align_corners=True)
        return out

# HRNet Segmentation Model
class HRNetSeg(nn.Module):
    def __init__(self):
        super(HRNetSeg, self).__init__()
        self.stem_net = StemNet()

        self.bottle = BottleneckBlock()

        self.highbasic = BasicBlock(48)
        self.medbasic = BasicBlock(96)
        self.smallbasic = BasicBlock(192)
        self.tinybasic = BasicBlock(384)

        self.firstGenBlock = Stage01StreamGenerateBlock()

        self.secondFusion = Stage02Fuse()

        self.thirdReinForce = Stage03Reinforce()

        self.thirdFusion = Stage03fusion()

        self.forthReinForce = Stage04Reinforce()

        self.lastBock = LastBlock(1)

    def forward(self, x):
        out = self.stem_net(x)
        print("stem",out.size())
        #stage 1
        out = self.bottle(out)
        print("stage bottle",out.size())
        high,med = self.firstGenBlock(out)
        print("firstGen",out.size())
        #stage 2
        high = self.highbasic(high)
        med = self.medbasic(med)

        high, med, small = self.secondFusion(high, med)


        #stage 3
        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)

        high, med, small = self.thirdReinForce(high,med,small)

        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)

        high, med, small = self.thirdReinForce(high,med,small)

        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)

        high, med, small = self.thirdReinForce(high,med,small)

        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)

        high, med, small, tiny = self.thirdFusion(high,med,small)


        #stage 4
        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)
        tiny = self.tinybasic(tiny)

        high, med, small, tiny = self.forthReinForce(high,med,small,tiny)

        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)
        tiny = self.tinybasic(tiny)

        high, med, small, tiny = self.forthReinForce(high,med,small,tiny)

        high = self.highbasic(high)
        med = self.medbasic(med)        
        small = self.smallbasic(small)
        tiny = self.tinybasic(tiny)

        high, med, small, tiny = self.forthReinForce(high,med,small,tiny)

        x = self.lastBock(high,med,small,tiny)
        return x