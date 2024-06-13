import torch
import torch.nn as nn

from model.backbone import Xception

class ASPP(nn.Module):
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, _in_channels, _out_channels):
        super(ASPP, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=True):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU())
        
        def AtrousConvLayer2d(in_channels, out_channels, kernel_size=3, padding=0, bias=True, dilation=0):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=1, bias=bias, dilation = dilation),
                                 nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU())
        
        dilations = [1, 6, 12, 18]

        self.aspp1 = AtrousConvLayer2d(_in_channels, _out_channels, 1, dilation=dilations[0], bias=False, padding=0)
        self.aspp2 = AtrousConvLayer2d(_in_channels, _out_channels, 3, dilation=dilations[1], bias=False, padding=dilations[1])
        self.aspp3 = AtrousConvLayer2d(_in_channels, _out_channels, 3, dilation=dilations[2], bias=False, padding=dilations[2])
        self.aspp4 = AtrousConvLayer2d(_in_channels, _out_channels, 3, dilation=dilations[3], bias=False, padding=dilations[3])
        
        self.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), CBR2d(_in_channels, _out_channels, kernel_size=1, bias=False))

        # 1x1 Convolution 적용
        self.conv = CBR2d(_out_channels * 5, _out_channels, kernel_size=1, bias=False)

        self.dropout = nn.Dropout(0.5)

        self._init_weight()

    def forward(self, input): 
        aspp1 = self.aspp1(input)
        aspp2 = self.aspp2(input)
        aspp3 = self.aspp3(input)
        aspp4 = self.aspp4(input)
        
        pool = self.pool(input)

        pool = nn.functional.interpolate(pool, size=aspp4.size()[2:], mode='bilinear', align_corners=True)
        
        cat = torch.cat((aspp1, aspp2, aspp3, aspp4, pool), dim=1)
      
        # 1x1 Convolution 적용
        conv = self.conv(cat)

        output = self.dropout(conv)

        return output

class Decoder(nn.Module):
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def __init__(self, num_classes):
        super(Decoder, self).__init__()
        
        self.num_classes = num_classes

        def CBR2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=True):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU())
        
        self.low_level_conv = CBR2d(128, 48, kernel_size=1, bias=False)
        
        self.conv = nn.Sequential(CBR2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
                                  nn.Dropout(0.5),
                                  CBR2d(256, 256, kernel_size=3, padding=1, bias=False),
                                  nn.Dropout(0.1),
                                  nn.Conv2d(256, num_classes, kernel_size=1, stride=1))

        self._init_weight()

    def forward(self, aspp, low_level):
        low_level = self.low_level_conv(low_level)

        aspp = nn.functional.interpolate(aspp, scale_factor=4, mode='bilinear', align_corners=True)

        cat = torch.cat((aspp, low_level), dim=1)

        conv = self.conv(cat)

        output = nn.functional.interpolate(conv, scale_factor=4, mode='bilinear', align_corners=True)

        return output

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes):
        super(DeepLabV3Plus, self).__init__()
        
        # Backbone(Xception)
        self.backbone = Xception.AlignedXception(16, nn.BatchNorm2d)
        
        # Atrous Spatial Pyramid Pooling
        self.aspp = ASPP(2048, 256)
        
        # Decoder
        self.decoder = Decoder(num_classes)

    def forward(self, input):
        # Backbone(Xception)
        e, low_level = self.backbone(input)

        # Atrous Spatial Pyramid Pooling
        aspp = self.aspp(e)

        # Decoder
        d = self.decoder(aspp, low_level)

        output = d

        return output
    
    