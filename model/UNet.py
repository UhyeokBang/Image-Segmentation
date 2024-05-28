import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        def CBR2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=True):
            return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(num_features=out_channels),
                                 nn.ReLU())
        
        def ConvLayer2d(in_channels, out_channels, kernel_size=3, padding=0, stride=1, bias=True):
            return nn.Sequential(CBR2d(in_channels,  out_channels, kernel_size, padding, stride, bias),
                                 CBR2d(out_channels, out_channels, kernel_size, padding, stride, bias))
        
        def PoolLayer2d(kernel_size=2, stride=2):
            return nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

        def UpConvLayer2d(in_channels, out_channels, kernel_size=3, stride=1):
            return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

        self.encoder1 = ConvLayer2d(  1,  64)
        self.pool1 = PoolLayer2d(2, 2)

        self.encoder2 = ConvLayer2d( 64, 128)
        self.pool2 = PoolLayer2d(2, 2)

        self.encoder3 = ConvLayer2d(128, 256)
        self.pool3 = PoolLayer2d(2, 2)

        self.encoder4 = ConvLayer2d(256, 512)
        self.pool4 = PoolLayer2d(2, 2)

        self.bridge = nn.Sequential(ConvLayer2d(512, 1024), nn.Dropout(0.5))

        self.upconv4 = UpConvLayer2d(1024, 1024)
        self.decoder4 = ConvLayer2d(1024, 512)

        self.upconv3 = UpConvLayer2d(1024, 1024)
        self.decoder3 = ConvLayer2d( 512, 256)

        self.upconv2 = UpConvLayer2d(1024, 1024)
        self.decoder2 = ConvLayer2d( 256, 128)

        self.upconv1 = UpConvLayer2d(1024, 1024)
        self.decoder1 = ConvLayer2d( 128,  64)

        self.fc = nn.Conv2d(64, 1, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, input):
        def encode(encoder, pool, input):
            e = encoder(input)
            p = pool(e)

            return (e, p)

        def decode(upconv, e, decoder, input):
            u = upconv(input)
            # Skip-Connection 수행
            c = torch.cat((e, u), dim=1)
            d = decoder(c)

            return(u, c, d)

        # Encoder 레이어 4단계 수행
        (e1, p1) = encode(self.encoder1, self.pool1, input) 
        (e2, p2) = encode(self.encoder2, self.pool2, p1) 
        (e3, p3) = encode(self.encoder3, self.pool3, p2) 
        (e4, p4) = encode(self.encoder4, self.pool4, p3) 

        # BottleNeck 레이어 수행
        bridge = self.bridge(p4)

        # Decoder 레이어 4단계 수행
        (_, _, d4) = decode(self.upconv4, e4, self.decoder4, bridge)
        (_, _, d3) = decode(self.upconv3, e3, self.decoder3, d4)
        (_, _, d2) = decode(self.upconv2, e2, self.decoder2, d3)
        (_, _, d1) = decode(self.upconv1, e1, self.decoder1, d2)

        # 클래스 분류
        output = self.fc(d1) 

        return output