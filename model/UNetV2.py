import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.optim as optim


class SatelliteDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".tif", "_FGT.tif"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# 데이터 경로 설정
image_dir = "C:/Users/s_zmfldlwx/Desktop/2024-1학기/OSSP-1/팀 프로젝트/New_Sample/2.원천데이터/1.항공사진_Fine_512픽셀"
mask_dir = "C:/Users/s_zmfldlwx/Desktop/2024-1학기/OSSP-1/팀 프로젝트/New_Sample/1.라벨링데이터/1.항공사진_Fine_512픽셀/1.Ground_Truth_Tiff"

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 데이터셋과 데이터로더 생성
dataset = SatelliteDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

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

        self.upconv3 = UpConvLayer2d(512, 256)
        self.decoder3 = ConvLayer2d(512, 256)

        self.upconv2 = UpConvLayer2d(256, 128)
        self.decoder2 = ConvLayer2d(256, 128)

        self.upconv1 = UpConvLayer2d(128, 64)
        self.decoder1 = ConvLayer2d(128, 64)

        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=1)

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
        c4 = torch.cat((u4, e4), dim=1)
        d4 = self.decoder4(c4)

        u3 = self.upconv3(d4)
        c3 = torch.cat((u3, e3), dim=1)
        d3 = self.decoder3(c3)

        u2 = self.upconv2(d3)
        c2 = torch.cat((u2, e2), dim=1)
        d2 = self.decoder2(c2)

        u1 = self.upconv1(d2)
        c1 = torch.cat((u1, e1), dim=1)
        d1 = self.decoder1(c1)

        output = self.output_conv(d1)

        return output

# 모델 생성
in_channels = 3  # RGB 위성 사진
out_channels = 9  # 9개 클래스

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 모델 생성 및 GPU로 이동
model = UNet(in_channels=3, out_channels=9).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        images = images.to(device)
        masks = masks.squeeze(1).to(device)  # 차원 줄이기
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training Finished.")

# 디렉토리 확인 및 생성
save_dir = './../model_dict_save'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 모델 저장
torch.save(model.state_dict(), os.path.join(save_dir, 'unet_model.pth'))