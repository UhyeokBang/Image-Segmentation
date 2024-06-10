import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np

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
out_channels = 1  # 그레이스케일 분류 결과
model = UNet(in_channels, out_channels)

# 모델 요약
print(model)

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
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask

# 데이터 전처리
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# 데이터셋과 데이터로더 생성
image_dir = "/path/to/satellite/images"
mask_dir = "/path/to/masks"
dataset = SatelliteDataset(image_dir, mask_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 25
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks.long())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader)}")

print("Training Finished.")

# 학습된 모델을 로드합니다 (이미 학습된 모델 파일이 있는 경우)
model_path = 'path/to/saved/model.pth'
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(model_path))
model.eval()

# 입력 이미지 로드 및 전처리
input_image_path = 'path/to/input/image.png'
input_image = Image.open(input_image_path).convert('RGB')
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])
input_tensor = transform(input_image).unsqueeze(0)  # 배치 차원 추가

# 모델 예측
with torch.no_grad():
    output = model(input_tensor)
    output = output.squeeze(0)  # 배치 차원 제거
    output = torch.sigmoid(output)  # 활성화 함수 적용 (필요에 따라)

# 결과를 Numpy 배열로 변환하고 정규화
output_np = output.cpu().numpy()
output_np = (output_np * 255).astype(np.uint8)  # 정규화 (0-255 스케일)

# 라벨링 결과를 Tiff 이미지로 저장
output_image = Image.fromarray(output_np[0])  # 1채널 이미지로 변환
output_image.save('path/to/output/labeled_image.tiff')

print("라벨링 이미지가 저장되었습니다.")