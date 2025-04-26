import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class UNetRes(nn.Module):
    def __init__(self):
        super().__init__()
        ch1, ch2, ch3, ch4 = 64, 128, 256, 512

        self.init_conv = nn.Sequential(nn.Conv2d(1,   ch1, 3, padding=1), nn.BatchNorm2d(ch1), nn.ReLU(inplace=True))
        self.enc2_map = nn.Sequential(nn.Conv2d(ch1, ch2, 3, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(inplace=True))
        self.enc3_map = nn.Sequential(nn.Conv2d(ch2, ch3, 3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU(inplace=True))
        self.bottle_map = nn.Sequential(nn.Conv2d(ch3, ch4, 3, padding=1), nn.BatchNorm2d(ch4), nn.ReLU(inplace=True))

        self.enc1 = ResBlock(ch1)
        self.enc2 = ResBlock(ch2)
        self.enc3 = ResBlock(ch3)
        self.bottleneck = ResBlock(ch4)
        self.reduce3 = nn.Conv2d(ch4+ch3, ch3, 1)
        self.reduce2 = nn.Conv2d(ch3+ch2, ch2, 1)
        self.reduce1 = nn.Conv2d(ch2+ch1, ch1, 1)
        self.dec3 = ResBlock(ch3)
        self.dec2 = ResBlock(ch2)
        self.dec1 = ResBlock(ch1)

        self.final = nn.Conv2d(ch1, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        x = self.init_conv(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.enc2_map(self.pool(e1)))
        e3 = self.enc3(self.enc3_map(self.pool(e2)))
        b  = self.bottleneck(self.bottle_map(self.pool(e3)))

        d3 = torch.cat([self.up(b), e3], dim=1)
        d3 = self.dec3(self.reduce3(d3))
        d2 = torch.cat([self.up(d3), e2], dim=1)
        d2 = self.dec2(self.reduce2(d2))
        d1 = torch.cat([self.up(d2), e1], dim=1)
        d1 = self.dec1(self.reduce1(d1))

        return torch.sigmoid(self.final(d1))
