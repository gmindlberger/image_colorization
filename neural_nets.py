import torch.nn as nn
import torch

class ImageColorizer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ImageColorizer, self).__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # Decoder
        self.up1   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1  = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2  = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        # Aktivierungen
        self.relu   = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.enc1(x))
        x = self.pool1(x)
        x = self.relu(self.enc2(x))
        x = self.pool2(x)
        x = self.relu(self.enc3(x))
        x = self.up1(x)
        x = self.relu(self.dec1(x))
        x = self.up2(x)
        x = self.relu(self.dec2(x))
        x = self.sigmoid(self.final(x))
        return x

######################################################

class ImageSuperRes(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)  # (B, 64, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)  # (B, 64, H/2, W/2)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)  # (B, 128, H/2, W/2)
        self.pool2 = nn.MaxPool2d(2, 2)  # (B, 128, H/4, W/4)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)  # (B, 256, H/4, W/4)
        self.enc4 = nn.Conv2d(256, 512, 3, padding=1)  # (B, 512, H/4, W/4)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # (B, 256, H/2, W/2)
        self.dec1 = nn.Conv2d(256, 256, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # (B, 128, H, W)
        self.dec2 = nn.Conv2d(128, 128, 3, padding=1)

        # Final Conv
        self.dec3 = nn.Conv2d(128, 64, 3, padding=1)  # (B, 64, H, W)
        self.final = nn.Conv2d(64, 3, 3, padding=1)  # (B, 3, H, W)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x = self.relu(self.enc1(x))
        x = self.pool1(x)
        x = self.relu(self.enc2(x))
        x = self.pool2(x)
        x = self.relu(self.enc3(x))
        x = self.relu(self.enc4(x))

        # Decoder
        x = self.relu(self.up1(x))
        x = self.relu(self.dec1(x))
        x = self.relu(self.up2(x))
        x = self.relu(self.dec2(x))
        x = self.relu(self.dec3(x))
        x = self.sigmoid(self.final(x))

        return x

######################################################

def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        ch1, ch2, ch3, ch4 = (64, 128, 256, 512)
        # encoder
        self.enc1 = conv_block(1, ch1)
        self.enc2 = conv_block(ch1, ch2)
        self.enc3 = conv_block(ch2, ch3)
        # bottleneck
        self.bottleneck = conv_block(ch3, ch4)
        # decoder
        self.dec3 = conv_block(ch4 + ch3, ch3)
        self.dec2 = conv_block(ch3 + ch2, ch2)
        self.dec1 = conv_block(ch2 + ch1, ch1)

        # final projection
        self.final = nn.Conv2d(ch1, 3, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1))  # cat  -> (B, 512+256=768, H/4, W/4)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1)) # cat  -> (B, 256+128=384, H/2, W/2)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1)) # cat  -> (B, 128+ 64=192, H,   W)

        return torch.sigmoid(self.final(d1))

######################################################

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
