import torch
import torch.nn as nn


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


# Shapes:
#     enc1: (B, 1, H, W) -> (B, 64, H, W)
#     enc2: (B, 64, H, W) -> (B, 128, H/2, W/2)
#     enc3: (B, 128, H/2, W/2) -> (B, 256, H/4, W/4)
#     bottleneck: (B, 256, H/4, W/4) -> (B, 512, H/8, W/8)
#     dec3: (B, 512, H/8, W/8) -> (B, 256, H/4, W/4)
#     dec2: (B, 256, H/4, W/4) -> (B, 128, H/2, W/2)
#     dec1: (B, 128, H/2, W/2) -> (B, 64, H, W)
#     final: (B, 64, H, W) -> (B, 3, H, W)
#     output: (B, 3, H, W)
