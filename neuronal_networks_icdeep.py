import torch.nn as nn


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
