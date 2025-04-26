import torch.nn as nn

class ImageColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.enc3 = nn.Conv2d(128, 256, 3, padding=1)
        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Conv2d(256, 128, 3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Conv2d(128, 64, 3, padding=1)
        self.final = nn.Conv2d(64, 3, 3, padding=1)
        self.relu = nn.ReLU()
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

