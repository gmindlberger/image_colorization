import torch.nn as nn
import torch
import torchvision.models as models

def count_params(model):
    '''
    returns the number of trainable parameters in some model
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


######################################################

class ImageColorizerLAB(nn.Module):
    """
    Simple encoder-decoder for LAB colorization (inputs/outputs in [-1,1]).

    Args:
        in_channels:   Anzahl der Eingangskanäle (z.B. 1 für L, 3 für L+A+B)
        out_channels:  Anzahl der Ausgabe­kanäle (z.B. 2 für A+B, 3 für L+A+B)
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.enc1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.enc3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)

        self.up1   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1  = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.up2   = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2  = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.final = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,C_in,H,W) -> y: (B,C_out,H,W) in [-1,1]."""
        x = self.relu(self.enc1(x))
        x = self.pool1(x)
        x = self.relu(self.enc2(x))
        x = self.pool2(x)
        x = self.relu(self.enc3(x))

        x = self.up1(x)
        x = self.relu(self.dec1(x))
        x = self.up2(x)
        x = self.relu(self.dec2(x))
        return self.tanh(self.final(x))


######################################################

class ImageColorizer(nn.Module):
    """
    Simple encoder-decoder for image colorization with customizable channels.

    Args:
        in_channels:   Number of input channels (e.g., 1 for grayscale L, 3 for RGB).
        out_channels:  Number of output channels (e.g., 2 for AB color channels, 3 for full RGB).
    """
    def __init__(self, in_channels: int, out_channels: int):
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
        # Activations
        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, in_channels, H, W).
        Returns:
            Output tensor of shape (B, out_channels, H, W) in [0,1].
        """
        x = self.relu(self.enc1(x))
        x = self.pool1(x)
        x = self.relu(self.enc2(x))
        x = self.pool2(x)
        x = self.relu(self.enc3(x))
        x = self.up1(x)
        x = self.relu(self.dec1(x))
        x = self.up2(x)
        x = self.relu(self.dec2(x))
        return self.sigmoid(self.final(x))

######################################################

class ImageSuperRes(nn.Module):
    """
    Simple encoder-decoder super-resolution network.

    Args:
        in_channels:   Number of input channels (e.g., 1 for grayscale, 3 for RGB).
        out_channels:  Number of output channels (e.g., 3 for RGB output).
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(in_channels,  64, 3, padding=1)   # (B, 64, H, W)
        self.pool1 = nn.MaxPool2d(2, 2)                          # (B, 64, H/2, W/2)
        self.enc2 = nn.Conv2d(64,            128, 3, padding=1) # (B,128,H/2,W/2)
        self.pool2 = nn.MaxPool2d(2, 2)                          # (B,128,H/4,W/4)
        self.enc3 = nn.Conv2d(128,          256, 3, padding=1)  # (B,256,H/4,W/4)
        self.enc4 = nn.Conv2d(256,          512, 3, padding=1)  # (B,512,H/4,W/4)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)  # (B,256,H/2,W/2)
        self.dec1 = nn.Conv2d(256, 256, 3, padding=1)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)  # (B,128,H,   W)
        self.dec2 = nn.Conv2d(128, 128, 3, padding=1)

        # Final Conv
        self.dec3  = nn.Conv2d(128,  64, 3, padding=1)                 # (B, 64, H, W)
        self.final = nn.Conv2d(64,   out_channels, 3, padding=1)       # (B, out_channels, H, W)

        self.relu    = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.sigmoid(self.final(x))

######################################################

def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class UNet(nn.Module):
    """
    Simple U-Net for image-to-image tasks.

    Args:
        in_channels:   Number of input channels.
        out_channels:  Number of output channels.
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3):
        super().__init__()
        ch1, ch2, ch3, ch4 = (64, 128, 256, 512)
        # encoder
        self.enc1 = conv_block(in_channels, ch1)
        self.enc2 = conv_block(ch1, ch2)
        self.enc3 = conv_block(ch2, ch3)
        # bottleneck
        self.bottleneck = conv_block(ch3, ch4)
        # decoder
        self.dec3 = conv_block(ch4 + ch3, ch3)
        self.dec2 = conv_block(ch3 + ch2, ch2)
        self.dec1 = conv_block(ch2 + ch1, ch1)

        # final projection
        self.final = nn.Conv2d(ch1, out_channels, kernel_size=1)

        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x) # -> (B, 64, 128, 128)
        e2 = self.enc2(self.pool(e1)) # -> (B, 128, 64, 64)
        e3 = self.enc3(self.pool(e2)) # -> (B, 256, 32, 32)
        b  = self.bottleneck(self.pool(e3)) # -> (B, 512, 16, 16)
        d3 = self.dec3(torch.cat([self.up(b), e3], dim=1)) # (B, 256, 32, 32)
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1)) # (B, 128, 64, 64)
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1)) # (B, 64, 128, 128)
        return torch.sigmoid(self.final(d1)) # -> (B, out_channels, 128, 128)


######################################################

class ResBlock(nn.Module):
    """
    Residual block with two conv layers and a skip connection.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(channels)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class UNetRes(nn.Module):
    """
    U-Net with residual blocks.

    Args:
        in_channels:   Number of input channels (e.g., 1 for L or grayscale).
        out_channels:  Number of output channels (e.g., 2 for AB channels, 3 for RGB).
    """
    def __init__(self,
                 in_channels: int = 1,
                 out_channels: int = 3):
        super().__init__()
        # define channel sizes
        ch1, ch2, ch3, ch4 = 64, 128, 256, 512

        # Initial conv block
        # maps in_channels -> ch1
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, ch1, 3, padding=1),
            nn.BatchNorm2d(ch1),
            nn.ReLU(inplace=True)
        )
        # Encoder mapping layers
        self.enc2_map  = nn.Sequential(nn.Conv2d(ch1, ch2, 3, padding=1), nn.BatchNorm2d(ch2), nn.ReLU(inplace=True))
        self.enc3_map  = nn.Sequential(nn.Conv2d(ch2, ch3, 3, padding=1), nn.BatchNorm2d(ch3), nn.ReLU(inplace=True))
        self.bottle_map = nn.Sequential(nn.Conv2d(ch3, ch4, 3, padding=1), nn.BatchNorm2d(ch4), nn.ReLU(inplace=True))

        # Residual encoder blocks
        self.enc1      = ResBlock(ch1)
        self.enc2      = ResBlock(ch2)
        self.enc3      = ResBlock(ch3)
        self.bottleneck = ResBlock(ch4)

        # 1x1 reduces after concatenation in decoder
        self.reduce3 = nn.Conv2d(ch4 + ch3, ch3, 1)
        self.reduce2 = nn.Conv2d(ch3 + ch2, ch2, 1)
        self.reduce1 = nn.Conv2d(ch2 + ch1, ch1, 1)

        # Residual decoder blocks
        self.dec3 = ResBlock(ch3)
        self.dec2 = ResBlock(ch2)
        self.dec1 = ResBlock(ch1)

        # Final output conv
        # maps ch1 -> out_channels
        self.final = nn.Conv2d(ch1, out_channels, 1)

        # pooling and upsampling
        self.pool = nn.MaxPool2d(2)
        self.up   = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder path
        x = self.init_conv(x) # -> (B, 64, 128, 128)
        e1 = self.enc1(x) # -> (B, 64, 128, 128)
        e2 = self.enc2(self.enc2_map(self.pool(e1))) # -> (B, 128, 64, 64)
        e3 = self.enc3(self.enc3_map(self.pool(e2))) # -> (B, 256, 32, 32)
        b  = self.bottleneck(self.bottle_map(self.pool(e3))) # -> (B, 512, 16, 16)

        # Decoder path with skip connections
        d3 = torch.cat([self.up(b), e3], dim=1) # (B, 512 + 256 = 768, 32, 32)
        d3 = self.dec3(self.reduce3(d3)) # -> (B, 256, 32, 32)
        d2 = torch.cat([self.up(d3), e2], dim=1) # (B, 256 + 128 = 384, 64, 64)
        d2 = self.dec2(self.reduce2(d2)) # -> (B, 128, 64, 64)
        d1 = torch.cat([self.up(d2), e1], dim=1) # (B, 128 + 64 = 192, 128, 128)
        d1 = self.dec1(self.reduce1(d1)) # -> (B, 64, 128, 128)

        # Output in [0,1]
        return torch.sigmoid(self.final(d1)) # -> (B, out_channels, 128, 128)

#######################################################

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, upsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 5, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.upsample = upsample
        self.activation = activation or nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.res_conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            self.res_bn = nn.BatchNorm2d(out_channels)
        else:
            self.res_conv = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.res_conv:
            identity = self.res_bn(self.res_conv(identity))

        out += identity

        if self.upsample:
            out = self.relu(out)
            out = self.upsample(out)
        else:
            out = self.activation(out)

        return out


class ColorizeNet(nn.Module):
    """
    Colorization network: grayscale L-channel to AB color channels.

    Architecture:
      - Encoder: first 3 stages of pretrained ResNet-18 (conv1 to layer2).
      - Decoder: three BasicBlock layers with upsampling.

    Args:
        in_channels:  Input channels (1 for L-channel).
        out_channels: Output channels (2 for AB channels).
    """
    def __init__(self, in_channels: int = 1, out_channels: int = 2):
        super().__init__()
        # load ResNet-18 encoder
        resnet18 = models.resnet18(pretrained=True)
        # adapt first conv to in_channels
        weight = resnet18.conv1.weight.mean(dim=1, keepdim=True)
        resnet18.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7,
                                   stride=2, padding=3, bias=False)
        resnet18.conv1.weight = nn.Parameter(weight)
        # use layers up to layer2
        self.encoder = nn.Sequential(*list(resnet18.children())[:6])
        # decoder blocks
        self.decoder = nn.Sequential(
            self._make_layer(BasicBlock, 128, 64, upsample=nn.Upsample(scale_factor=2)),
            self._make_layer(BasicBlock, 64, 32, upsample=nn.Upsample(scale_factor=2)),
            self._make_layer(BasicBlock, 32, out_channels, activation=nn.Sigmoid())
        )

    def _make_layer(self, block, in_ch, out_ch, activation=None, upsample=None):
        layers = []
        layers.append(block(in_ch, out_ch, activation=None, upsample=upsample))
        layers.append(block(out_ch, out_ch, activation=activation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        # Upscale to match original image size
        x = nn.functional.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)
        return x
#######################################################