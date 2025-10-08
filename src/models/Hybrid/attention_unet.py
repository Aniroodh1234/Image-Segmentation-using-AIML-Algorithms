"""
Attention U-Net implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionBlock(nn.Module):
    """Attention block for Attention U-Net"""
    
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class ConvBlock(nn.Module):
    """Convolutional block"""
    
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class AttentionUNet(nn.Module):
    """Attention U-Net for image segmentation"""
    
    def __init__(self, n_channels=3, n_classes=5):
        super(AttentionUNet, self).__init__()
        
        # Encoder
        self.conv1 = ConvBlock(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.conv3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.conv4 = ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.conv5 = ConvBlock(512, 1024)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionBlock(F_g=512, F_l=512, F_int=256)
        self.upconv1 = ConvBlock(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(F_g=256, F_l=256, F_int=128)
        self.upconv2 = ConvBlock(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(F_g=128, F_l=128, F_int=64)
        self.upconv3 = ConvBlock(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(F_g=64, F_l=64, F_int=32)
        self.upconv4 = ConvBlock(128, 64)
        
        # Output
        self.out = nn.Conv2d(64, n_classes, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.pool1(x1))
        x3 = self.conv3(self.pool2(x2))
        x4 = self.conv4(self.pool3(x3))
        x5 = self.conv5(self.pool4(x4))
        
        # Decoder with attention
        d1 = self.up1(x5)
        x4 = self.att1(g=d1, x=x4)
        d1 = torch.cat([x4, d1], dim=1)
        d1 = self.upconv1(d1)
        
        d2 = self.up2(d1)
        x3 = self.att2(g=d2, x=x3)
        d2 = torch.cat([x3, d2], dim=1)
        d2 = self.upconv2(d2)
        
        d3 = self.up3(d2)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat([x2, d3], dim=1)
        d3 = self.upconv3(d3)
        
        d4 = self.up4(d3)
        x1 = self.att4(g=d4, x=x1)
        d4 = torch.cat([x1, d4], dim=1)
        d4 = self.upconv4(d4)
        
        out = self.out(d4)
        
        return out