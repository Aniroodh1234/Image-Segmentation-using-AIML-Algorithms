"""
SegNet implementation
"""
import torch
import torch.nn as nn


class SegNet(nn.Module):
    """SegNet architecture"""
    
    def __init__(self, n_channels=3, n_classes=5, n_init_features=64):
        super(SegNet, self).__init__()
        
        # Encoder
        self.enc1 = self._make_encoder_block(n_channels, n_init_features)
        self.enc2 = self._make_encoder_block(n_init_features, n_init_features*2)
        self.enc3 = self._make_encoder_block(n_init_features*2, n_init_features*4)
        self.enc4 = self._make_encoder_block(n_init_features*4, n_init_features*8)
        
        self.pool = nn.MaxPool2d(2, 2, return_indices=True)
        
        # Decoder
        self.unpool = nn.MaxUnpool2d(2, 2)
        
        self.dec4 = self._make_decoder_block(n_init_features*8, n_init_features*4)
        self.dec3 = self._make_decoder_block(n_init_features*4, n_init_features*2)
        self.dec2 = self._make_decoder_block(n_init_features*2, n_init_features)
        self.dec1 = nn.Sequential(
            nn.Conv2d(n_init_features, n_init_features, 3, padding=1),
            nn.BatchNorm2d(n_init_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_init_features, n_classes, 3, padding=1)
        )
    
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder with pooling indices
        x1 = self.enc1(x)
        x1_pooled, idx1 = self.pool(x1)
        
        x2 = self.enc2(x1_pooled)
        x2_pooled, idx2 = self.pool(x2)
        
        x3 = self.enc3(x2_pooled)
        x3_pooled, idx3 = self.pool(x3)
        
        x4 = self.enc4(x3_pooled)
        x4_pooled, idx4 = self.pool(x4)
        
        # Decoder with unpooling
        x = self.unpool(x4_pooled, idx4)
        x = self.dec4(x)
        
        x = self.unpool(x, idx3)
        x = self.dec3(x)
        
        x = self.unpool(x, idx2)
        x = self.dec2(x)
        
        x = self.unpool(x, idx1)
        x = self.dec1(x)
        
        return x