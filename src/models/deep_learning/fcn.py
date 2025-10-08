"""
Fully Convolutional Network (FCN) implementation
Fixed to support multiple backbones dynamically
"""
import torch
import torch.nn as nn
import torchvision.models as models


class FCN(nn.Module):
    """FCN-8s for semantic segmentation with dynamic backbone support"""
    
    def __init__(self, n_channels=3, n_classes=5, backbone='resnet50', pretrained=True):
        super(FCN, self).__init__()
        
        self.backbone_name = backbone
        
        # Load pretrained ResNet
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            # ResNet50 channel counts
            self.pool3_channels = 512
            self.pool4_channels = 1024
            self.pool5_channels = 2048
        elif backbone == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            # ResNet34 channel counts
            self.pool3_channels = 128
            self.pool4_channels = 256
            self.pool5_channels = 512
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            # ResNet18 channel counts
            self.pool3_channels = 128
            self.pool4_channels = 256
            self.pool5_channels = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}. Choose from: resnet18, resnet34, resnet50")
        
        # Encoder (ResNet layers)
        self.layer1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer2 = resnet.layer1
        self.layer3 = resnet.layer2
        self.layer4 = resnet.layer3
        self.layer5 = resnet.layer4
        
        # FCN head - now dynamic based on backbone!
        self.fc6 = nn.Conv2d(self.pool5_channels, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()
        
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()
        
        self.score_fr = nn.Conv2d(4096, n_classes, 1)
        
        # Skip connection scoring layers
        self.score_pool4 = nn.Conv2d(self.pool4_channels, n_classes, 1)
        self.score_pool3 = nn.Conv2d(self.pool3_channels, n_classes, 1)
        
        # Upsampling layers
        self.upscore2 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False
        )
        self.upscore_pool4 = nn.ConvTranspose2d(
            n_classes, n_classes, 4, stride=2, bias=False
        )
        self.upscore8 = nn.ConvTranspose2d(
            n_classes, n_classes, 16, stride=8, bias=False
        )
    
    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.layer1(x)      # 1/4 size
        x = self.layer2(x)      # 1/4 size
        pool3 = self.layer3(x)  # 1/8 size
        pool4 = self.layer4(pool3)  # 1/16 size
        pool5 = self.layer5(pool4)  # 1/32 size
        
        # FCN layers
        h = self.relu6(self.fc6(pool5))
        h = self.drop6(h)
        h = self.relu7(self.fc7(h))
        h = self.drop7(h)
        h = self.score_fr(h)
        
        # Upsample 2x
        h = self.upscore2(h)
        upscore2 = h
        
        # Add pool4 skip connection
        h = self.score_pool4(pool4)
        # Center crop to match size
        h = self._center_crop(h, upscore2.size())
        score_pool4c = h
        
        # Combine
        h = upscore2 + score_pool4c
        
        # Upsample 2x
        h = self.upscore_pool4(h)
        upscore_pool4 = h
        
        # Add pool3 skip connection
        h = self.score_pool3(pool3)
        # Center crop to match size
        h = self._center_crop(h, upscore_pool4.size())
        score_pool3c = h
        
        # Combine
        h = upscore_pool4 + score_pool3c
        
        # Final upsampling 8x
        h = self.upscore8(h)
        
        # Center crop to match input size
        h = self._center_crop(h, input_size)
        
        return h
    
    def _center_crop(self, layer, target_size):
        """
        Center crop layer to target size
        
        Args:
            layer: Input tensor (B, C, H, W)
            target_size: Target size (H, W) or (B, C, H, W)
        
        Returns:
            Cropped tensor
        """
        if isinstance(target_size, torch.Size):
            if len(target_size) == 4:
                target_height, target_width = target_size[2], target_size[3]
            else:
                target_height, target_width = target_size[0], target_size[1]
        else:
            target_height, target_width = target_size
        
        _, _, layer_height, layer_width = layer.size()
        
        # Calculate crop offsets
        xy1 = (layer_width - target_width) // 2
        xy2 = (layer_height - target_height) // 2
        
        # Crop
        return layer[:, :, xy2:xy2+target_height, xy1:xy1+target_width]