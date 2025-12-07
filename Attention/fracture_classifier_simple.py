"""
Simplified 3D Vertebra Fracture Classifier for Grad-CAM++ Generation.

This is a lighter version of the classifier designed for small datasets like VerSe2019.
Uses fewer parameters and more regularization to prevent overfitting.

Architecture: Lightweight 3D CNN with ~2M parameters (vs 33M in original)
Input: Straightened vertebra volume (256, 256, 64)
Output: Binary classification (normal/mild=0, moderate/severe=1)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """Simple 3D convolution block with BatchNorm, ReLU, and Dropout."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dropout=0.2):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class SimpleResBlock3D(nn.Module):
    """Simplified residual block with dropout."""
    
    def __init__(self, channels, dropout=0.2):
        super(SimpleResBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = out + identity
        out = self.relu(out)
        return out


class SimpleFractureClassifier(nn.Module):
    """
    Lightweight 3D CNN for Binary Fracture Classification.
    
    Designed for small datasets (~500 samples) with:
    - Fewer parameters (~2M vs 33M)
    - More dropout for regularization
    - Shallower architecture
    
    Binary classification:
        - Class 0: Normal/Mild (Genant grade 0-1)
        - Class 1: Moderate/Severe (Genant grade 2-3)
    """
    
    def __init__(self, in_channels=1, num_classes=2, dropout=0.3):
        super(SimpleFractureClassifier, self).__init__()
        
        self.dropout_rate = dropout
        
        # Stem: Initial downsampling
        self.stem = nn.Sequential(
            ConvBlock3D(in_channels, 32, kernel_size=7, stride=2, padding=3, dropout=dropout),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Stage 1: 32 -> 64 channels
        self.stage1 = nn.Sequential(
            ConvBlock3D(32, 64, stride=1, dropout=dropout),
            SimpleResBlock3D(64, dropout=dropout)
        )
        
        # Stage 2: 64 -> 128 channels (downsample)
        self.stage2 = nn.Sequential(
            ConvBlock3D(64, 128, stride=2, dropout=dropout),
            SimpleResBlock3D(128, dropout=dropout)
        )
        
        # Stage 3: 128 -> 256 channels (downsample) - Target for Grad-CAM++
        self.stage3 = nn.Sequential(
            ConvBlock3D(128, 256, stride=2, dropout=dropout),
            SimpleResBlock3D(256, dropout=dropout)
        )
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # Heavy dropout before FC
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: (B, 1, D, H, W) -> typically (B, 1, 64, 256, 256)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)  # Features for Grad-CAM++
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_gradcam_target_layer(self):
        """Returns the target layer for Grad-CAM++ extraction."""
        return self.stage3


def create_simple_classifier(pretrained_path=None, device='cuda'):
    """
    Create and optionally load pretrained classifier.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model on specified device
    """
    model = SimpleFractureClassifier(in_channels=1, num_classes=2, dropout=0.3)
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    model = SimpleFractureClassifier()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test input: (batch, channels, depth, height, width)
    x = torch.randn(2, 1, 64, 256, 256)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Input shape: {x.shape}")
        print(f"Output shape: {output.shape}")  # Should be (2, 2)
        print(f"Predictions: {torch.argmax(output, dim=1)}")
    
    # Test Grad-CAM target layer
    target_layer = model.get_gradcam_target_layer()
    print(f"Grad-CAM target layer: {target_layer}")
