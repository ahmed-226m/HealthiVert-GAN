"""
DualPath 3D CNN Classifier for Vertebra Fracture Detection.

Based on Reference [27] from HealthiVert-GAN paper (DeepMedic-style architecture).
Adapted for image-level binary classification to enable Grad-CAM++ generation.

Architecture:
- Normal pathway: High resolution features (receptive field ~17³)
- Subsampled pathway: Larger context (receptive field ~51³)
- Fusion + Global pooling for image-level classification

Input: Straightened vertebra volume (H, W, D) - typically (128, 128, 64) or (256, 256, 64)
Output: Binary classification (0=No fracture, 1=Has fracture)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NormalPathway(nn.Module):
    """
    Normal resolution pathway.
    Processes input at full resolution for fine-grained features.
    Receptive field: ~17³ voxels
    """
    
    def __init__(self, in_channels=1, base_channels=30):
        super(NormalPathway, self).__init__()
        
        # 8 convolutional layers (as per DeepMedic reference)
        self.layers = nn.Sequential(
            ConvBlock3D(in_channels, base_channels, 3, 1),      # Conv1
            ConvBlock3D(base_channels, base_channels, 3, 1),    # Conv2
            ConvBlock3D(base_channels, base_channels, 3, 1),    # Conv3
            ConvBlock3D(base_channels, base_channels, 3, 1),    # Conv4
            ConvBlock3D(base_channels, base_channels * 2, 3, 1), # Conv5 - increase channels
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1), # Conv6
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1), # Conv7
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1), # Conv8
        )
    
    def forward(self, x):
        return self.layers(x)


class SubsampledPathway(nn.Module):
    """
    Subsampled (low resolution) pathway.
    Processes downsampled input for larger contextual information.
    Receptive field: ~51³ voxels
    """
    
    def __init__(self, in_channels=1, base_channels=30, downsample_factor=3):
        super(SubsampledPathway, self).__init__()
        
        self.downsample_factor = downsample_factor
        
        # Downsampling layer
        self.downsample = nn.MaxPool3d(kernel_size=downsample_factor, stride=downsample_factor)
        
        # 8 convolutional layers (same structure as normal pathway)
        self.layers = nn.Sequential(
            ConvBlock3D(in_channels, base_channels, 3, 1),
            ConvBlock3D(base_channels, base_channels, 3, 1),
            ConvBlock3D(base_channels, base_channels, 3, 1),
            ConvBlock3D(base_channels, base_channels, 3, 1),
            ConvBlock3D(base_channels, base_channels * 2, 3, 1),
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1),
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1),
            ConvBlock3D(base_channels * 2, base_channels * 2, 3, 1),
        )
    
    def forward(self, x):
        # Downsample first
        x = self.downsample(x)
        return self.layers(x)


class DualPathClassifier(nn.Module):
    """
    Dual-Pathway 3D CNN for Image-Level Fracture Classification.
    
    Based on DeepMedic architecture (Reference [27]), adapted for:
    - Image-level binary classification (not voxel-wise segmentation)
    - Grad-CAM++ extraction for HealthiVert-GAN HGAM module
    
    Architecture:
        1. Normal pathway: High-resolution feature extraction
        2. Subsampled pathway: Multi-scale context
        3. Fusion: Concatenate and merge pathways
        4. Global pooling + Classifier: Image-level prediction
    
    Binary classification:
        - Class 0: No compression fracture (Genant grade 0)
        - Class 1: Has compression fracture (Genant grade 1-3)
    """
    
    def __init__(self, in_channels=1, num_classes=2, base_channels=30):
        super(DualPathClassifier, self).__init__()
        
        self.base_channels = base_channels
        
        # Dual pathways
        self.normal_pathway = NormalPathway(in_channels, base_channels)
        self.subsampled_pathway = SubsampledPathway(in_channels, base_channels)
        
        # Fusion layer (concatenate both pathways: 60 + 60 = 120 channels)
        fusion_channels = base_channels * 2 * 2  # Both pathways output base_channels*2
        self.fusion = nn.Sequential(
            nn.Conv3d(fusion_channels, base_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_channels * 2),
            nn.ReLU(inplace=True),
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(base_channels * 2, base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(base_channels, num_classes)
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
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, D, H, W)
               Typically (B, 1, 64, 256, 256) for straightened vertebra volumes
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Normal pathway (full resolution)
        normal_features = self.normal_pathway(x)
        
        # Subsampled pathway (downsampled then processed)
        sub_features = self.subsampled_pathway(x)
        
        # Upsample subsampled features to match normal pathway size
        sub_features = F.interpolate(
            sub_features, 
            size=normal_features.shape[2:], 
            mode='trilinear', 
            align_corners=False
        )
        
        # Concatenate pathways
        fused = torch.cat([normal_features, sub_features], dim=1)
        
        # Fusion layers - this is the target for Grad-CAM++
        fused = self.fusion(fused)
        
        # Global pooling
        pooled = self.global_pool(fused)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_gradcam_target_layer(self):
        """Returns the target layer for Grad-CAM++ extraction."""
        return self.fusion


def create_dualpath_classifier(pretrained_path=None, device='cuda'):
    """
    Create and optionally load pretrained DualPath classifier.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
        device: 'cuda' or 'cpu'
    
    Returns:
        Loaded model on specified device
    """
    model = DualPathClassifier(in_channels=1, num_classes=2, base_channels=30)
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    model = DualPathClassifier()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test input: (batch, channels, depth, height, width)
    # Straightened vertebra volumes are typically (64, 256, 256) or similar
    x = torch.randn(2, 1, 64, 128, 128)  # Smaller for testing
    
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
    
    # Test with full-size input
    print("\nTesting with full-size input...")
    x_full = torch.randn(1, 1, 64, 256, 256)
    with torch.no_grad():
        output_full = model(x_full)
        print(f"Full input shape: {x_full.shape}")
        print(f"Full output shape: {output_full.shape}")
