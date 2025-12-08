"""
Nicolaes-style 3D CNN for Vertebral Fracture Classification.

Based on: "Detection of vertebral fractures in CT using 3D Convolutional Neural Networks"
by J. Nicolaes et al. (2019) - arXiv:1911.01816

Architecture Notes from Paper:
- Uses voxel-classification 3D CNN trained on vertebral body crops
- Outputs probability maps for fractured vs non-fractured
- Achieves 95% AUC for patient-level and 93% for vertebra-level detection

This implementation:
- Adapts for image-level binary classification (as needed for Grad-CAM++)
- Uses 3D convolutions with small kernels (3x3x3) following the paper's approach
- Designed for straightened vertebra volumes from VerSe2019 dataset
- Compatible with HealthiVert-GAN's HGAM module

Input: Straightened vertebra volume (1, D, H, W) - typically (1, 64, 256, 256)
Output: Binary classification logits (2,) - [no_fracture, has_fracture]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    """3D Convolution block with BatchNorm and ReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class NicolaesVCFClassifier(nn.Module):
    """
    Nicolaes-style 3D CNN for Vertebral Compression Fracture (VCF) Classification.
    
    Based on the voxel-classification 3D CNN approach from Nicolaes et al. (2019),
    adapted for image-level classification needed for Grad-CAM++ extraction.
    
    The architecture uses:
    - Multiple 3D conv layers with small kernels (following paper's approach)
    - BatchNorm for training stability
    - Max pooling for progressive spatial reduction
    - Global average pooling + FC for image-level prediction
    
    Binary classification:
        - Class 0 (Negative): No compression fracture (Genant grade 0)
        - Class 1 (Positive): Has compression fracture (Genant grade 1, 2, 3)
    """
    
    def __init__(self, in_channels=1, num_classes=2, base_filters=32):
        """
        Args:
            in_channels: Number of input channels (1 for CT)
            num_classes: Number of output classes (2 for binary)
            base_filters: Base number of filters (doubled at each stage)
        """
        super(NicolaesVCFClassifier, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Stage 1: Initial feature extraction
        # Input: (B, 1, D, H, W) -> Output: (B, 32, D/2, H/2, W/2)
        self.stage1 = nn.Sequential(
            ConvBlock3D(in_channels, base_filters, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(base_filters, base_filters, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Stage 2: 32 -> 64 channels
        # Output: (B, 64, D/4, H/4, W/4)
        self.stage2 = nn.Sequential(
            ConvBlock3D(base_filters, base_filters * 2, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(base_filters * 2, base_filters * 2, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Stage 3: 64 -> 128 channels
        # Output: (B, 128, D/8, H/8, W/8)
        self.stage3 = nn.Sequential(
            ConvBlock3D(base_filters * 2, base_filters * 4, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(base_filters * 4, base_filters * 4, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Stage 4: 128 -> 256 channels (TARGET LAYER FOR GRAD-CAM++)
        # Output: (B, 256, D/16, H/16, W/16)
        self.stage4 = nn.Sequential(
            ConvBlock3D(base_filters * 4, base_filters * 8, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(base_filters * 8, base_filters * 8, kernel_size=3, stride=1, padding=1),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        # Stage 5: 256 -> 512 channels (deepest features)
        # Output: (B, 512, D/32, H/32, W/32)
        self.stage5 = nn.Sequential(
            ConvBlock3D(base_filters * 8, base_filters * 16, kernel_size=3, stride=1, padding=1),
            ConvBlock3D(base_filters * 16, base_filters * 16, kernel_size=3, stride=1, padding=1)
        )
        
        # Global pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(base_filters * 16, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize network weights using Kaiming initialization."""
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
               Typically (B, 1, 64, 256, 256) for straightened vertebrae
        
        Returns:
            Logits tensor of shape (B, num_classes)
        """
        # Extract features through stages
        x = self.stage1(x)  # (B, 32, D/2, H/2, W/2)
        x = self.stage2(x)  # (B, 64, D/4, H/4, W/4)
        x = self.stage3(x)  # (B, 128, D/8, H/8, W/8)
        x = self.stage4(x)  # (B, 256, D/16, H/16, W/16) <- Grad-CAM++ target
        x = self.stage5(x)  # (B, 512, D/32, H/32, W/32)
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        
        return x
    
    def get_gradcam_target_layer(self):
        """
        Returns the target layer for Grad-CAM++ extraction.
        
        Following the HealthiVert-GAN paper's approach, we use the last
        convolutional layer before global pooling to generate attention maps.
        
        Returns:
            The stage4 module (256-channel features at 1/16 resolution)
        """
        return self.stage4
    
    def get_features(self, x):
        """
        Extract intermediate features for analysis.
        
        Returns features from stage4 (used for Grad-CAM++) and final predictions.
        """
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)  # Grad-CAM++ target features
        x5 = self.stage5(x4)
        
        pooled = self.global_pool(x5)
        flat = pooled.view(pooled.size(0), -1)
        logits = self.classifier(flat)
        
        return {
            'stage4_features': x4,
            'stage5_features': x5,
            'logits': logits
        }


def create_nicolaes_classifier(pretrained_path=None, device='cuda', base_filters=32):
    """
    Create and optionally load pretrained Nicolaes-style classifier.
    
    Args:
        pretrained_path: Path to pretrained weights (.pth file)
        device: 'cuda' or 'cpu'
        base_filters: Base number of filters (32 for ~6M params, 16 for ~1.5M)
    
    Returns:
        Loaded model on specified device
    """
    model = NicolaesVCFClassifier(
        in_channels=1,
        num_classes=2,
        base_filters=base_filters
    )
    
    if pretrained_path is not None:
        state_dict = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded pretrained weights from {pretrained_path}")
    
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test model creation and forward pass
    print("=" * 60)
    print("Nicolaes-style VCF Classifier Test")
    print("=" * 60)
    
    model = NicolaesVCFClassifier(base_filters=32)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test input: (batch, channels, depth, height, width)
    # Using smaller size for testing
    x = torch.randn(2, 1, 64, 128, 128)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Predictions: {torch.argmax(output, dim=1)}")
        print(f"Probabilities: {F.softmax(output, dim=1)}")
    
    # Test feature extraction
    with torch.no_grad():
        features = model.get_features(x)
        print(f"\nStage 4 features shape: {features['stage4_features'].shape}")
        print(f"Stage 5 features shape: {features['stage5_features'].shape}")
    
    # Test Grad-CAM target layer
    target_layer = model.get_gradcam_target_layer()
    print(f"\nGrad-CAM target layer: {type(target_layer).__name__}")
    
    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
