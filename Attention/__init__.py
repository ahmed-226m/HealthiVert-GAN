"""
HealthiVert-GAN Attention Module.

This module provides Grad-CAM++ attention heatmap generation for the
HealthiVert-Guided Attention Module (HGAM).

Components:
    - VertebraClassifier: 3D SE-ResNet for fracture classification
    - GradCAMPlusPlus3D: Grad-CAM++ implementation for 3D volumes
    - Training pipeline: Scripts for classifier training
    - Heatmap generation: Scripts for batch heatmap creation

Usage:
    1. Train classifier:
        python Attention/train_classifier.py --dataroot ./datasets/straightened
    
    2. Generate heatmaps:
        python Attention/grad_CAM_3d_sagittal.py \\
            --dataroot ./datasets/straightened \\
            --classifier_path ./checkpoints/fracture_classifier/best_model.pth
"""

from .fracture_classifier import VertebraClassifier, create_classifier
from .grad_CAM_3d_sagittal import GradCAMPlusPlus3D, generate_heatmaps

__all__ = [
    'VertebraClassifier',
    'create_classifier', 
    'GradCAMPlusPlus3D',
    'generate_heatmaps'
]
