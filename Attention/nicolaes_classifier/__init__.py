"""
Nicolaes-style 3D CNN for Vertebral Fracture Detection.

Based on: "Detection of vertebral fractures in CT using 3D Convolutional Neural Networks"
by J. Nicolaes et al. (2019) - Reference [27] in HealthiVert-GAN paper.

Original paper: https://arxiv.org/abs/1911.01816

The original method uses:
- Voxel-wise classification 3D CNN
- Cropped vertebral body as input
- Binary classification: fractured vs non-fractured

This implementation adapts the concept for:
- Image-level classification (for Grad-CAM++ in HealthiVert-GAN)
- Straightened vertebra volumes from VerSe2019
- Binary classification: Genant 0 (no fracture) vs Genant 1-3 (has fracture)
"""

from .model import NicolaesVCFClassifier, create_nicolaes_classifier

__all__ = ['NicolaesVCFClassifier', 'create_nicolaes_classifier']
