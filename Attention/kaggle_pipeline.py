"""
Complete Kaggle Pipeline for HealthiVert-GAN with Grad-CAM++.

This script provides a complete end-to-end pipeline for running HealthiVert-GAN
on Kaggle with the VerSe19 dataset.

Copy each cell to your Kaggle notebook and run sequentially.
"""

# =============================================================================
# CELL 1: Environment Setup
# =============================================================================
"""
# Install dependencies (run once)
!pip install nibabel SimpleITK scipy scikit-learn tqdm

# Clone repository (if not uploaded as dataset)
# !git clone https://github.com/zhibaishouheilab/HealthiVert-GAN.git

# Add to path
import sys
sys.path.insert(0, '/kaggle/input/healthivert-gan')
# Or if cloned:
# sys.path.insert(0, '/kaggle/working/HealthiVert-GAN')

# Install straighten module
!pip install -e /kaggle/input/healthivert-gan/straighten/
"""

# =============================================================================
# CELL 2: Import Libraries
# =============================================================================
"""
import os
import json
import numpy as np
import torch
import nibabel as nib
from tqdm.notebook import tqdm

# HealthiVert-GAN imports
from Attention.fracture_classifier import VertebraClassifier, create_classifier
from Attention.train_classifier import VertebraDataset, train_epoch, evaluate
from Attention.grad_CAM_3d_sagittal import GradCAMPlusPlus3D, generate_heatmaps

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

# =============================================================================
# CELL 3: Configure Paths (MODIFY FOR YOUR SETUP)
# =============================================================================
"""
# ============ IMPORTANT: Update these paths for your Kaggle setup ============

# Path to VerSe19 raw dataset (uploaded to Kaggle)
VERSE_RAW_PATH = '/kaggle/input/verse19/raw'

# Path to HealthiVert-GAN code (uploaded as dataset or cloned)
CODE_PATH = '/kaggle/input/healthivert-gan'

# Working directory for outputs
WORK_DIR = '/kaggle/working'

# Derived paths (usually don't need to change)
STRAIGHTENED_PATH = f'{WORK_DIR}/straightened'
HEATMAP_PATH = f'{WORK_DIR}/Attention/heatmap'
CHECKPOINT_PATH = f'{WORK_DIR}/checkpoints/fracture_classifier'

# Create directories
import os
os.makedirs(STRAIGHTENED_PATH, exist_ok=True)
os.makedirs(f'{STRAIGHTENED_PATH}/CT', exist_ok=True)
os.makedirs(f'{STRAIGHTENED_PATH}/label', exist_ok=True)
os.makedirs(HEATMAP_PATH, exist_ok=True)
os.makedirs(CHECKPOINT_PATH, exist_ok=True)

print("Directories created successfully!")
"""

# =============================================================================
# CELL 4: Preprocessing - Spine Straightening
# =============================================================================
"""
# Note: This assumes VerSe19 data is organized as:
# VERSE_RAW_PATH/
#   sub-verse004/
#     sub-verse004_ct.nii.gz
#     sub-verse004_seg-vert_msk.nii.gz
#     sub-verse004_seg-subreg_ctd.json

# Modify straighten scripts for Kaggle paths
# Option A: Run the original scripts with modified paths (recommended)
# Option B: Process in Python directly

# ------- Option A: Using original scripts -------
# First, update the paths in the scripts:

# Edit location_json_local.py to use VERSE_RAW_PATH
# Edit straighten_mask_3d.py to use VERSE_RAW_PATH and STRAIGHTENED_PATH

# Then run:
# !python {CODE_PATH}/straighten/location_json_local.py
# !python {CODE_PATH}/straighten/straighten_mask_3d.py

# ------- Option B: Simple processing (for testing) -------
# If the straightening is already done externally, just copy files:
# !cp -r /path/to/preprocessed/* {STRAIGHTENED_PATH}/

print("Preprocessing step - implement based on your setup")
"""

# =============================================================================
# CELL 5: Phase 1 - Train Fracture Classifier
# =============================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
json_path = f'{CODE_PATH}/vertebra_data.json'

# Create datasets
train_dataset = VertebraDataset(
    dataroot=STRAIGHTENED_PATH,
    json_path=json_path,
    phase='train',
    augment=True
)

test_dataset = VertebraDataset(
    dataroot=STRAIGHTENED_PATH,
    json_path=json_path,
    phase='test',
    augment=False
)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2)

# Create model
model = VertebraClassifier(in_channels=1, num_classes=2, use_se=True)
model = model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Class weights
train_labels = [s['label'] for s in train_dataset.samples]
class_weights = torch.tensor([1.0/train_labels.count(0), 1.0/train_labels.count(1)]).to(device)
class_weights = class_weights / class_weights.sum() * 2

# Training setup
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

# Training loop
num_epochs = 50
best_f1 = 0

for epoch in range(num_epochs):
    # Train
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, criterion, optimizer, device)
    
    # Evaluate
    test_loss, test_acc, test_f1, _, _, _ = evaluate(model, test_loader, criterion, device)
    
    scheduler.step(test_f1)
    
    print(f"Epoch {epoch+1}/{num_epochs} - Train: loss={train_loss:.4f}, acc={train_acc:.4f}, f1={train_f1:.4f} | "
          f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}, f1={test_f1:.4f}")
    
    # Save best model
    if test_f1 > best_f1:
        best_f1 = test_f1
        torch.save(model.state_dict(), f'{CHECKPOINT_PATH}/best_model.pth')
        print(f"  -> Saved best model (F1: {best_f1:.4f})")

print(f"\\nTraining complete! Best F1: {best_f1:.4f}")
"""

# =============================================================================
# CELL 6: Phase 2 - Generate Grad-CAM++ Heatmaps
# =============================================================================
"""
from Attention.grad_CAM_3d_sagittal import GradCAMPlusPlus3D, resize_cam_to_volume, create_classifier

# Load trained classifier
model = create_classifier(
    pretrained_path=f'{CHECKPOINT_PATH}/best_model.pth',
    device=device
)
model.eval()

# Setup Grad-CAM++
target_layer = model.get_gradcam_target_layer()
gradcam = GradCAMPlusPlus3D(model, target_layer)

# Get CT files
ct_folder = f'{STRAIGHTENED_PATH}/CT'
ct_files = [f for f in os.listdir(ct_folder) if f.endswith('.nii.gz')]
print(f"Found {len(ct_files)} CT volumes")

# Generate heatmaps
for file_name in tqdm(ct_files, desc='Generating heatmaps'):
    ct_path = os.path.join(ct_folder, file_name)
    output_path = os.path.join(HEATMAP_PATH, file_name)
    
    if os.path.exists(output_path):
        continue
    
    # Load CT
    ct_nii = nib.load(ct_path)
    ct_data = ct_nii.get_fdata().astype(np.float32)
    
    # Normalize
    ct_norm = (ct_data - ct_data.min()) / (ct_data.max() - ct_data.min() + 1e-8)
    
    # Prepare tensor
    ct_tensor = torch.from_numpy(ct_norm).float()
    ct_tensor = ct_tensor.permute(2, 0, 1).unsqueeze(0).unsqueeze(0).to(device)
    
    # Generate CAM (class 0 = normal)
    cam, _ = gradcam.generate(ct_tensor, target_class=0)
    
    # Resize
    cam_resized = resize_cam_to_volume(cam, ct_data.shape)
    cam_resized = np.clip(cam_resized, 0, 1)
    
    # Save
    nib.save(nib.Nifti1Image(cam_resized.astype(np.float32), ct_nii.affine), output_path)

print(f"\\nGenerated {len(os.listdir(HEATMAP_PATH))} heatmaps")
"""

# =============================================================================
# CELL 7: Phase 3 - Train HealthiVert-GAN (Optional)
# =============================================================================
"""
# Now you can train the main HealthiVert-GAN model!

# Update the CAM folder path in aligned_dataset.py first
# Or set it via command line argument if supported

# Run training
!python {CODE_PATH}/train.py \\
    --dataroot {STRAIGHTENED_PATH} \\
    --name healthivert_verse19 \\
    --model pix2pix \\
    --direction BtoA \\
    --batch_size 8 \\
    --n_epochs 1000 \\
    --gpu_ids 0
"""

# =============================================================================
# CELL 8: Verification
# =============================================================================
"""
# Verify heatmap quality
import matplotlib.pyplot as plt

sample_files = os.listdir(HEATMAP_PATH)[:3]

fig, axes = plt.subplots(len(sample_files), 3, figsize=(15, 5*len(sample_files)))

for i, file_name in enumerate(sample_files):
    cam = nib.load(os.path.join(HEATMAP_PATH, file_name)).get_fdata()
    ct = nib.load(os.path.join(ct_folder, file_name)).get_fdata()
    
    # Middle slice
    mid_z = cam.shape[2] // 2
    
    axes[i, 0].imshow(ct[:, :, mid_z], cmap='gray')
    axes[i, 0].set_title(f'CT - {file_name}')
    
    axes[i, 1].imshow(cam[:, :, mid_z], cmap='jet')
    axes[i, 1].set_title(f'CAM Heatmap')
    
    axes[i, 2].imshow(ct[:, :, mid_z], cmap='gray')
    axes[i, 2].imshow(cam[:, :, mid_z], cmap='jet', alpha=0.4)
    axes[i, 2].set_title('Overlay')

plt.tight_layout()
plt.savefig(f'{WORK_DIR}/heatmap_verification.png')
plt.show()

print("Verification complete!")
"""
