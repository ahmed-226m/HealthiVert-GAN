"""
Training Script for Vertebra Fracture Classifier.

This script trains either the full 3D SE-ResNet classifier or a simpler 
lightweight model for binary fracture detection on small datasets.

Usage:
    python Attention/train_classifier.py --dataroot ./datasets/straightened --epochs 50
    python Attention/train_classifier.py --dataroot ./datasets/straightened --simple --epochs 30

For Kaggle:
    Run cells in notebook format (see examples below)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import nibabel as nib
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from scipy.ndimage import rotate, zoom

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Attention.fracture_classifier import VertebraClassifier, create_classifier
from Attention.fracture_classifier_simple import SimpleFractureClassifier, create_simple_classifier
from Attention.fracture_classifier_dualpath import DualPathClassifier, create_dualpath_classifier


class VertebraDataset(Dataset):
    """
    Dataset for loading straightened vertebra volumes.
    
    Uses vertebra_data.json to determine fracture grades:
        - Class 0: Normal/Mild (Genant grade 0-1)
        - Class 1: Moderate/Severe (Genant grade 2-3)
    """
    
    def __init__(self, dataroot, json_path, phase='train', augment=False):
        """
        Args:
            dataroot: Path to straightened dataset (contains CT/ folder)
            json_path: Path to vertebra_data.json
            phase: 'train' or 'test'
            augment: Whether to apply data augmentation
        """
        self.dataroot = dataroot
        self.ct_folder = os.path.join(dataroot, 'CT')
        self.phase = phase
        self.augment = augment
        
        # Load vertebra data
        with open(json_path, 'r') as f:
            vertebra_data = json.load(f)
        
        # Extract samples for this phase
        self.samples = []
        if phase in vertebra_data:
            for vertebra_id, genant_grade in vertebra_data[phase].items():
                # Check for both .nii.gz and .nii extensions (Kaggle may decompress)
                ct_path_gz = os.path.join(self.ct_folder, f"{vertebra_id}.nii.gz")
                ct_path_nii = os.path.join(self.ct_folder, f"{vertebra_id}.nii")
                
                if os.path.exists(ct_path_gz):
                    ct_path = ct_path_gz
                elif os.path.exists(ct_path_nii):
                    ct_path = ct_path_nii
                else:
                    continue  # File not found, skip
                
                # Binary classification per paper:
                # Class 0 (Negative): No compression fracture (Genant grade 0)
                # Class 1 (Positive): Has compression fracture (Genant grade 1, 2, 3)
                label = 0 if int(genant_grade) == 0 else 1
                self.samples.append({
                    'id': vertebra_id,
                    'path': ct_path,
                    'label': label,
                    'genant': int(genant_grade)
                })
        
        print(f"Loaded {len(self.samples)} samples for {phase}")
        
        # Class distribution
        labels = [s['label'] for s in self.samples]
        print(f"  Class 0 (No fracture - Genant 0): {labels.count(0)}")
        print(f"  Class 1 (Has fracture - Genant 1-3): {labels.count(1)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load CT volume
        ct_nii = nib.load(sample['path'])
        ct_data = ct_nii.get_fdata().astype(np.float32)
        
        # Normalize to [0, 1]
        ct_min, ct_max = ct_data.min(), ct_data.max()
        if ct_max - ct_min > 0:
            ct_data = (ct_data - ct_min) / (ct_max - ct_min)
        
        # Data augmentation for training
        if self.augment and self.phase == 'train':
            ct_data = self._augment(ct_data)
        
        # Convert to tensor: (H, W, D) -> (1, D, H, W) for 3D conv
        ct_tensor = torch.from_numpy(ct_data).float()
        ct_tensor = ct_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, 64, 256, 256)
        
        label = torch.tensor(sample['label'], dtype=torch.long)
        
        return {
            'image': ct_tensor,
            'label': label,
            'id': sample['id'],
            'genant': sample['genant']
        }
    
    def _augment(self, ct_data):
        """Enhanced data augmentation for small datasets."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            ct_data = np.flip(ct_data, axis=1).copy()
        
        # Random vertical flip (less common but adds variety)
        if np.random.random() > 0.7:
            ct_data = np.flip(ct_data, axis=0).copy()
        
        # Random intensity shift
        shift = np.random.uniform(-0.15, 0.15)
        ct_data = np.clip(ct_data + shift, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            contrast = np.random.uniform(0.8, 1.2)
            mean = ct_data.mean()
            ct_data = np.clip((ct_data - mean) * contrast + mean, 0, 1)
        
        # Random Gaussian noise
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.02, ct_data.shape)
            ct_data = np.clip(ct_data + noise, 0, 1)
        
        return ct_data.astype(np.float32)
    
    def get_sample_weights(self):
        """Get sample weights for WeightedRandomSampler to handle class imbalance."""
        labels = [s['label'] for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        # Higher weight for minority class
        class_weights = [1.0 / c for c in class_counts]
        sample_weights = [class_weights[label] for label in labels]
        return sample_weights


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch['id'])
    
    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1, all_preds, all_labels, all_ids


def main(args):
    """Main training function."""
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 1:
        print(f"  GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"  GPU 1: {torch.cuda.get_device_name(1)}")
    
    # Paths
    json_path = args.json_path if args.json_path else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'vertebra_data.json'
    )
    
    # Create datasets
    train_dataset = VertebraDataset(
        dataroot=args.dataroot,
        json_path=json_path,
        phase='train',
        augment=True
    )
    
    test_dataset = VertebraDataset(
        dataroot=args.dataroot,
        json_path=json_path,
        phase='test',
        augment=False
    )
    
    # Adjust batch size for multi-GPU (effective batch size = batch_size * num_gpus)
    effective_batch_size = args.batch_size * max(1, num_gpus)
    print(f"Effective batch size: {effective_batch_size} ({args.batch_size} x {max(1, num_gpus)} GPUs)")
    
    # Create weighted sampler for class imbalance
    if args.weighted_sampling:
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        print("Using WeightedRandomSampler for class imbalance")
        train_shuffle = False
    else:
        sampler = None
        train_shuffle = True
    
    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_batch_size,
        shuffle=train_shuffle,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model - choose architecture
    if args.simple:
        print("Using SimpleFractureClassifier (lightweight, ~2M params)")
        model = SimpleFractureClassifier(in_channels=1, num_classes=2, dropout=0.3)
    elif args.dualpath:
        print("Using DualPathClassifier (Reference [27], ~1M params)")
        model = DualPathClassifier(in_channels=1, num_classes=2, base_channels=30)
    else:
        print("Using VertebraClassifier (full SE-ResNet, ~33M params)")
        model = VertebraClassifier(in_channels=1, num_classes=2, use_se=True)
    
    # Use DataParallel for multi-GPU training
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Class weights for imbalanced data
    train_labels = [s['label'] for s in train_dataset.samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * 2  # Normalize
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_f1 = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Evaluate
        test_loss, test_acc, test_f1, preds, labels, ids = evaluate(
            model, test_loader, criterion, device
        )
        print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
        
        # Learning rate scheduler
        scheduler.step(test_f1)
        
        # Get the underlying model (unwrap DataParallel if used)
        model_to_save = model.module if hasattr(model, 'module') else model
        
        # Save best model with verification
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(model_to_save.state_dict(), best_model_path)
            # Verify save was successful
            if os.path.exists(best_model_path) and os.path.getsize(best_model_path) > 0:
                print(f"Saved best model with F1: {best_f1:.4f}")
            else:
                print(f"WARNING: Failed to save best model!")
        
        # Save latest checkpoint with verification
        latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }, latest_ckpt_path)
    
    # Final evaluation
    print("\n" + "="*50)
    print("Final Evaluation on Test Set")
    print("="*50)
    
    # Load best model (unwrap DataParallel if used)
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    _, test_acc, test_f1, preds, labels, ids = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Macro-F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['No Fracture (G0)', 'Has Fracture (G1-3)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))
    
    # Verify and print saved checkpoint locations
    print("\n" + "="*50)
    print("SAVED CHECKPOINTS:")
    print("="*50)
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    
    if os.path.exists(best_model_path):
        size_mb = os.path.getsize(best_model_path) / (1024 * 1024)
        print(f"✓ Best model: {best_model_path} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Best model NOT FOUND at {best_model_path}")
    
    if os.path.exists(latest_ckpt_path):
        size_mb = os.path.getsize(latest_ckpt_path) / (1024 * 1024)
        print(f"✓ Latest checkpoint: {latest_ckpt_path} ({size_mb:.2f} MB)")
    else:
        print(f"✗ Latest checkpoint NOT FOUND at {latest_ckpt_path}")
    
    print("\nTo download the model, run in Kaggle notebook:")
    print(f"  from IPython.display import FileLink")
    print(f"  FileLink('{best_model_path}')")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Vertebra Fracture Classifier')
    
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to straightened dataset')
    parser.add_argument('--json_path', type=str, default=None,
                        help='Path to vertebra_data.json')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='./checkpoints/fracture_classifier',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--simple', action='store_true',
                        help='Use SimpleFractureClassifier (lightweight, ~2M params)')
    parser.add_argument('--dualpath', action='store_true',
                        help='Use DualPathClassifier (Reference [27] from paper, ~1M params)')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Use WeightedRandomSampler to handle class imbalance')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)


# =============================================================================
# Kaggle Notebook Usage Example
# =============================================================================
"""
# Kaggle Notebook Cells:

# Cell 1: Setup
import sys
sys.path.insert(0, '/kaggle/input/healthivert-gan')
from Attention.train_classifier import *

# Cell 2: Create datasets
train_dataset = VertebraDataset(
    dataroot='/kaggle/working/straightened',
    json_path='/kaggle/input/healthivert-gan/vertebra_data.json',
    phase='train',
    augment=True
)

test_dataset = VertebraDataset(
    dataroot='/kaggle/working/straightened',
    json_path='/kaggle/input/healthivert-gan/vertebra_data.json',
    phase='test',
    augment=False
)

# Cell 3: Train
class Args:
    dataroot = '/kaggle/working/straightened'
    json_path = '/kaggle/input/healthivert-gan/vertebra_data.json'
    checkpoint_dir = '/kaggle/working/checkpoints/fracture_classifier'
    epochs = 50
    batch_size = 4
    lr = 0.0001
    weight_decay = 0.0001
    num_workers = 2

main(Args())
"""
