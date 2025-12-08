"""
Training script for Nicolaes-style VCF Classifier.

Based on: "Detection of vertebral fractures in CT using 3D Convolutional Neural Networks"
by J. Nicolaes et al. (2019) - Reference [27] in HealthiVert-GAN paper.

Usage:
    python train.py --dataroot /path/to/straightened --json_path /path/to/vertebra_data.json
    
Example (Kaggle):
    !python Attention/nicolaes_classifier/train.py \
        --dataroot /kaggle/input/straightened-spine-ct \
        --json_path /kaggle/working/HealthiVert-GAN/vertebra_data.json \
        --checkpoint_dir /kaggle/working/checkpoints/nicolaes_classifier \
        --epochs 30 \
        --batch_size 8 \
        --weighted_sampling
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import nibabel as nib
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Import the model
from model import NicolaesVCFClassifier


class VertebraDataset(Dataset):
    """
    Dataset for loading straightened vertebra volumes for VCF classification.
    
    Uses vertebra_data.json to determine fracture grades:
        - Class 0 (Negative): No fracture (Genant grade 0)
        - Class 1 (Positive): Has fracture (Genant grade 1, 2, 3)
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
                # Check for both .nii.gz and .nii extensions
                ct_path_gz = os.path.join(self.ct_folder, f"{vertebra_id}.nii.gz")
                ct_path_nii = os.path.join(self.ct_folder, f"{vertebra_id}.nii")
                
                if os.path.exists(ct_path_gz):
                    ct_path = ct_path_gz
                elif os.path.exists(ct_path_nii):
                    ct_path = ct_path_nii
                else:
                    continue
                
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
        ct_data = np.clip(ct_data, -1000, 3000)
        ct_data = (ct_data + 1000) / 4000
        
        # Apply augmentation
        if self.augment:
            ct_data = self._augment(ct_data)
        
        # Convert to tensor: (D, H, W) -> (1, D, H, W)
        ct_tensor = torch.from_numpy(ct_data).unsqueeze(0)
        
        return {
            'ct': ct_tensor,
            'label': sample['label'],
            'id': sample['id'],
            'genant': sample['genant']
        }
    
    def _augment(self, volume):
        """Apply data augmentation for training."""
        # Random horizontal flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=2).copy()
        
        # Random vertical flip
        if np.random.random() > 0.5:
            volume = np.flip(volume, axis=1).copy()
        
        # Random intensity shift
        if np.random.random() > 0.5:
            shift = np.random.uniform(-0.1, 0.1)
            volume = np.clip(volume + shift, 0, 1)
        
        # Random contrast adjustment
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.9, 1.1)
            mean = volume.mean()
            volume = np.clip((volume - mean) * factor + mean, 0, 1)
        
        # Random Gaussian noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.02, volume.shape)
            volume = np.clip(volume + noise, 0, 1)
        
        return volume.astype(np.float32)
    
    def get_sample_weights(self):
        """Get weights for WeightedRandomSampler to handle class imbalance."""
        labels = [s['label'] for s in self.samples]
        class_counts = [labels.count(0), labels.count(1)]
        
        # Handle zero counts
        class_weights = [1.0 / max(c, 1) for c in class_counts]
        sample_weights = [class_weights[label] for label in labels]
        
        return torch.DoubleTensor(sample_weights)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for batch in pbar:
        ct = batch['ct'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        outputs = model(ct)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.3f}'})
    
    avg_loss = running_loss / len(dataloader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            ct = batch['ct'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(ct)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_ids.extend(batch['id'])
    
    avg_loss = running_loss / len(dataloader)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    return avg_loss, acc, f1, all_preds, all_labels, all_ids


def main(args):
    """Main training function."""
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check for multiple GPUs
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    if num_gpus > 1:
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Paths
    json_path = args.json_path if args.json_path else os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
    
    # Adjust batch size for multi-GPU
    effective_batch_size = args.batch_size * max(1, num_gpus)
    print(f"Effective batch size: {effective_batch_size} ({args.batch_size} x {max(1, num_gpus)} GPUs)")
    
    # Create dataloaders
    if args.weighted_sampling:
        print("Using WeightedRandomSampler for class imbalance")
        sample_weights = train_dataset.get_sample_weights()
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=effective_batch_size,
            shuffle=True,
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
    
    # Create model
    print("Using NicolaesVCFClassifier (based on Nicolaes et al. 2019)")
    model = NicolaesVCFClassifier(
        in_channels=1,
        num_classes=2,
        base_filters=args.base_filters
    )
    
    # Multi-GPU support
    if num_gpus > 1:
        print(f"Using DataParallel with {num_gpus} GPUs")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Loss function with class weights
    train_labels = [s['label'] for s in train_dataset.samples]
    class_counts = [train_labels.count(0), train_labels.count(1)]
    total = sum(class_counts)
    class_weights = torch.FloatTensor([total / (2 * c) for c in class_counts]).to(device)
    print(f"Class weights: {class_weights}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Create checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    best_f1 = 0.0
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
        
        # Update scheduler
        scheduler.step(test_f1)
        
        # Save best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"Saved best model with F1: {best_f1:.4f}")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_f1': best_f1
        }, os.path.join(checkpoint_dir, 'latest_checkpoint.pth'))
    
    # Final evaluation
    print("\n" + "=" * 50)
    print("Final Evaluation on Test Set")
    print("=" * 50)
    
    # Load best model
    model_to_load = model.module if hasattr(model, 'module') else model
    model_to_load.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth')))
    _, test_acc, test_f1, preds, labels, ids = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Test Accuracy: {test_acc:.4f}")
    print(f"Best Test Macro-F1: {test_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=['No Fracture (G0)', 'Has Fracture (G1-3)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, preds))


def parse_args():
    parser = argparse.ArgumentParser(description='Train Nicolaes-style VCF Classifier')
    
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to straightened dataset')
    parser.add_argument('--json_path', type=str, default=None,
                        help='Path to vertebra_data.json')
    parser.add_argument('--checkpoint_dir', type=str, 
                        default='./checkpoints/nicolaes_classifier',
                        help='Directory to save checkpoints')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--base_filters', type=int, default=32,
                        help='Base number of filters (32 for ~6M params, 16 for ~1.5M)')
    parser.add_argument('--weighted_sampling', action='store_true',
                        help='Use WeightedRandomSampler for class imbalance')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
