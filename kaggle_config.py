"""
Kaggle Environment Configuration for HealthiVert-GAN.

This script clones the repository and sets up all paths for Kaggle.

Usage in Kaggle notebook:
    # Cell 1: Clone and setup
    !git clone https://github.com/ahmed-226m/HealthiVert-GAN.git /kaggle/working/HealthiVert-GAN
    %cd /kaggle/working/HealthiVert-GAN
    %run kaggle_config.py
"""

import os
import sys
import subprocess

def clone_repo(repo_url='https://github.com/ahmed-226m/HealthiVert-GAN.git', 
               target_dir='/kaggle/working/HealthiVert-GAN'):
    """
    Clone the HealthiVert-GAN repository.
    Call this first if repo is not uploaded as dataset.
    """
    if os.path.exists(target_dir):
        print(f"Repository already exists at {target_dir}")
        return target_dir
    
    print(f"Cloning repository to {target_dir}...")
    subprocess.run(['git', 'clone', repo_url, target_dir], check=True)
    print("Clone complete!")
    return target_dir


def setup_kaggle_env(code_path=None, verse_path=None):
    """
    Configure all environment paths for Kaggle.
    
    Args:
        code_path: Path to HealthiVert-GAN code. 
                   Default: /kaggle/working/HealthiVert-GAN (cloned)
        verse_path: Path to VerSe19 dataset.
                    Default: /kaggle/input/verse19/raw
    
    Returns:
        Dictionary of all paths for reference
    """
    
    # ============================================================
    # PATHS - MODIFY IF NEEDED
    # ============================================================
    
    # Path to HealthiVert-GAN code
    # Option 1: Cloned to /kaggle/working/HealthiVert-GAN
    # Option 2: Uploaded as dataset to /kaggle/input/healthivert-gan
    if code_path is None:
        if os.path.exists('/kaggle/working/HealthiVert-GAN'):
            code_path = '/kaggle/working/HealthiVert-GAN'
        elif os.path.exists('/kaggle/input/healthivert-gan'):
            code_path = '/kaggle/input/healthivert-gan'
        else:
            code_path = '/kaggle/working/HealthiVert-GAN'
    
    CODE_PATH = code_path
    
    # Path to VerSe19 dataset
    # Common patterns:
    # - /kaggle/input/verse19/raw
    # - /kaggle/input/verse-2019/...
    # - /kaggle/input/verse19-dataset/...
    if verse_path is None:
        # Try to auto-detect
        possible_paths = [
            '/kaggle/input/verse19/raw',
            '/kaggle/input/verse-2019/raw', 
            '/kaggle/input/verse19-dataset/raw',
            '/kaggle/input/verse19',
        ]
        for p in possible_paths:
            if os.path.exists(p):
                verse_path = p
                break
        if verse_path is None:
            verse_path = '/kaggle/input/verse19/raw'  # Default
    
    VERSE_INPUT_PATH = verse_path
    
    # Working directory for all outputs
    WORK_DIR = '/kaggle/working'
    
    # ============================================================
    # DERIVED PATHS
    # ============================================================
    
    STRAIGHTENED_PATH = f'{WORK_DIR}/straightened'
    HEATMAP_PATH = f'{WORK_DIR}/Attention/heatmap'
    CLASSIFIER_CHECKPOINT = f'{WORK_DIR}/checkpoints/fracture_classifier'
    GAN_CHECKPOINT = f'{WORK_DIR}/checkpoints/healthivert_gan'
    OUTPUT_PATH = f'{WORK_DIR}/output_3d/sagittal/fine'
    
    # ============================================================
    # SET ENVIRONMENT VARIABLES
    # ============================================================
    
    # For preprocessing scripts
    os.environ['VERSE_DATA_FOLDER'] = VERSE_INPUT_PATH
    os.environ['VERTEBRA_JSON_PATH'] = f'{CODE_PATH}/vertebra_data.json'
    os.environ['STRAIGHTEN_OUTPUT_FOLDER'] = STRAIGHTENED_PATH
    
    # For training/inference
    os.environ['CAM_FOLDER'] = HEATMAP_PATH
    os.environ['CT_FOLDER'] = f'{STRAIGHTENED_PATH}/CT'
    os.environ['MODEL_PATH'] = f'{GAN_CHECKPOINT}/latest_net_G.pth'
    os.environ['OUTPUT_FOLDER'] = OUTPUT_PATH
    
    # Classifier paths
    os.environ['CLASSIFIER_CHECKPOINT'] = CLASSIFIER_CHECKPOINT
    os.environ['CLASSIFIER_MODEL_PATH'] = f'{CLASSIFIER_CHECKPOINT}/best_model.pth'
    
    # ============================================================
    # PYTHON PATH SETUP
    # ============================================================
    
    paths_to_add = [
        CODE_PATH,
        f'{CODE_PATH}/straighten',
    ]
    
    for p in paths_to_add:
        if p not in sys.path:
            sys.path.insert(0, p)
    
    # ============================================================
    # CREATE DIRECTORIES
    # ============================================================
    
    directories = [
        STRAIGHTENED_PATH,
        f'{STRAIGHTENED_PATH}/CT',
        f'{STRAIGHTENED_PATH}/label',
        HEATMAP_PATH,
        CLASSIFIER_CHECKPOINT,
        GAN_CHECKPOINT,
        OUTPUT_PATH,
        f'{OUTPUT_PATH}/CT_fake',
        f'{OUTPUT_PATH}/label_fake',
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    # ============================================================
    # INSTALL STRAIGHTEN MODULE
    # ============================================================
    
    try:
        import straighten
        print("✓ straighten module already installed")
    except ImportError:
        print("Installing straighten module...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '-e', 
            f'{CODE_PATH}/straighten/', '-q'
        ])
        print("✓ straighten module installed")
    
    # ============================================================
    # PRINT CONFIGURATION
    # ============================================================
    
    print("=" * 60)
    print("HealthiVert-GAN Kaggle Environment Configured!")
    print("=" * 60)
    print(f"\nCode path:        {CODE_PATH}")
    print(f"VerSe19 input:    {VERSE_INPUT_PATH}")
    print(f"Straightened:     {STRAIGHTENED_PATH}")
    print(f"Heatmaps:         {HEATMAP_PATH}")
    print(f"Classifier ckpt:  {CLASSIFIER_CHECKPOINT}")
    print(f"GAN checkpoint:   {GAN_CHECKPOINT}")
    print(f"Output:           {OUTPUT_PATH}")
    print("\n✓ All directories created")
    print("✓ Python path configured")
    print("✓ Environment variables set")
    print("=" * 60)
    
    return {
        'CODE_PATH': CODE_PATH,
        'VERSE_INPUT_PATH': VERSE_INPUT_PATH,
        'STRAIGHTENED_PATH': STRAIGHTENED_PATH,
        'HEATMAP_PATH': HEATMAP_PATH,
        'CLASSIFIER_CHECKPOINT': CLASSIFIER_CHECKPOINT,
        'GAN_CHECKPOINT': GAN_CHECKPOINT,
        'OUTPUT_PATH': OUTPUT_PATH,
    }


def print_workflow():
    """Print the complete Kaggle workflow."""
    print("""
================================================================================
HEALTHIVERT-GAN KAGGLE WORKFLOW
================================================================================

STEP 0: Clone Repository (if not uploaded as dataset)
    !git clone https://github.com/YOUR_USERNAME/HealthiVert-GAN.git /kaggle/working/HealthiVert-GAN

STEP 1: Setup Environment
    %cd /kaggle/working/HealthiVert-GAN
    %run kaggle_config.py

STEP 2: Preprocessing - Generate Centroids (if needed)
    !python straighten/location_json_local.py

STEP 3: Preprocessing - Spine Straightening
    !python straighten/straighten_mask_3d.py

STEP 4: Train Fracture Classifier (~2-4 hours)
    !python Attention/train_classifier.py \\
        --dataroot /kaggle/working/straightened \\
        --epochs 50 \\
        --batch_size 4

STEP 5: Generate Grad-CAM++ Heatmaps
    !python Attention/grad_CAM_3d_sagittal.py \\
        --dataroot /kaggle/working/straightened \\
        --classifier_path /kaggle/working/checkpoints/fracture_classifier/best_model.pth \\
        --output_dir /kaggle/working/Attention/heatmap

STEP 6: Train HealthiVert-GAN
    !python train.py \\
        --dataroot /kaggle/working/straightened \\
        --name healthivert_verse19 \\
        --model pix2pix \\
        --direction BtoA \\
        --batch_size 8 \\
        --n_epochs 1000

STEP 7: Inference
    !python eval_3d_sagittal_twostage.py

STEP 8: Evaluation
    !python evaluation/RHLV_quantification.py
    !python evaluation/SVM_grading.py

================================================================================
""")


def print_quick_start():
    """Print minimal quick start for copy-paste."""
    print("""
# ============================================================
# KAGGLE QUICK START - Copy these cells to your notebook
# ============================================================

# Cell 1: Clone and Setup
!git clone https://github.com/YOUR_USERNAME/HealthiVert-GAN.git /kaggle/working/HealthiVert-GAN
%cd /kaggle/working/HealthiVert-GAN
from kaggle_config import setup_kaggle_env, print_workflow
paths = setup_kaggle_env()
print_workflow()

# Cell 2: Preprocessing (modify paths if your VerSe19 structure differs)
!python straighten/straighten_mask_3d.py

# Cell 3: Train Classifier
!python Attention/train_classifier.py --dataroot /kaggle/working/straightened --epochs 50

# Cell 4: Generate Heatmaps
!python Attention/grad_CAM_3d_sagittal.py \\
    --dataroot /kaggle/working/straightened \\
    --classifier_path /kaggle/working/checkpoints/fracture_classifier/best_model.pth

# Cell 5: Train HealthiVert-GAN
!python train.py --dataroot /kaggle/working/straightened --name healthivert_verse19
""")


# Auto-run setup when executed directly
if __name__ == '__main__':
    setup_kaggle_env()
    print_workflow()

