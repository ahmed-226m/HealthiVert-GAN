"""
Kaggle Environment Configuration for HealthiVert-GAN.

Run this file at the start of your Kaggle notebook to set all paths correctly.

Usage:
    # In your Kaggle notebook Cell 1:
    %run /kaggle/input/healthivert-gan/kaggle_config.py
    
    # Or import and call setup:
    from kaggle_config import setup_kaggle_env
    setup_kaggle_env()
"""

import os
import sys

def setup_kaggle_env():
    """
    Configure all environment paths for Kaggle.
    Call this at the start of your notebook.
    """
    
    # ============================================================
    # MODIFY THESE PATHS IF YOUR DATASET NAMES ARE DIFFERENT
    # ============================================================
    
    # Path to HealthiVert-GAN code (uploaded as Kaggle dataset)
    CODE_PATH = '/kaggle/input/healthivert-gan'
    
    # Path to VerSe19 dataset (uploaded as Kaggle dataset)
    # Typical structure: /kaggle/input/verse19/raw/sub-verseXXX/...
    VERSE_INPUT_PATH = '/kaggle/input/verse19/raw'
    
    # Working directory for all outputs
    WORK_DIR = '/kaggle/working'
    
    # ============================================================
    # DERIVED PATHS (usually no need to modify)
    # ============================================================
    
    # Preprocessing output
    STRAIGHTENED_PATH = f'{WORK_DIR}/straightened'
    
    # Grad-CAM++ heatmaps
    HEATMAP_PATH = f'{WORK_DIR}/Attention/heatmap'
    
    # Classifier checkpoints
    CLASSIFIER_CHECKPOINT = f'{WORK_DIR}/checkpoints/fracture_classifier'
    
    # HealthiVert-GAN checkpoints
    GAN_CHECKPOINT = f'{WORK_DIR}/checkpoints/healthivert_gan'
    
    # Output directory for inference
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
    
    # Add code path to Python path
    if CODE_PATH not in sys.path:
        sys.path.insert(0, CODE_PATH)
    
    # Add straighten module to path
    straighten_path = f'{CODE_PATH}/straighten'
    if straighten_path not in sys.path:
        sys.path.insert(0, straighten_path)
    
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
    """Print the complete workflow for reference."""
    print("""
================================================================================
HEALTHIVERT-GAN KAGGLE WORKFLOW
================================================================================

STEP 1: Setup Environment
    %run /kaggle/input/healthivert-gan/kaggle_config.py
    # or: from kaggle_config import setup_kaggle_env; setup_kaggle_env()

STEP 2: Preprocessing (Spine Straightening)
    # Generate vertebral centroids (if not in VerSe19 JSON format)
    !python /kaggle/input/healthivert-gan/straighten/location_json_local.py
    
    # Straighten spine and extract vertebrae
    !python /kaggle/input/healthivert-gan/straighten/straighten_mask_3d.py

STEP 3: Train Fracture Classifier
    !python /kaggle/input/healthivert-gan/Attention/train_classifier.py \\
        --dataroot /kaggle/working/straightened \\
        --epochs 50 \\
        --batch_size 4

STEP 4: Generate Grad-CAM++ Heatmaps
    !python /kaggle/input/healthivert-gan/Attention/grad_CAM_3d_sagittal.py \\
        --dataroot /kaggle/working/straightened \\
        --classifier_path /kaggle/working/checkpoints/fracture_classifier/best_model.pth \\
        --output_dir /kaggle/working/Attention/heatmap

STEP 5: Train HealthiVert-GAN
    !python /kaggle/input/healthivert-gan/train.py \\
        --dataroot /kaggle/working/straightened \\
        --name healthivert_verse19 \\
        --model pix2pix \\
        --direction BtoA \\
        --batch_size 8 \\
        --n_epochs 1000

STEP 6: Inference (Generate Pseudo-Healthy Vertebrae)
    !python /kaggle/input/healthivert-gan/eval_3d_sagittal_twostage.py

STEP 7: Evaluate (RHLV Quantification & SVM Grading)
    !python /kaggle/input/healthivert-gan/evaluation/RHLV_quantification.py
    !python /kaggle/input/healthivert-gan/evaluation/SVM_grading.py

================================================================================
""")


# Auto-run setup when executed directly
if __name__ == '__main__':
    setup_kaggle_env()
    print_workflow()
