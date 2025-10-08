import os
from pathlib import Path
from src.main import EarlyStopping
from utils.config import PATIENCE

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MASKS_DIR = DATA_DIR / "masks"
AUGMENTED_DIR = DATA_DIR / "augmented"

# Output paths
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
PREDICTION_DIR = OUTPUT_DIR / "predictions"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MASKS_DIR, 
                 AUGMENTED_DIR, CHECKPOINT_DIR, PREDICTION_DIR, LOG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Image settings
IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3
NUM_CLASSES = 5  # Background, Grass, Trees, Sky, Structures

# Training settings
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping

# Data augmentation
AUGMENTATION_PROB = 0.5
ROTATION_RANGE = 15
ZOOM_RANGE = 0.1
BRIGHTNESS_RANGE = 0.2

# Model settings
MODEL_CONFIGS = {
    'unet': {
        'encoder': 'resnet34',
        'encoder_weights': 'imagenet',
        'activation': 'softmax',
    },
    'fcn': {
        'backbone': 'resnet50',
        'pretrained': True,
    },
    'segnet': {
        'n_init_features': 64,
    },
    'vit': {
        'patch_size': 16,
        'embed_dim': 768,
        'depth': 12,
    }
}

# Device settings
DEVICE = 'cuda'  # or 'cpu'
NUM_WORKERS = 4

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 5

from utils.config import PATIENCE
early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

# Check early stopping
# Make sure best_val_loss is defined before this block, for example:
best_val_loss = float('inf')  # or assign it to the best validation loss so far

# Define val_loss before using it
val_loss = 0.0  # Replace with the actual validation loss value from your training loop

if early_stopping(val_loss):
    print(f"\nEarly stopping triggered after {epoch+1} epochs")
    print(f"Best validation loss: {best_val_loss:.4f}")