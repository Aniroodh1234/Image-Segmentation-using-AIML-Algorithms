"""
Inference script for testing trained models on new images
"""
from html import parser
import sys
from pathlib import Path
from src.utils.config import NUM_CLASSES
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

sys.path.append(str(Path(__file__).parent.parent / 'src'))
# Add after line 12
from models.hybrid.attention_unet import AttentionUNet

from utils.config import *
from models.deep_learning.unet import UNet
from models.deep_learning.fcn import FCN
from models.deep_learning.segnet import SegNet
from data.preprocessing import get_validation_augmentation


def load_model(model_name, checkpoint_path, device):
    """Load trained model"""
    if model_name == 'unet':
        model = UNet(n_channels=IMG_CHANNELS, n_classes=NUM_CLASSES)
    elif model_name == 'fcn':
        model = FCN(n_channels=IMG_CHANNELS, n_classes=NUM_CLASSES)
    elif model_name == 'segnet':
        model = SegNet(n_channels=IMG_CHANNELS, n_classes=NUM_CLASSES)
    elif model_name == 'attention_unet':
        model = AttentionUNet(n_channels=IMG_CHANNELS, n_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: unet, fcn, segnet, attention_unet")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model


def predict_image(model, image_path, device):
    """Run prediction on a single image"""
    # Load image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize
    image_resized = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    
    # Apply transforms
    transform = get_validation_augmentation()
    transformed = transform(image=image_resized)
    input_tensor = transformed['image'].unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Resize back to original
    pred_mask_original = cv2.resize(pred_mask.astype(np.uint8), 
                                   (original_size[1], original_size[0]),
                                   interpolation=cv2.INTER_NEAREST)
    
    return image, pred_mask_original


def visualize_prediction(image, mask, save_path=None):
    """Visualize prediction result"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Segmentation mask
    axes[1].imshow(mask, cmap='tab10')
    axes[1].set_title('Segmentation Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    # Overlay
    overlay = image.copy()
    colored_mask = plt.cm.tab10(mask / NUM_CLASSES)[:, :, :3] * 255
    overlay = cv2.addWeighted(overlay.astype(np.uint8), 0.6, 
                             colored_mask.astype(np.uint8), 0.4, 0)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # REPLACE line 92
    parser.add_argument('--model', type=str, required=True,
                   choices=['unet', 'fcn', 'segnet', 'attention_unet'],  # ADDED
                   help='Model to use')
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Run inference on images')
    parser.add_argument('--model', type=str, required=True,
                       choices=['unet', 'fcn', 'segnet'],
                       help='Model to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading {args.model} model from {args.checkpoint}...")
    model = load_model(args.model, args.checkpoint, device)
    
    # Run prediction
    print(f"Running prediction on {args.image}...")
    image, pred_mask = predict_image(model, args.image, device)
    
    # Visualize
    output_path = args.output if args.output else None
    visualize_prediction(image, pred_mask, save_path=output_path)
    
    print("✓ Inference completed!")


if __name__ == '__main__':
    main()