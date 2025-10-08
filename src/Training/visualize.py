"""
Visualization utilities
"""
import matplotlib.pyplot as plt
import torch
import numpy as np
from pathlib import Path


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Plot Dice score
    axes[1].plot(history['train_dice'], label='Train Dice', linewidth=2)
    axes[1].plot(history['val_dice'], label='Val Dice', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Dice Coefficient', fontsize=12)
    axes[1].set_title('Training and Validation Dice Score', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions(model, dataloader, device, num_images=5, save_path=None):
    """Visualize model predictions"""
    model.eval()
    
    images_shown = 0
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5*num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            pred_masks = torch.argmax(outputs, dim=1)
            
            for i in range(images.shape[0]):
                if images_shown >= num_images:
                    break
                
                # Original image
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * np.array([0.229, 0.224, 0.225])) + np.array([0.485, 0.456, 0.406])
                img = np.clip(img, 0, 1)
                
                axes[images_shown, 0].imshow(img)
                axes[images_shown, 0].set_title('Original Image', fontsize=12, fontweight='bold')
                axes[images_shown, 0].axis('off')
                
                # Ground truth
                axes[images_shown, 1].imshow(masks[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
                axes[images_shown, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
                axes[images_shown, 1].axis('off')
                
                # Prediction
                axes[images_shown, 2].imshow(pred_masks[i].cpu().numpy(), cmap='tab10', vmin=0, vmax=9)
                axes[images_shown, 2].set_title('Prediction', fontsize=12, fontweight='bold')
                axes[images_shown, 2].axis('off')
                
                images_shown += 1
            
            if images_shown >= num_images:
                break
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved predictions to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_confusion_matrix(pred, target, num_classes, save_path=None):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    pred_flat = pred.cpu().numpy().flatten()
    target_flat = target.cpu().numpy().flatten()
    
    cm = confusion_matrix(target_flat, pred_flat, labels=range(num_classes))
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(num_classes),
                yticklabels=range(num_classes))
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {save_path}")
    else:
        plt.show()
    
    plt.close()