import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from utils.config import *
from utils.metrics import SegmentationMetrics
from data.dataloader import get_dataloaders
from training.train import train_epoch, validate_epoch
from training.visualize import plot_training_history, visualize_predictions

# Import models
from models.deep_learning.unet import UNet
from models.deep_learning.fcn import FCN
from models.deep_learning.segnet import SegNet
from models.hybrid.attention_unet import AttentionUNet  # ADDED


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience=10, min_delta=0.0, verbose=True):
        """
        Args:
            patience: How many epochs to wait after last improvement
            min_delta: Minimum change to qualify as improvement
            verbose: If True, prints messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    
    def __call__(self, val_loss):
        """
        Call this after each epoch with validation loss
        
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


def get_model(model_name, n_channels=3, n_classes=5):
    """Initialize model based on name"""
    if model_name == 'unet':
        return UNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'fcn':
        return FCN(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'segnet':
        return SegNet(n_channels=n_channels, n_classes=n_classes)
    elif model_name == 'attention_unet':
        return AttentionUNet(n_channels=n_channels, n_classes=n_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: unet, fcn, segnet, attention_unet")


def main():
    parser = argparse.ArgumentParser(description='Train segmentation model')
    parser.add_argument('--model', type=str, default='unet',
                       choices=['unet', 'fcn', 'segnet', 'attention_unet'],  # FIXED
                       help='Model to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                       help='Learning rate')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    model = get_model(args.model, n_channels=IMG_CHANNELS, n_classes=NUM_CLASSES)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )
    
    # ADDED: Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['val_loss']
        print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_dice': [],
        'val_dice': []
    }
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"  Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Dice:   {val_dice:.4f}")
        
        # Learning rate scheduling
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"  Learning rate adjusted: {current_lr:.2e} → {new_lr:.2e}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = CHECKPOINT_DIR / f"{args.model}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
                'train_loss': train_loss,
                'train_dice': train_dice,
            }, checkpoint_path)
            print(f"  ✅ Saved best model to {checkpoint_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % SAVE_INTERVAL == 0:
            checkpoint_path = CHECKPOINT_DIR / f"{args.model}_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_dice': val_dice,
            }, checkpoint_path)
            print(f"  💾 Saved checkpoint to {checkpoint_path}")
        
        # ADDED: Check early stopping
        if early_stopping(val_loss):
            print(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            print(f"Best validation loss: {best_val_loss:.4f}")
            break
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    # Plot training history
    print("\nGenerating training history plot...")
    plot_training_history(
        history, 
        save_path=OUTPUT_DIR / 'results' / f'{args.model}_history.png'
    )
    
    # Visualize predictions
    print("Generating prediction visualizations...")
    visualize_predictions(
        model, val_loader, device, num_images=5,
        save_path=PREDICTION_DIR / f'{args.model}_predictions.png'
    )
    
    print(f"\n✅ All done!")
    print(f"📊 Best validation loss: {best_val_loss:.4f}")
    print(f"💾 Best model saved at: {CHECKPOINT_DIR / f'{args.model}_best.pt'}")
    print(f"📈 Results saved in: {OUTPUT_DIR / 'results'}")


if __name__ == '__main__':
    main()