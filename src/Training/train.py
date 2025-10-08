import torch
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.metrics import dice_coefficient, iou_score, pixel_accuracy
from utils.config import NUM_CLASSES


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            pred_masks = torch.argmax(outputs, dim=1)
            # FIXED: Pass num_classes parameter
            dice = dice_coefficient(pred_masks, masks, num_classes=NUM_CLASSES)
        
        running_loss += loss.item()
        running_dice += dice
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{dice:.4f}'
        })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    
    return epoch_loss, epoch_dice


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    running_acc = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            pred_masks = torch.argmax(outputs, dim=1)
            # FIXED: Pass num_classes parameter to all metrics
            dice = dice_coefficient(pred_masks, masks, num_classes=NUM_CLASSES)
            iou = iou_score(pred_masks, masks, num_classes=NUM_CLASSES)
            acc = pixel_accuracy(pred_masks, masks)
            
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            running_acc += acc
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice': f'{dice:.4f}',
                'iou': f'{iou:.4f}'
            })
    
    epoch_loss = running_loss / len(dataloader)
    epoch_dice = running_dice / len(dataloader)
    epoch_iou = running_iou / len(dataloader)
    epoch_acc = running_acc / len(dataloader)
    
    return epoch_loss, epoch_dice


def evaluate_model(model, dataloader, device, num_classes):
    """Comprehensive evaluation"""
    from utils.metrics import SegmentationMetrics, precision_recall_f1
    
    model.eval()
    metrics = SegmentationMetrics(num_classes)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc='Evaluating'):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            metrics.update(outputs, masks)
            
            pred_masks = torch.argmax(outputs, dim=1)
            all_preds.append(pred_masks.cpu())
            all_targets.append(masks.cpu())
    
    # Get overall metrics
    results = metrics.get_results()
    
    # Calculate per-class metrics
    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    class_metrics = precision_recall_f1(all_preds, all_targets, num_classes)
    
    results['per_class'] = class_metrics
    
    return results