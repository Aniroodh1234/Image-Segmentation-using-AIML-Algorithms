import numpy as np
import torch
import torch.nn.functional as F


def dice_coefficient(pred, target, num_classes=5, smooth=1e-6):
    """
    Calculate Dice coefficient for multi-class segmentation
    
    Args:
        pred: Predicted class indices (B, H, W) or (B*H*W)
        target: Ground truth class indices (B, H, W) or (B*H*W)
        num_classes: Number of classes
        smooth: Smoothing factor to avoid division by zero
    
    Returns:
        Mean Dice score across all classes
    """
    dice_scores = []
    
    for c in range(num_classes):
        # Create binary masks for current class
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        # Flatten tensors
        pred_c = pred_c.contiguous().view(-1)
        target_c = target_c.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = (pred_c * target_c).sum()
        
        # Dice coefficient
        dice = (2. * intersection + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_scores.append(dice.item())
    
    # Return mean across all classes
    return np.mean(dice_scores)


def iou_score(pred, target, num_classes=5, smooth=1e-6):
    """
    Calculate IoU (Jaccard Index) for multi-class segmentation
    
    Args:
        pred: Predicted class indices (B, H, W) or (B*H*W)
        target: Ground truth class indices (B, H, W) or (B*H*W)
        num_classes: Number of classes
        smooth: Smoothing factor
    
    Returns:
        Mean IoU score across all classes
    """
    iou_scores = []
    
    for c in range(num_classes):
        # Create binary masks for current class
        pred_c = (pred == c).float()
        target_c = (target == c).float()
        
        # Flatten tensors
        pred_c = pred_c.contiguous().view(-1)
        target_c = target_c.contiguous().view(-1)
        
        # Calculate intersection and union
        intersection = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum() - intersection
        
        # IoU score
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou.item())
    
    # Return mean across all classes
    return np.mean(iou_scores)


def pixel_accuracy(pred, target):
    """Calculate pixel-wise accuracy"""
    correct = (pred == target).sum()
    total = target.numel()
    return (correct / total).item()


def precision_recall_f1(pred, target, num_classes):
    """Calculate precision, recall, and F1 score per class"""
    results = {}
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        tp = (pred_cls & target_cls).sum().item()
        fp = (pred_cls & ~target_cls).sum().item()
        fn = (~pred_cls & target_cls).sum().item()
        
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        
        results[f'class_{cls}'] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': target_cls.sum().item()
        }
    
    return results


class SegmentationMetrics:
    """Comprehensive metrics calculator for multi-class segmentation"""
    
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics"""
        self.dice_scores = []
        self.iou_scores = []
        self.pixel_accs = []
    
    def update(self, pred, target):
        """
        Update metrics with new batch
        
        Args:
            pred: Model output logits (B, C, H, W)
            target: Ground truth labels (B, H, W)
        """
        # Convert logits to class predictions
        pred = torch.argmax(pred, dim=1)
        
        # Calculate metrics
        self.dice_scores.append(
            dice_coefficient(pred, target, self.num_classes)
        )
        self.iou_scores.append(
            iou_score(pred, target, self.num_classes)
        )
        self.pixel_accs.append(
            pixel_accuracy(pred, target)
        )
    
    def get_results(self):
        """Get average results across all batches"""
        return {
            'dice': np.mean(self.dice_scores),
            'iou': np.mean(self.iou_scores),
            'pixel_accuracy': np.mean(self.pixel_accs)
        }