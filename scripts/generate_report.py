"""
Generate comparison report for all models
"""
import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from src.utils.config import OUTPUT_DIR
from utils.config import *


def generate_report():
    """Generate comprehensive comparison report"""
    print("Generating comparison report...")
    
    # Collect all results
    results = {}
    
    # Check for trained models
    models = ['unet', 'fcn', 'segnet']
    
    for model in models:
        checkpoint_path = CHECKPOINT_DIR / f"{model}_best.pt"
        if checkpoint_path.exists():
            # Load checkpoint
            import torch
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            results[model] = {
                'val_loss': checkpoint.get('val_loss', 'N/A'),
                'val_dice': checkpoint.get('val_dice', 'N/A'),
            }
    
    # Create comparison visualization
    if results:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        model_names = list(results.keys())
        val_losses = [results[m]['val_loss'] for m in model_names 
                     if results[m]['val_loss'] != 'N/A']
        val_dices = [results[m]['val_dice'] for m in model_names 
                    if results[m]['val_dice'] != 'N/A']
        
        if val_losses:
            # Plot validation loss
            axes[0].bar(range(len(val_losses)), val_losses, color='skyblue', edgecolor='navy', linewidth=2)
            axes[0].set_xticks(range(len(val_losses)))
            axes[0].set_xticklabels([m.upper() for m in model_names], fontsize=12)
            axes[0].set_ylabel('Validation Loss', fontsize=12)
            axes[0].set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
            axes[0].grid(axis='y', alpha=0.3)
        
        if val_dices:
            # Plot Dice score
            axes[1].bar(range(len(val_dices)), val_dices, color='lightcoral', edgecolor='darkred', linewidth=2)
            axes[1].set_xticks(range(len(val_dices)))
            axes[1].set_xticklabels([m.upper() for m in model_names], fontsize=12)
            axes[1].set_ylabel('Dice Coefficient', fontsize=12)
            axes[1].set_title('Model Comparison - Dice Score', fontsize=14, fontweight='bold')
            axes[1].grid(axis='y', alpha=0.3)
            axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save
        comparison_path = OUTPUT_DIR / 'results' / 'model_comparison.png'
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved comparison chart to {comparison_path}")
        plt.close()
        
        # Save JSON report
        json_path = OUTPUT_DIR / 'results' / 'results_summary.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"✓ Saved results summary to {json_path}")
    
    # Generate README for results
    readme_content = f"""# Segmentation Model Comparison Report
This report summarizes the performance of different segmentation models trained on the dataset.
## Models Evaluated
"""