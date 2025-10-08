#!/bin/bash
# Complete workflow for processing your park image

echo "🖼️  Image Segmentation Workflow"
echo "================================"

# Step 1: Setup
echo ""
echo "Step 1: Activating environment..."
conda activate img_seg_env

# Step 2: Prepare dataset
echo ""
echo "Step 2: Preparing dataset (creating masks)..."
python scripts/prepare_dataset.py

# Step 3: Test classical methods
echo ""
echo "Step 3: Running classical segmentation methods..."
python scripts/run_classical_methods.py

# Step 4: Train U-Net
echo ""
echo "Step 4: Training U-Net model..."
python main.py --model unet --epochs 30 --batch_size 8

# Step 5: Train Attention U-Net
echo ""
echo "Step 5: Training Attention U-Net model..."
python main.py --model attention_unet --epochs 30 --batch_size 8

# Step 6: Generate report
echo ""
echo "Step 6: Generating comparison report..."
python scripts/generate_report.py

# Step 7: Run inference
echo ""
echo "Step 7: Running inference on test image..."
python scripts/inference.py \
    --model unet \
    --checkpoint outputs/checkpoints/unet_best.pt \
    --image data/raw/park_landscape.jpg \
    --output outputs/predictions/park_segmented.png

echo ""
echo "✅ Workflow completed!"
echo ""
echo "📁 Check these directories for results:"
echo "   - outputs/classical_results/    (Classical methods)"
echo "   - outputs/checkpoints/          (Trained models)"
echo "   - outputs/predictions/          (Segmentation results)"
echo "   - outputs/results/              (Comparison reports)"