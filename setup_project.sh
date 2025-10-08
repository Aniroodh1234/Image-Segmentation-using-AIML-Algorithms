#!/bin/bash

echo "========================================="
echo "Image Segmentation Project Setup"
echo "========================================="

# Create directory structure
echo "Creating directory structure..."

mkdir -p image_segmentation_project
cd image_segmentation_project

# Data directories
mkdir -p data/{raw,processed,masks,augmented}

# Source directories
mkdir -p src/{data,models/{classical,deep_learning,hybrid},training,utils}

# Output directories
mkdir -p outputs/{logs,checkpoints,predictions,results}

# Other directories
mkdir -p scripts notebooks reports/figures

# Create .gitkeep files for empty directories
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/masks/.gitkeep
touch data/augmented/.gitkeep
touch outputs/checkpoints/.gitkeep
touch outputs/predictions/.gitkeep
touch outputs/logs/.gitkeep
touch outputs/results/.gitkeep

# Create __init__.py files
touch src/__init__.py
touch src/data/__init__.py
touch src/models/__init__.py
touch src/models/classical/__init__.py
touch src/models/deep_learning/__init__.py
touch src/models/hybrid/__init__.py
touch src/training/__init__.py
touch src/utils/__init__.py

echo "✓ Directory structure created!"
echo ""
echo "Next steps:"
echo "1. Copy all Python files into their respective directories"
echo "2. Copy requirements.txt and environment.yml to root"
echo "3. Run: pip install -r requirements.txt"
echo "4. Run: python scripts/prepare_dataset.py"
echo "5. Run: bash scripts/run_all.sh"
echo ""
echo "Project structure:"
tree -L 2 -I '__pycache__|*.pyc'