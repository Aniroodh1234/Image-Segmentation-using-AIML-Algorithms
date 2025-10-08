# Image Segmentation Project
## Multi-Algorithm Image Segmentation Framework

A comprehensive image segmentation project implementing classical ML, deep learning, and hybrid approaches.

## 🎯 Project Overview
This project implements 13+ segmentation algorithms across three categories:
- **Classical ML**: Adaptive Thresholding, Edge Detection, Watershed, K-Means
- **Deep Learning**: FCN, U-Net, SegNet, Mask R-CNN, Vision Transformers
- **Hybrid**: CNN+CRF, GANs, Attention U-Net

## 🚀 Quick Start

### Installation
```bash
# Create conda environment
conda env create -f environment.yml
conda activate img_seg_env

# Or use pip
pip install -r requirements.txt

# Install the project
pip install -e .