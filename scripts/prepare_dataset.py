import sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import shutil

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import *
from data.preprocessing import create_segmentation_mask


def prepare_dataset():
    """Prepare dataset by creating masks and organizing files"""
    print("Preparing dataset...")
    
    # Get all images from raw directory
    image_files = list(RAW_DATA_DIR.glob('*.jpg')) + list(RAW_DATA_DIR.glob('*.png'))
    
    if len(image_files) == 0:
        print("⚠️ No images found in data/raw/")
        print("Creating sample dataset...")
        create_sample_dataset()
        image_files = list(RAW_DATA_DIR.glob('*.jpg'))
    
    print(f"Found {len(image_files)} images")
    
    # Create masks directory
    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each image
    for img_path in tqdm(image_files, desc="Creating masks"):
        # Read image
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        
        # Create mask using K-means
        mask = create_segmentation_mask(image, method='kmeans', 
                                       n_clusters=NUM_CLASSES)
        
        # Save mask
        mask_path = MASKS_DIR / f"{img_path.stem}.png"
        cv2.imwrite(str(mask_path), mask)
        
        # Copy to processed directory
        processed_path = PROCESSED_DATA_DIR / img_path.name
        shutil.copy(img_path, processed_path)
    
    print(f"✓ Created {len(image_files)} masks")
    print(f"✓ Processed images saved to {PROCESSED_DATA_DIR}")
    print(f"✓ Masks saved to {MASKS_DIR}")


def create_sample_dataset(n_samples=50):
    """Create sample dataset for demonstration"""
    print("Creating sample landscape images...")
    
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    for i in tqdm(range(n_samples), desc="Generating samples"):
        # Create synthetic landscape image
        img = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Sky (top part - blue gradient)
        sky_height = np.random.randint(80, 120)
        for y in range(sky_height):
            intensity = int(200 - (y / sky_height) * 50)
            img[y, :] = [intensity, intensity, 255]
        
        # Grass (bottom part - green)
        grass_color = [34, np.random.randint(139, 180), 34]
        img[sky_height:, :] = grass_color
        
        # Add some trees (dark green circles)
        n_trees = np.random.randint(3, 8)
        for _ in range(n_trees):
            center_x = np.random.randint(20, 236)
            center_y = np.random.randint(sky_height, 200)
            radius = np.random.randint(15, 35)
            tree_color = [0, np.random.randint(80, 120), 0]
            cv2.circle(img, (center_x, center_y), radius, tree_color, -1)
        
        # Add structures (rectangles)
        if np.random.random() > 0.5:
            x1 = np.random.randint(50, 150)
            y1 = np.random.randint(sky_height, 180)
            w = np.random.randint(30, 60)
            h = np.random.randint(40, 70)
            structure_color = [np.random.randint(100, 180)] * 3
            cv2.rectangle(img, (x1, y1), (x1+w, y1+h), structure_color, -1)
        
        # Add noise
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Save
        cv2.imwrite(str(RAW_DATA_DIR / f"sample_{i:03d}.jpg"), 
                   cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    print(f"✓ Created {n_samples} sample images in {RAW_DATA_DIR}")


if __name__ == '__main__':
    prepare_dataset()