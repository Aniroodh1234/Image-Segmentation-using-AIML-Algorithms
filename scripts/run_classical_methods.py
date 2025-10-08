import sys
from pathlib import Path
import cv2
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from utils.config import *
from models.classical.kmeans import run_kmeans_segmentation
from models.classical.watershed import run_watershed_segmentation
from models.classical.edge_detection import run_edge_detection
from models.classical.adaptive_thresholding import run_adaptive_thresholding


def run_all_classical_methods():
    """Run all classical segmentation methods"""
    print("Running classical segmentation methods...")
    
    # Get first image from raw data
    image_files = list(RAW_DATA_DIR.glob('*.jpg')) + list(RAW_DATA_DIR.glob('*.png'))
    
    if len(image_files) == 0:
        print("No images found!")
        return
    
    test_image = image_files[0]
    print(f"Testing on image: {test_image.name}")
    
    # Create output directory
    classical_output = OUTPUT_DIR / 'classical_results'
    classical_output.mkdir(parents=True, exist_ok=True)
    
    # 1. K-Means
    print("\n1. Running K-Means Clustering...")
    kmeans_output = classical_output / 'kmeans'
    kmeans_output.mkdir(exist_ok=True)
    
    for k in [3, 5, 7]:
        output_path = kmeans_output / f"kmeans_k{k}.png"
        run_kmeans_segmentation(test_image, output_path, n_clusters=k)
        print(f"   ✓ K={k} saved to {output_path}")
    
    # 2. Watershed
    print("\n2. Running Watershed Algorithm...")
    watershed_output = classical_output / 'watershed'
    watershed_output.mkdir(exist_ok=True)
    run_watershed_segmentation(test_image, watershed_output / 'watershed.png')
    print(f"   ✓ Saved to {watershed_output}")
    
    # 3. Edge Detection
    print("\n3. Running Edge Detection...")
    edge_output = classical_output / 'edges'
    run_edge_detection(test_image, edge_output, 
                      methods=['sobel', 'canny', 'laplacian', 'prewitt'])
    
    # 4. Adaptive Thresholding
    print("\n4. Running Adaptive Thresholding...")
    threshold_output = classical_output / 'thresholding'
    run_adaptive_thresholding(test_image, threshold_output)
    
    print(f"\n✓ All classical methods completed!")
    print(f"Results saved to: {classical_output}")


if __name__ == '__main__':
    run_all_classical_methods()