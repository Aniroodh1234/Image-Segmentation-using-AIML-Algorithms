import numpy as np
import cv2
from sklearn.cluster import KMeans
from pathlib import Path


class KMeansSegmentation:
    """K-Means based image segmentation"""
    
    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans = None
    
    def segment(self, image):
        """
        Segment image using K-Means clustering
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            segmented_mask: Segmentation mask (H, W)
        """
        h, w, c = image.shape
        
        # Reshape image to 2D array of pixels
        pixels = image.reshape(-1, c)
        
        # Apply K-Means
        self.kmeans = KMeans(n_clusters=self.n_clusters, 
                            random_state=self.random_state,
                            n_init=10)
        labels = self.kmeans.fit_predict(pixels)
        
        # Reshape back to image dimensions
        segmented_mask = labels.reshape(h, w)
        
        return segmented_mask
    
    def visualize(self, image, mask):
        """Visualize segmentation result"""
        # Create colored segmentation
        segmented_image = np.zeros_like(image)
        colors = np.random.randint(0, 255, size=(self.n_clusters, 3))
        
        for i in range(self.n_clusters):
            segmented_image[mask == i] = colors[i]
        
        return segmented_image


def run_kmeans_segmentation(image_path, output_path, n_clusters=5):
    """Run K-Means segmentation on an image"""
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply segmentation
    segmenter = KMeansSegmentation(n_clusters=n_clusters)
    mask = segmenter.segment(image)
    
    # Visualize
    result = segmenter.visualize(image, mask)
    
    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)
    
    return mask, result