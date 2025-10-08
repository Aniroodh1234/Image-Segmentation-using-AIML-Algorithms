"""
Watershed algorithm for image segmentation
"""
import numpy as np
import cv2
from scipy import ndimage as ndi


class WatershedSegmentation:
    """Watershed algorithm for segmentation"""
    
    def __init__(self, min_distance=20):
        self.min_distance = min_distance
    
    def segment(self, image):
        """
        Segment image using watershed algorithm
        
        Args:
            image: Input image (H, W, C)
        
        Returns:
            markers: Segmentation markers
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply threshold
        ret, thresh = cv2.threshold(gray, 0, 255, 
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 
                                     255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        return markers
    
    def visualize(self, image, markers):
        """Visualize watershed segmentation"""
        # Create colored image
        result = image.copy()
        result[markers == -1] = [255, 0, 0]  # Mark boundaries in red
        
        return result


def run_watershed_segmentation(image_path, output_path):
    """Run watershed segmentation on an image"""
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Apply segmentation
    segmenter = WatershedSegmentation()
    markers = segmenter.segment(image)
    
    # Visualize
    result = segmenter.visualize(image, markers)
    
    # Save result
    result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), result_bgr)
    
    return markers, result