"""
Adaptive thresholding for segmentation
"""
import cv2
import numpy as np


class AdaptiveThresholding:
    """Adaptive thresholding methods"""
    
    @staticmethod
    def mean_threshold(image, block_size=11, C=2):
        """
        Adaptive threshold using mean of neighborhood
        
        Args:
            image: Input image
            block_size: Size of pixel neighborhood (odd number)
            C: Constant subtracted from mean
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        
        return thresh
    
    @staticmethod
    def gaussian_threshold(image, block_size=11, C=2):
        """
        Adaptive threshold using Gaussian-weighted sum
        
        Args:
            image: Input image
            block_size: Size of pixel neighborhood (odd number)
            C: Constant subtracted from weighted mean
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, C
        )
        
        return thresh
    
    @staticmethod
    def otsu_threshold(image):
        """Otsu's thresholding"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's thresholding
        ret, thresh = cv2.threshold(blur, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
    
    @staticmethod
    def multi_level_threshold(image, n_classes=5):
        """Multi-level thresholding for segmentation"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate thresholds
        thresholds = np.linspace(0, 255, n_classes + 1)[1:-1]
        
        # Create segmentation mask
        mask = np.zeros_like(gray, dtype=np.uint8)
        
        for i, thresh in enumerate(thresholds):
            mask[gray >= thresh] = i + 1
        
        return mask


def run_adaptive_thresholding(image_path, output_dir):
    """Run various adaptive thresholding methods"""
    from pathlib import Path
    
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    thresholder = AdaptiveThresholding()
    
    # Apply different methods
    mean_thresh = thresholder.mean_threshold(image)
    gaussian_thresh = thresholder.gaussian_threshold(image)
    otsu_thresh = thresholder.otsu_threshold(image)
    multi_thresh = thresholder.multi_level_threshold(image)
    
    # Save results
    cv2.imwrite(str(output_dir / "mean_threshold.png"), mean_thresh)
    cv2.imwrite(str(output_dir / "gaussian_threshold.png"), gaussian_thresh)
    cv2.imwrite(str(output_dir / "otsu_threshold.png"), otsu_thresh)
    cv2.imwrite(str(output_dir / "multi_threshold.png"), multi_thresh * 50)
    
    print(f"Saved thresholding results to {output_dir}")
    
    return {
        'mean': mean_thresh,
        'gaussian': gaussian_thresh,
        'otsu': otsu_thresh,
        'multi': multi_thresh
    }