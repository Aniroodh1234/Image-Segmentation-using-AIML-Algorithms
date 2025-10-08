"""
Edge detection algorithms for segmentation
"""
import cv2
import numpy as np


class EdgeDetection:
    """Edge detection methods"""
    
    @staticmethod
    def sobel(image, ksize=3):
        """Sobel edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Sobel in X and Y directions
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
        
        # Combine
        sobel = np.sqrt(sobelx**2 + sobely**2)
        sobel = np.uint8(sobel / sobel.max() * 255)
        
        return sobel
    
    @staticmethod
    def canny(image, threshold1=50, threshold2=150):
        """Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny
        edges = cv2.Canny(blurred, threshold1, threshold2)
        
        return edges
    
    @staticmethod
    def laplacian(image, ksize=3):
        """Laplacian edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
        laplacian = np.uint8(np.absolute(laplacian))
        
        return laplacian
    
    @staticmethod
    def prewitt(image):
        """Prewitt edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Prewitt kernels
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        
        prewittx = cv2.filter2D(gray, -1, kernelx)
        prewitty = cv2.filter2D(gray, -1, kernely)
        
        prewitt = np.sqrt(prewittx**2 + prewitty**2)
        prewitt = np.uint8(prewitt / prewitt.max() * 255)
        
        return prewitt


def run_edge_detection(image_path, output_dir, methods=['sobel', 'canny', 'laplacian']):
    """Run multiple edge detection methods"""
    from pathlib import Path
    
    # Read image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = EdgeDetection()
    results = {}
    
    for method in methods:
        if method == 'sobel':
            edges = detector.sobel(image)
        elif method == 'canny':
            edges = detector.canny(image)
        elif method == 'laplacian':
            edges = detector.laplacian(image)
        elif method == 'prewitt':
            edges = detector.prewitt(image)
        else:
            print(f"Unknown method: {method}")
            continue
        
        results[method] = edges
        
        # Save result
        output_path = output_dir / f"{method}_edges.png"
        cv2.imwrite(str(output_path), edges)
        print(f"Saved {method} edges to {output_path}")
    
    return results