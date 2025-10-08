"""
Data preprocessing and augmentation
"""
from email.mime import image
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

def create_segmentation_mask(image, method='kmeans', n_clusters=5):
    """
    Create pseudo-labels for unsupervised segmentation
    """
    h, w = image.shape[:2]
    
    if method == 'kmeans':
        from sklearn.cluster import KMeans
        
        # Reshape image
        pixels = image.reshape(-1, 3)
        
        # Apply K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(pixels)
        
        # Reshape back to image
        mask = labels.reshape(h, w)
        
    elif method == 'watershed':
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        # Finding sure foreground area

        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 
                                   255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # FIXED: Handle watershed boundaries (-1 values)
        # Assign boundary pixels to nearest region
        mask = markers.copy()
        mask[mask == -1] = 0  # Assign boundaries to background
        
        # Get unique labels (excluding -1 and 0)
        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]
        
        # FIXED: Normalize to 0 to n_clusters-1 range properly
        mask_normalized = np.zeros_like(mask, dtype=np.uint8)
        
        if len(unique_labels) > 0:
            # Map labels to 0, 1, 2, ..., min(n_clusters-1, len(unique_labels)-1)
            for i, label in enumerate(unique_labels[:n_clusters]):
                mask_normalized[mask == label] = i
            
            # If we have more regions than n_clusters, merge extras into last class
            if len(unique_labels) > n_clusters:
                for label in unique_labels[n_clusters:]:
                    mask_normalized[mask == label] = n_clusters - 1
        
        mask = mask_normalized
        # Normalize to 0-n_clusters range
        unique_labels = np.unique(mask)
        mask_normalized = np.zeros_like(mask)
        for i, label in enumerate(unique_labels[:n_clusters]):
            mask_normalized[mask == label] = i
        mask = mask_normalized
    
    return mask.astype(np.uint8)

def get_training_augmentation():
    """Get training augmentation pipeline"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.OneOf([
            A.ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, 
                             alpha_affine=120 * 0.03),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MedianBlur(blur_limit=3, p=0.5),
        ], p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_augmentation():
    """Get validation augmentation pipeline"""
    return A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def preprocess_image(image_path, target_size=(256, 256)):
    """Load and preprocess a single image"""
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image