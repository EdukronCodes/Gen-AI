"""
Surgical Instrument Recognition Data Preprocessing
Handles image preprocessing, augmentation, and feature extraction for surgical instruments
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Image processing imports
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

# Machine Learning imports
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SurgicalImagePreprocessor:
    """Preprocessor for surgical instrument images"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.image_size = config.get('image_size', (224, 224))
        self.augmentation_config = config.get('augmentation', {})
        self.feature_extraction_config = config.get('feature_extraction', {})
        
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from path"""
        try:
            if not os.path.exists(image_path):
                logger.error(f"Image path does not exist: {image_path}")
                return None
            
            # Load image using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None
    
    def resize_image(self, image: np.ndarray, target_size: Tuple[int, int] = None) -> np.ndarray:
        """Resize image to target size"""
        try:
            if target_size is None:
                target_size = self.image_size
            
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
            return resized_image
            
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image pixel values"""
        try:
            # Normalize to [0, 1] range
            normalized_image = image.astype(np.float32) / 255.0
            return normalized_image
            
        except Exception as e:
            logger.error(f"Error normalizing image: {e}")
            return image
    
    def apply_basic_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply basic preprocessing steps"""
        try:
            # Resize image
            image = self.resize_image(image)
            
            # Normalize image
            image = self.normalize_image(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Error in basic preprocessing: {e}")
            return image
    
    def apply_advanced_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply advanced preprocessing for surgical instruments"""
        try:
            # Convert to grayscale for some operations
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply histogram equalization for better contrast
            equalized = cv2.equalizeHist(blurred)
            
            # Convert back to RGB
            processed = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(processed, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            processed = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error in advanced preprocessing: {e}")
            return image
    
    def create_augmentation_pipeline(self) -> A.Compose:
        """Create image augmentation pipeline"""
        try:
            augmentations = []
            
            # Geometric transformations
            if self.augmentation_config.get('rotation', True):
                augmentations.append(A.Rotate(limit=15, p=0.5))
            
            if self.augmentation_config.get('horizontal_flip', True):
                augmentations.append(A.HorizontalFlip(p=0.5))
            
            if self.augmentation_config.get('vertical_flip', False):
                augmentations.append(A.VerticalFlip(p=0.3))
            
            if self.augmentation_config.get('shift_scale_rotate', True):
                augmentations.append(A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.5
                ))
            
            # Color transformations
            if self.augmentation_config.get('brightness_contrast', True):
                augmentations.append(A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ))
            
            if self.augmentation_config.get('hue_saturation', True):
                augmentations.append(A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ))
            
            if self.augmentation_config.get('rgb_shift', True):
                augmentations.append(A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5))
            
            # Noise and blur
            if self.augmentation_config.get('gaussian_noise', True):
                augmentations.append(A.GaussNoise(var_limit=(10.0, 50.0), p=0.3))
            
            if self.augmentation_config.get('gaussian_blur', True):
                augmentations.append(A.GaussianBlur(blur_limit=(3, 7), p=0.3))
            
            # Elastic transformations
            if self.augmentation_config.get('elastic_transform', True):
                augmentations.append(A.ElasticTransform(
                    alpha=1, sigma=50, alpha_affine=50, p=0.3
                ))
            
            # Grid distortion
            if self.augmentation_config.get('grid_distortion', True):
                augmentations.append(A.GridDistortion(
                    num_steps=5, distort_limit=0.3, p=0.3
                ))
            
            # Optical distortion
            if self.augmentation_config.get('optical_distortion', True):
                augmentations.append(A.OpticalDistortion(
                    distort_limit=0.2, shift_limit=0.15, p=0.3
                ))
            
            # Normalization and tensor conversion
            augmentations.extend([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
            
            return A.Compose(augmentations)
            
        except Exception as e:
            logger.error(f"Error creating augmentation pipeline: {e}")
            return A.Compose([
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def extract_image_features(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract handcrafted features from surgical instrument images"""
        try:
            features = {}
            
            # Convert to grayscale for feature extraction
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Basic statistical features
            features['mean_intensity'] = np.mean(gray)
            features['std_intensity'] = np.std(gray)
            features['min_intensity'] = np.min(gray)
            features['max_intensity'] = np.max(gray)
            features['median_intensity'] = np.median(gray)
            
            # Histogram features
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            features['histogram_entropy'] = -np.sum(hist * np.log2(hist + 1e-10))
            features['histogram_skewness'] = self.calculate_skewness(hist)
            features['histogram_kurtosis'] = self.calculate_kurtosis(hist)
            
            # Texture features using GLCM (Gray Level Co-occurrence Matrix)
            glcm_features = self.extract_glcm_features(gray)
            features.update(glcm_features)
            
            # Edge features
            edge_features = self.extract_edge_features(gray)
            features.update(edge_features)
            
            # Shape features
            shape_features = self.extract_shape_features(gray)
            features.update(shape_features)
            
            # Color features (if RGB)
            if len(image.shape) == 3:
                color_features = self.extract_color_features(image)
                features.update(color_features)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting image features: {e}")
            return {}
    
    def calculate_skewness(self, hist: np.ndarray) -> float:
        """Calculate skewness of histogram"""
        try:
            bins = np.arange(256)
            mean = np.sum(bins * hist.flatten()) / np.sum(hist)
            std = np.sqrt(np.sum(((bins - mean) ** 2) * hist.flatten()) / np.sum(hist))
            skewness = np.sum(((bins - mean) ** 3) * hist.flatten()) / (np.sum(hist) * (std ** 3))
            return skewness
        except:
            return 0.0
    
    def calculate_kurtosis(self, hist: np.ndarray) -> float:
        """Calculate kurtosis of histogram"""
        try:
            bins = np.arange(256)
            mean = np.sum(bins * hist.flatten()) / np.sum(hist)
            std = np.sqrt(np.sum(((bins - mean) ** 2) * hist.flatten()) / np.sum(hist))
            kurtosis = np.sum(((bins - mean) ** 4) * hist.flatten()) / (np.sum(hist) * (std ** 4)) - 3
            return kurtosis
        except:
            return 0.0
    
    def extract_glcm_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract GLCM texture features"""
        try:
            features = {}
            
            # Calculate GLCM for different directions
            directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
            
            for i, (dx, dy) in enumerate(directions):
                # Calculate GLCM
                glcm = self.calculate_glcm(gray, dx, dy)
                
                # Extract features from GLCM
                contrast = np.sum(glcm * np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1])))
                homogeneity = np.sum(glcm / (1 + np.square(np.arange(glcm.shape[0])[:, None] - np.arange(glcm.shape[1]))))
                energy = np.sum(glcm ** 2)
                correlation = np.sum(glcm * np.outer(np.arange(glcm.shape[0]), np.arange(glcm.shape[1])))
                
                features[f'glcm_contrast_{i}'] = contrast
                features[f'glcm_homogeneity_{i}'] = homogeneity
                features[f'glcm_energy_{i}'] = energy
                features[f'glcm_correlation_{i}'] = correlation
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting GLCM features: {e}")
            return {}
    
    def calculate_glcm(self, gray: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Calculate Gray Level Co-occurrence Matrix"""
        try:
            # Quantize image to 8 levels
            gray_quantized = (gray // 32).astype(np.uint8)
            
            # Initialize GLCM
            glcm = np.zeros((8, 8), dtype=np.float32)
            
            # Calculate co-occurrence matrix
            for i in range(gray_quantized.shape[0] - abs(dy)):
                for j in range(gray_quantized.shape[1] - abs(dx)):
                    val1 = gray_quantized[i, j]
                    val2 = gray_quantized[i + dy, j + dx]
                    glcm[val1, val2] += 1
            
            # Normalize
            glcm = glcm / np.sum(glcm)
            
            return glcm
            
        except Exception as e:
            logger.error(f"Error calculating GLCM: {e}")
            return np.zeros((8, 8))
    
    def extract_edge_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract edge-based features"""
        try:
            features = {}
            
            # Sobel edges
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            features['sobel_mean'] = np.mean(sobel_magnitude)
            features['sobel_std'] = np.std(sobel_magnitude)
            features['sobel_max'] = np.max(sobel_magnitude)
            
            # Canny edges
            canny = cv2.Canny(gray, 50, 150)
            features['canny_edge_density'] = np.sum(canny > 0) / (canny.shape[0] * canny.shape[1])
            
            # Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['laplacian_variance'] = np.var(laplacian)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting edge features: {e}")
            return {}
    
    def extract_shape_features(self, gray: np.ndarray) -> Dict[str, float]:
        """Extract shape-based features"""
        try:
            features = {}
            
            # Threshold image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Area and perimeter
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                features['contour_area'] = area
                features['contour_perimeter'] = perimeter
                features['circularity'] = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
                
                # Bounding rectangle
                x, y, w, h = cv2.boundingRect(largest_contour)
                features['aspect_ratio'] = w / h if h > 0 else 0
                features['extent'] = area / (w * h) if w * h > 0 else 0
                
                # Convex hull
                hull = cv2.convexHull(largest_contour)
                hull_area = cv2.contourArea(hull)
                features['solidity'] = area / hull_area if hull_area > 0 else 0
            else:
                features['contour_area'] = 0
                features['contour_perimeter'] = 0
                features['circularity'] = 0
                features['aspect_ratio'] = 0
                features['extent'] = 0
                features['solidity'] = 0
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting shape features: {e}")
            return {}
    
    def extract_color_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract color-based features"""
        try:
            features = {}
            
            # Convert to different color spaces
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # RGB features
            for i, color in enumerate(['red', 'green', 'blue']):
                features[f'{color}_mean'] = np.mean(image[:, :, i])
                features[f'{color}_std'] = np.std(image[:, :, i])
                features[f'{color}_skewness'] = self.calculate_skewness(
                    cv2.calcHist([image[:, :, i]], [0], None, [256], [0, 256])
                )
            
            # HSV features
            for i, channel in enumerate(['hue', 'saturation', 'value']):
                features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
                features[f'{channel}_std'] = np.std(hsv[:, :, i])
            
            # LAB features
            for i, channel in enumerate(['l', 'a', 'b']):
                features[f'{channel}_mean'] = np.mean(lab[:, :, i])
                features[f'{channel}_std'] = np.std(lab[:, :, i])
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting color features: {e}")
            return {}
    
    def preprocess_image_dataset(self, image_paths: List[str], labels: List[str] = None) -> Dict[str, Any]:
        """Preprocess entire image dataset"""
        try:
            processed_data = {
                'images': [],
                'features': [],
                'labels': [],
                'augmented_images': [],
                'augmented_labels': []
            }
            
            # Load and preprocess images
            for i, image_path in enumerate(image_paths):
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Load image
                image = self.load_image(image_path)
                if image is None:
                    continue
                
                # Apply preprocessing
                processed_image = self.apply_basic_preprocessing(image)
                processed_data['images'].append(processed_image)
                
                # Extract features
                features = self.extract_image_features(processed_image)
                processed_data['features'].append(features)
                
                # Store label
                if labels and i < len(labels):
                    processed_data['labels'].append(labels[i])
            
            # Convert features to DataFrame
            if processed_data['features']:
                features_df = pd.DataFrame(processed_data['features'])
                processed_data['features_df'] = features_df
            
            # Encode labels
            if processed_data['labels']:
                processed_data['encoded_labels'] = self.label_encoder.fit_transform(processed_data['labels'])
            
            # Create augmentation pipeline
            augmentation_pipeline = self.create_augmentation_pipeline()
            
            # Apply augmentation
            if self.augmentation_config.get('apply_augmentation', True):
                for i, image in enumerate(processed_data['images']):
                    # Apply augmentation multiple times
                    for _ in range(self.augmentation_config.get('augmentation_factor', 2)):
                        augmented = augmentation_pipeline(image=image)['image']
                        processed_data['augmented_images'].append(augmented)
                        
                        if processed_data['labels'] and i < len(processed_data['labels']):
                            processed_data['augmented_labels'].append(processed_data['labels'][i])
            
            logger.info(f"Preprocessing completed. Processed {len(processed_data['images'])} images")
            if processed_data['augmented_images']:
                logger.info(f"Generated {len(processed_data['augmented_images'])} augmented images")
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error preprocessing image dataset: {e}")
            return {}
    
    def create_torch_transforms(self, is_training: bool = True) -> transforms.Compose:
        """Create PyTorch transforms for training/inference"""
        try:
            if is_training:
                transform_list = [
                    transforms.Resize(self.image_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(degrees=15),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            else:
                transform_list = [
                    transforms.Resize(self.image_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]
            
            return transforms.Compose(transform_list)
            
        except Exception as e:
            logger.error(f"Error creating torch transforms: {e}")
            return transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def save_preprocessor(self, filepath: str):
        """Save preprocessor state"""
        try:
            preprocessor_state = {
                'label_encoder': self.label_encoder,
                'scaler': self.scaler,
                'config': self.config,
                'image_size': self.image_size
            }
            
            joblib.dump(preprocessor_state, filepath)
            logger.info(f"Preprocessor saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving preprocessor: {e}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessor state"""
        try:
            preprocessor_state = joblib.load(filepath)
            
            self.label_encoder = preprocessor_state['label_encoder']
            self.scaler = preprocessor_state['scaler']
            self.config = preprocessor_state['config']
            self.image_size = preprocessor_state['image_size']
            
            logger.info(f"Preprocessor loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading preprocessor: {e}")

# Example usage
if __name__ == "__main__":
    # Example configuration
    config = {
        'image_size': (224, 224),
        'augmentation': {
            'rotation': True,
            'horizontal_flip': True,
            'brightness_contrast': True,
            'hue_saturation': True,
            'gaussian_noise': True,
            'apply_augmentation': True,
            'augmentation_factor': 2
        },
        'feature_extraction': {
            'extract_handcrafted_features': True,
            'extract_color_features': True,
            'extract_texture_features': True
        }
    }
    
    preprocessor = SurgicalImagePreprocessor(config)
    
    # Example image paths (replace with actual paths)
    image_paths = [
        "path/to/surgical/instrument1.jpg",
        "path/to/surgical/instrument2.jpg",
        "path/to/surgical/instrument3.jpg"
    ]
    
    labels = ["scalpel", "forceps", "scissors"]
    
    # Preprocess dataset
    processed = preprocessor.preprocess_image_dataset(image_paths, labels)
    
    print(f"Processed {len(processed['images'])} images")
    print(f"Extracted {len(processed['features'])} feature sets")
    if processed['augmented_images']:
        print(f"Generated {len(processed['augmented_images'])} augmented images")
    
    # Create torch transforms
    train_transforms = preprocessor.create_torch_transforms(is_training=True)
    val_transforms = preprocessor.create_torch_transforms(is_training=False)
    
    print("Transforms created successfully") 