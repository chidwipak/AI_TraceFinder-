#!/usr/bin/env python3
"""
New Objective 2: Comprehensive Tampered/Original Detection - Feature Extraction
Extracts comprehensive forensic features utilizing ALL available data including binary masks
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import ndimage, stats
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from scipy.signal import find_peaks

# Image processing
from skimage import filters, feature, measure, segmentation
from skimage.filters import gabor, sobel, roberts, scharr
from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from skimage.measure import shannon_entropy
from skimage.morphology import disk, binary_opening, binary_closing

# Wavelet analysis
import pywt

# Additional libraries
import mahotas as mh
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveObjective2FeatureExtractor:
    def __init__(self, preprocessed_path="new_objective2_preprocessed", output_path="new_objective2_features"):
        self.preprocessed_path = preprocessed_path
        self.output_path = output_path
        self.target_size = (224, 224)
        
        # Create output directories
        self.create_output_directories()
        
        # Feature containers
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path,
            f"{self.output_path}/feature_vectors",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/reports",
            f"{self.output_path}/mask_features"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def extract_comprehensive_features(self, image_path, mask_path=None):
        """Extract comprehensive forensic features from image and optional mask"""
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, self.target_size)
            
            # Load mask if available
            mask = None
            if mask_path and os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, self.target_size)
            
            features = {}
            
            # 1. Basic Statistical Features
            features.update(self._extract_basic_statistical_features(image))
            
            # 2. Texture Features
            features.update(self._extract_texture_features(image))
            
            # 3. Edge Features
            features.update(self._extract_edge_features(image))
            
            # 4. Frequency Domain Features
            features.update(self._extract_frequency_features(image))
            
            # 5. Wavelet Features
            features.update(self._extract_wavelet_features(image))
            
            # 6. Error Level Analysis (ELA)
            features.update(self._extract_ela_features(image))
            
            # 7. Noise Analysis
            features.update(self._extract_noise_features(image))
            
            # 8. Tampering Detection Features
            features.update(self._extract_tampering_features(image))
            
            # 9. Mask-Aware Features (if mask available)
            if mask is not None:
                features.update(self._extract_mask_aware_features(image, mask))
            
            # 10. Advanced Forensic Features
            features.update(self._extract_advanced_forensic_features(image))
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features from {image_path}: {e}")
            return None
    
    def _extract_basic_statistical_features(self, image):
        """Extract basic statistical features"""
        features = {}
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic statistics
        features['mean_r'] = np.mean(image[:, :, 0])
        features['mean_g'] = np.mean(image[:, :, 1])
        features['mean_b'] = np.mean(image[:, :, 2])
        features['mean_gray'] = np.mean(gray)
        
        features['std_r'] = np.std(image[:, :, 0])
        features['std_g'] = np.std(image[:, :, 1])
        features['std_b'] = np.std(image[:, :, 2])
        features['std_gray'] = np.std(gray)
        
        features['skew_r'] = skew(image[:, :, 0].flatten())
        features['skew_g'] = skew(image[:, :, 1].flatten())
        features['skew_b'] = skew(image[:, :, 2].flatten())
        features['skew_gray'] = skew(gray.flatten())
        
        features['kurtosis_r'] = kurtosis(image[:, :, 0].flatten())
        features['kurtosis_g'] = kurtosis(image[:, :, 1].flatten())
        features['kurtosis_b'] = kurtosis(image[:, :, 2].flatten())
        features['kurtosis_gray'] = kurtosis(gray.flatten())
        
        # Histogram features
        hist_r, _ = np.histogram(image[:, :, 0], bins=256, range=(0, 256))
        hist_g, _ = np.histogram(image[:, :, 1], bins=256, range=(0, 256))
        hist_b, _ = np.histogram(image[:, :, 2], bins=256, range=(0, 256))
        
        features['hist_entropy_r'] = shannon_entropy(hist_r)
        features['hist_entropy_g'] = shannon_entropy(hist_g)
        features['hist_entropy_b'] = shannon_entropy(hist_b)
        
        # Color space features
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features['mean_h'] = np.mean(hsv[:, :, 0])
        features['mean_s'] = np.mean(hsv[:, :, 1])
        features['mean_v'] = np.mean(hsv[:, :, 2])
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features using various methods"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Local Binary Pattern (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        features['lbp_hist_entropy'] = shannon_entropy(lbp.astype(int))
        
        # Gray Level Co-occurrence Matrix (GLCM)
        try:
            distances = [1, 2, 3]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
            
            # GLCM properties
            features['glcm_contrast'] = np.mean(graycoprops(glcm, 'contrast'))
            features['glcm_dissimilarity'] = np.mean(graycoprops(glcm, 'dissimilarity'))
            features['glcm_homogeneity'] = np.mean(graycoprops(glcm, 'homogeneity'))
            features['glcm_energy'] = np.mean(graycoprops(glcm, 'energy'))
            features['glcm_correlation'] = np.mean(graycoprops(glcm, 'correlation'))
            features['glcm_asm'] = np.mean(graycoprops(glcm, 'ASM'))
        except Exception as e:
            logger.warning(f"GLCM feature extraction failed: {e}")
            features.update({f'glcm_{prop}': 0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']})
        
        # Gabor filters
        try:
            gabor_responses = []
            for frequency in [0.1, 0.3, 0.5]:
                for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                    filt_real, _ = gabor(gray, frequency=frequency, theta=theta)
                    gabor_responses.append(np.mean(filt_real))
            
            features['gabor_mean'] = np.mean(gabor_responses)
            features['gabor_std'] = np.std(gabor_responses)
        except Exception as e:
            logger.warning(f"Gabor feature extraction failed: {e}")
            features['gabor_mean'] = 0
            features['gabor_std'] = 0
        
        return features
    
    def _extract_edge_features(self, image):
        """Extract edge-related features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Sobel edges
        sobel_x = sobel(gray)
        sobel_y = sobel(gray)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        features['sobel_mean'] = np.mean(sobel_magnitude)
        features['sobel_std'] = np.std(sobel_magnitude)
        features['sobel_max'] = np.max(sobel_magnitude)
        
        # Roberts edges
        roberts_magnitude = np.sqrt(roberts(gray)**2)
        features['roberts_mean'] = np.mean(roberts_magnitude)
        features['roberts_std'] = np.std(roberts_magnitude)
        
        # Scharr edges
        scharr_magnitude = np.sqrt(scharr(gray)**2)
        features['scharr_mean'] = np.mean(scharr_magnitude)
        features['scharr_std'] = np.std(scharr_magnitude)
        
        # Canny edges
        canny = cv2.Canny(gray, 50, 150)
        features['canny_edge_density'] = np.sum(canny > 0) / (canny.shape[0] * canny.shape[1])
        
        # HOG features
        try:
            hog_features = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')
            features['hog_mean'] = np.mean(hog_features)
            features['hog_std'] = np.std(hog_features)
            features['hog_max'] = np.max(hog_features)
        except Exception as e:
            logger.warning(f"HOG feature extraction failed: {e}")
            features['hog_mean'] = 0
            features['hog_std'] = 0
            features['hog_max'] = 0
        
        return features
    
    def _extract_frequency_features(self, image):
        """Extract frequency domain features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # FFT features
        f_transform = fft2(gray)
        f_shift = fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        features['fft_mean'] = np.mean(magnitude_spectrum)
        features['fft_std'] = np.std(magnitude_spectrum)
        features['fft_max'] = np.max(magnitude_spectrum)
        
        # High frequency energy
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        high_freq_region = magnitude_spectrum[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        features['high_freq_energy'] = np.sum(high_freq_region)
        
        # DCT features
        dct_transform = cv2.dct(gray.astype(np.float32))
        features['dct_mean'] = np.mean(dct_transform)
        features['dct_std'] = np.std(dct_transform)
        
        return features
    
    def _extract_wavelet_features(self, image):
        """Extract wavelet-based features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Daubechies wavelets
        try:
            coeffs = pywt.dwt2(gray, 'db4')
            cA, (cH, cV, cD) = coeffs
            
            # Wavelet energy
            features['wavelet_energy_approx'] = np.sum(cA**2)
            features['wavelet_energy_horizontal'] = np.sum(cH**2)
            features['wavelet_energy_vertical'] = np.sum(cV**2)
            features['wavelet_energy_diagonal'] = np.sum(cD**2)
            
            # Wavelet entropy
            features['wavelet_entropy_approx'] = shannon_entropy(cA)
            features['wavelet_entropy_horizontal'] = shannon_entropy(cH)
            features['wavelet_entropy_vertical'] = shannon_entropy(cV)
            features['wavelet_entropy_diagonal'] = shannon_entropy(cD)
            
        except Exception as e:
            logger.warning(f"Wavelet feature extraction failed: {e}")
            features.update({f'wavelet_energy_{band}': 0 for band in ['approx', 'horizontal', 'vertical', 'diagonal']})
            features.update({f'wavelet_entropy_{band}': 0 for band in ['approx', 'horizontal', 'vertical', 'diagonal']})
        
        return features
    
    def _extract_ela_features(self, image):
        """Extract Error Level Analysis features"""
        features = {}
        
        # Convert to JPEG and back to detect compression artifacts
        try:
            # Save as JPEG and reload
            temp_path = '/tmp/temp_image.jpg'
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            jpeg_image = cv2.imread(temp_path)
            jpeg_image = cv2.cvtColor(jpeg_image, cv2.COLOR_BGR2RGB)
            
            # Calculate difference (ELA)
            ela = np.abs(image.astype(np.float32) - jpeg_image.astype(np.float32))
            
            features['ela_mean'] = np.mean(ela)
            features['ela_std'] = np.std(ela)
            features['ela_max'] = np.max(ela)
            
            # ELA in different regions
            h, w = ela.shape[:2]
            center_ela = ela[h//4:3*h//4, w//4:3*w//4]
            features['ela_center_mean'] = np.mean(center_ela)
            features['ela_center_std'] = np.std(center_ela)
            
            os.remove(temp_path)
            
        except Exception as e:
            logger.warning(f"ELA feature extraction failed: {e}")
            features['ela_mean'] = 0
            features['ela_std'] = 0
            features['ela_max'] = 0
            features['ela_center_mean'] = 0
            features['ela_center_std'] = 0
        
        return features
    
    def _extract_noise_features(self, image):
        """Extract noise-related features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Noise estimation using median filter
        median_filtered = cv2.medianBlur(gray, 5)
        noise = gray.astype(np.float32) - median_filtered.astype(np.float32)
        
        features['noise_mean'] = np.mean(noise)
        features['noise_std'] = np.std(noise)
        features['noise_variance'] = np.var(noise)
        
        # Noise in different frequency bands
        try:
            # Low-pass filter
            kernel = np.ones((5,5), np.float32) / 25
            low_pass = cv2.filter2D(gray, -1, kernel)
            high_freq_noise = gray.astype(np.float32) - low_pass.astype(np.float32)
            
            features['high_freq_noise_std'] = np.std(high_freq_noise)
            features['high_freq_noise_mean'] = np.mean(high_freq_noise)
        except Exception as e:
            logger.warning(f"High frequency noise extraction failed: {e}")
            features['high_freq_noise_std'] = 0
            features['high_freq_noise_mean'] = 0
        
        return features
    
    def _extract_tampering_features(self, image):
        """Extract tampering-specific features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Copy-move detection features
        try:
            # Block-based analysis
            block_size = 16
            h, w = gray.shape
            num_blocks_h = h // block_size
            num_blocks_w = w // block_size
            
            block_features = []
            for i in range(num_blocks_h):
                for j in range(num_blocks_w):
                    block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                    block_mean = np.mean(block)
                    block_std = np.std(block)
                    block_features.append([block_mean, block_std])
            
            block_features = np.array(block_features)
            
            # Similarity between blocks
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(block_features)
            
            # Remove diagonal (self-similarity)
            np.fill_diagonal(similarities, 0)
            
            features['block_similarity_max'] = np.max(similarities)
            features['block_similarity_mean'] = np.mean(similarities)
            features['block_similarity_std'] = np.std(similarities)
            
        except Exception as e:
            logger.warning(f"Copy-move detection failed: {e}")
            features['block_similarity_max'] = 0
            features['block_similarity_mean'] = 0
            features['block_similarity_std'] = 0
        
        # Splicing detection features
        try:
            # Gradient analysis for splicing
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            features['gradient_mean'] = np.mean(gradient_magnitude)
            features['gradient_std'] = np.std(gradient_magnitude)
            features['gradient_max'] = np.max(gradient_magnitude)
            
        except Exception as e:
            logger.warning(f"Splicing detection failed: {e}")
            features['gradient_mean'] = 0
            features['gradient_std'] = 0
            features['gradient_max'] = 0
        
        return features
    
    def _extract_mask_aware_features(self, image, mask):
        """Extract features using binary mask information"""
        features = {}
        
        # Normalize mask
        mask = mask.astype(np.float32) / 255.0
        
        # Features for masked region
        masked_region = image * mask[:, :, np.newaxis]
        gray_masked = cv2.cvtColor(masked_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Features for non-masked region
        non_masked_region = image * (1 - mask)[:, :, np.newaxis]
        gray_non_masked = cv2.cvtColor(non_masked_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        
        # Statistical features for masked region
        if np.sum(mask) > 0:
            features['masked_mean'] = np.mean(gray_masked[mask > 0])
            features['masked_std'] = np.std(gray_masked[mask > 0])
            features['masked_skew'] = skew(gray_masked[mask > 0])
            features['masked_kurtosis'] = kurtosis(gray_masked[mask > 0])
        else:
            features['masked_mean'] = 0
            features['masked_std'] = 0
            features['masked_skew'] = 0
            features['masked_kurtosis'] = 0
        
        # Statistical features for non-masked region
        if np.sum(1 - mask) > 0:
            features['non_masked_mean'] = np.mean(gray_non_masked[(1 - mask) > 0])
            features['non_masked_std'] = np.std(gray_non_masked[(1 - mask) > 0])
            features['non_masked_skew'] = skew(gray_non_masked[(1 - mask) > 0])
            features['non_masked_kurtosis'] = kurtosis(gray_non_masked[(1 - mask) > 0])
        else:
            features['non_masked_mean'] = 0
            features['non_masked_std'] = 0
            features['non_masked_skew'] = 0
            features['non_masked_kurtosis'] = 0
        
        # Mask properties
        features['mask_coverage'] = np.sum(mask) / (mask.shape[0] * mask.shape[1])
        features['mask_perimeter'] = np.sum(cv2.Canny((mask * 255).astype(np.uint8), 50, 150) > 0)
        
        # Texture difference between masked and non-masked regions
        try:
            if np.sum(mask) > 0 and np.sum(1 - mask) > 0:
                # LBP for masked region
                lbp_masked = local_binary_pattern(gray_masked, 8, 1, method='uniform')
                lbp_masked_hist = np.histogram(lbp_masked[mask > 0], bins=10)[0]
                
                # LBP for non-masked region
                lbp_non_masked = local_binary_pattern(gray_non_masked, 8, 1, method='uniform')
                lbp_non_masked_hist = np.histogram(lbp_non_masked[(1 - mask) > 0], bins=10)[0]
                
                # Chi-square distance between histograms
                features['texture_chi_square'] = np.sum((lbp_masked_hist - lbp_non_masked_hist)**2 / (lbp_masked_hist + lbp_non_masked_hist + 1e-10))
            else:
                features['texture_chi_square'] = 0
        except Exception as e:
            logger.warning(f"Texture difference extraction failed: {e}")
            features['texture_chi_square'] = 0
        
        return features
    
    def _extract_advanced_forensic_features(self, image):
        """Extract advanced forensic features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Zernike moments
        try:
            from skimage.measure import moments
            moments_result = moments(gray)
            features['moment_00'] = moments_result[0, 0]
            features['moment_01'] = moments_result[0, 1]
            features['moment_10'] = moments_result[1, 0]
            features['moment_11'] = moments_result[1, 1]
        except Exception as e:
            logger.warning(f"Zernike moments extraction failed: {e}")
            features.update({f'moment_{i}{j}': 0 for i in range(2) for j in range(2)})
        
        # Haralick texture features
        try:
            from skimage.feature import graycomatrix, graycoprops
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features['haralick_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features['haralick_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features['haralick_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features['haralick_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features['haralick_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
        except Exception as e:
            logger.warning(f"Haralick features extraction failed: {e}")
            features.update({f'haralick_{prop}': 0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']})
        
        # Tamura texture features
        try:
            # Coarseness
            coarseness = self._calculate_coarseness(gray)
            features['tamura_coarseness'] = coarseness
            
            # Contrast
            contrast = self._calculate_contrast(gray)
            features['tamura_contrast'] = contrast
            
            # Directionality
            directionality = self._calculate_directionality(gray)
            features['tamura_directionality'] = directionality
            
        except Exception as e:
            logger.warning(f"Tamura features extraction failed: {e}")
            features.update({f'tamura_{prop}': 0 for prop in ['coarseness', 'contrast', 'directionality']})
        
        return features
    
    def _calculate_coarseness(self, image):
        """Calculate Tamura coarseness"""
        h, w = image.shape
        coarseness = np.zeros((h, w))
        
        for k in range(1, 7):  # Different window sizes
            window_size = 2**k
            for i in range(h - window_size):
                for j in range(w - window_size):
                    window = image[i:i+window_size, j:j+window_size]
                    coarseness[i, j] += np.std(window)
        
        return np.mean(coarseness)
    
    def _calculate_contrast(self, image):
        """Calculate Tamura contrast"""
        return np.std(image) / (np.mean(image) + 1e-10)
    
    def _calculate_directionality(self, image):
        """Calculate Tamura directionality"""
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        gradient_direction = np.arctan2(grad_y, grad_x)
        
        # Histogram of gradient directions
        hist, _ = np.histogram(gradient_direction, bins=16, range=(-np.pi, np.pi))
        
        # Find peaks in histogram
        peaks, _ = find_peaks(hist, height=np.max(hist) * 0.1)
        
        return len(peaks)
    
    def extract_features_from_inventory(self):
        """Extract features from all images in the inventory"""
        logger.info("="*80)
        logger.info("EXTRACTING COMPREHENSIVE FEATURES")
        logger.info("="*80)
        
        # Load both train and test inventories
        train_inventory_path = f"{self.preprocessed_path}/metadata/train_split.csv"
        test_inventory_path = f"{self.preprocessed_path}/metadata/test_split.csv"
        
        train_df = pd.read_csv(train_inventory_path)
        test_df = pd.read_csv(test_inventory_path)
        
        # Combine train and test data
        inventory_df = pd.concat([train_df, test_df], ignore_index=True)
        
        all_features = []
        failed_extractions = 0
        
        for idx, row in inventory_df.iterrows():
            if idx % 50 == 0:
                logger.info(f"Processing image {idx+1}/{len(inventory_df)}")
            
            image_path = row['image_path']
            mask_path = row['mask_path'] if row['has_mask'] else None
            
            features = self.extract_comprehensive_features(image_path, mask_path)
            
            if features is not None:
                features['image_path'] = image_path
                features['label'] = row['label']
                features['tamper_type'] = row['tamper_type']
                features['category'] = row['category']
                features['has_mask'] = row['has_mask']
                all_features.append(features)
            else:
                failed_extractions += 1
                logger.warning(f"Failed to extract features from {image_path}")
        
        logger.info(f"Feature extraction completed!")
        logger.info(f"  Successful extractions: {len(all_features)}")
        logger.info(f"  Failed extractions: {failed_extractions}")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        
        # Save features
        features_path = f"{self.output_path}/feature_vectors/comprehensive_features.csv"
        features_df.to_csv(features_path, index=False)
        
        logger.info(f"Features saved to: {features_path}")
        
        return features_df
    
    def generate_feature_analysis_report(self, features_df):
        """Generate comprehensive feature analysis report"""
        logger.info("="*80)
        logger.info("GENERATING FEATURE ANALYSIS REPORT")
        logger.info("="*80)
        
        # Feature statistics
        feature_columns = [col for col in features_df.columns 
                          if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]
        
        # Feature importance analysis
        self._plot_feature_importance(features_df, feature_columns)
        
        # Feature correlation analysis
        self._plot_feature_correlation(features_df, feature_columns)
        
        # Class-wise feature analysis
        self._plot_class_wise_features(features_df, feature_columns)
        
        # Generate text report
        self._generate_feature_text_report(features_df, feature_columns)
        
        logger.info("Feature analysis report generated!")
    
    def _plot_feature_importance(self, df, feature_columns):
        """Plot feature importance analysis"""
        plt.figure(figsize=(15, 10))
        
        # Feature variance
        plt.subplot(2, 2, 1)
        feature_variance = df[feature_columns].var().sort_values(ascending=False)
        top_features = feature_variance.head(20)
        plt.barh(range(len(top_features)), top_features.values)
        plt.yticks(range(len(top_features)), top_features.index)
        plt.title('Top 20 Features by Variance')
        plt.xlabel('Variance')
        
        # Feature correlation with target
        plt.subplot(2, 2, 2)
        correlations = df[feature_columns].corrwith(df['label']).abs().sort_values(ascending=False)
        top_correlated = correlations.head(20)
        plt.barh(range(len(top_correlated)), top_correlated.values)
        plt.yticks(range(len(top_correlated)), top_correlated.index)
        plt.title('Top 20 Features by Correlation with Target')
        plt.xlabel('Absolute Correlation')
        
        # Feature distribution by class
        plt.subplot(2, 2, 3)
        feature_means_original = df[df['label'] == 0][feature_columns].mean()
        feature_means_tampered = df[df['label'] == 1][feature_columns].mean()
        feature_diff = np.abs(feature_means_original - feature_means_tampered).sort_values(ascending=False)
        top_diff = feature_diff.head(20)
        plt.barh(range(len(top_diff)), top_diff.values)
        plt.yticks(range(len(top_diff)), top_diff.index)
        plt.title('Top 20 Features by Class Difference')
        plt.xlabel('Absolute Difference')
        
        # Feature completeness
        plt.subplot(2, 2, 4)
        completeness = (1 - df[feature_columns].isnull().sum() / len(df)) * 100
        plt.hist(completeness, bins=20, alpha=0.7, edgecolor='black')
        plt.title('Feature Completeness Distribution')
        plt.xlabel('Completeness (%)')
        plt.ylabel('Number of Features')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/feature_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_correlation(self, df, feature_columns):
        """Plot feature correlation matrix"""
        plt.figure(figsize=(12, 10))
        
        # Select top features for correlation analysis
        correlations = df[feature_columns].corrwith(df['label']).abs().sort_values(ascending=False)
        top_features = correlations.head(20).index.tolist()
        
        correlation_matrix = df[top_features].corr()
        
        sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'shrink': 0.8})
        plt.title('Feature Correlation Matrix (Top 20 Features)')
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/feature_correlation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_wise_features(self, df, feature_columns):
        """Plot class-wise feature analysis"""
        plt.figure(figsize=(15, 10))
        
        # Feature means by class
        plt.subplot(2, 2, 1)
        original_means = df[df['label'] == 0][feature_columns].mean()
        tampered_means = df[df['label'] == 1][feature_columns].mean()
        
        top_features = df[feature_columns].corrwith(df['label']).abs().sort_values(ascending=False).head(10).index
        
        x = np.arange(len(top_features))
        width = 0.35
        
        plt.bar(x - width/2, original_means[top_features], width, label='Original', alpha=0.8)
        plt.bar(x + width/2, tampered_means[top_features], width, label='Tampered', alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Mean Value')
        plt.title('Top 10 Features: Mean Values by Class')
        plt.xticks(x, top_features, rotation=45)
        plt.legend()
        
        # Feature standard deviations by class
        plt.subplot(2, 2, 2)
        original_stds = df[df['label'] == 0][feature_columns].std()
        tampered_stds = df[df['label'] == 1][feature_columns].std()
        
        plt.bar(x - width/2, original_stds[top_features], width, label='Original', alpha=0.8)
        plt.bar(x + width/2, tampered_stds[top_features], width, label='Tampered', alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Standard Deviation')
        plt.title('Top 10 Features: Standard Deviations by Class')
        plt.xticks(x, top_features, rotation=45)
        plt.legend()
        
        # Box plots for top features
        plt.subplot(2, 1, 2)
        top_5_features = top_features[:5]
        df_melted = df[['label'] + list(top_5_features)].melt(id_vars='label', var_name='feature', value_name='value')
        df_melted['label'] = df_melted['label'].map({0: 'Original', 1: 'Tampered'})
        
        sns.boxplot(data=df_melted, x='feature', y='value', hue='label')
        plt.title('Distribution of Top 5 Features by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/class_wise_features.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_feature_text_report(self, df, feature_columns):
        """Generate comprehensive feature text report"""
        report_path = f"{self.output_path}/reports/feature_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE OBJECTIVE 2 FEATURE EXTRACTION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Feature Extraction Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Feature Overview
            f.write("FEATURE OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Features Extracted: {len(feature_columns)}\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Original Samples: {len(df[df['label'] == 0])}\n")
            f.write(f"Tampered Samples: {len(df[df['label'] == 1])}\n\n")
            
            # Feature Categories
            f.write("FEATURE CATEGORIES\n")
            f.write("-"*40 + "\n")
            feature_categories = {
                'Statistical': [f for f in feature_columns if any(x in f for x in ['mean', 'std', 'skew', 'kurtosis'])],
                'Texture': [f for f in feature_columns if any(x in f for x in ['lbp', 'glcm', 'gabor', 'haralick', 'tamura'])],
                'Edge': [f for f in feature_columns if any(x in f for x in ['sobel', 'roberts', 'scharr', 'canny', 'hog', 'gradient'])],
                'Frequency': [f for f in feature_columns if any(x in f for x in ['fft', 'dct', 'wavelet'])],
                'ELA': [f for f in feature_columns if 'ela' in f],
                'Noise': [f for f in feature_columns if 'noise' in f],
                'Tampering': [f for f in feature_columns if any(x in f for x in ['block_similarity', 'texture_chi_square'])],
                'Mask-aware': [f for f in feature_columns if any(x in f for x in ['masked', 'non_masked', 'mask_'])],
            }
            
            for category, features in feature_categories.items():
                if features:
                    f.write(f"{category}: {len(features)} features\n")
            
            f.write("\n")
            
            # Top Features by Correlation
            f.write("TOP FEATURES BY CORRELATION WITH TARGET\n")
            f.write("-"*40 + "\n")
            correlations = df[feature_columns].corrwith(df['label']).abs().sort_values(ascending=False)
            for i, (feature, corr) in enumerate(correlations.head(20).items()):
                f.write(f"{i+1:2d}. {feature}: {corr:.4f}\n")
            
            f.write("\n")
            
            # Feature Quality Analysis
            f.write("FEATURE QUALITY ANALYSIS\n")
            f.write("-"*40 + "\n")
            
            # Missing values
            missing_counts = df[feature_columns].isnull().sum()
            features_with_missing = missing_counts[missing_counts > 0]
            f.write(f"Features with missing values: {len(features_with_missing)}\n")
            if len(features_with_missing) > 0:
                for feature, count in features_with_missing.head(10).items():
                    f.write(f"  {feature}: {count} missing values\n")
            
            # Zero variance features
            feature_variance = df[feature_columns].var()
            zero_variance_features = feature_variance[feature_variance == 0]
            f.write(f"Zero variance features: {len(zero_variance_features)}\n")
            
            # Constant features
            constant_features = []
            for feature in feature_columns:
                if df[feature].nunique() == 1:
                    constant_features.append(feature)
            f.write(f"Constant features: {len(constant_features)}\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("FEATURE EXTRACTION RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write("1. Remove zero variance and constant features\n")
            f.write("2. Handle missing values appropriately\n")
            f.write("3. Use feature selection techniques to reduce dimensionality\n")
            f.write("4. Consider feature engineering for better discriminative power\n")
            f.write("5. Utilize mask-aware features for tampered regions\n")
            f.write("6. Apply feature scaling before model training\n")
            f.write("7. Use ensemble methods to combine different feature types\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_feature_extraction(self):
        """Run complete feature extraction pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE OBJECTIVE 2 FEATURE EXTRACTION")
        logger.info("="*80)
        
        # Extract features from inventory
        features_df = self.extract_features_from_inventory()
        
        # Generate feature analysis report
        self.generate_feature_analysis_report(features_df)
        
        logger.info("✅ COMPREHENSIVE FEATURE EXTRACTION COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        
        return features_df

def main():
    """Main function to run comprehensive feature extraction"""
    extractor = ComprehensiveObjective2FeatureExtractor()
    features_df = extractor.run_complete_feature_extraction()
    
    print("\n" + "="*80)
    print("FEATURE EXTRACTION SUMMARY")
    print("="*80)
    print(f"Total features extracted: {len(features_df.columns) - 5}")  # Excluding metadata columns
    print(f"Total samples processed: {len(features_df)}")
    print(f"Original samples: {len(features_df[features_df['label'] == 0])}")
    print(f"Tampered samples: {len(features_df[features_df['label'] == 1])}")
    print("\nNext step: Run baseline ML models with these comprehensive features!")

if __name__ == "__main__":
    main()
