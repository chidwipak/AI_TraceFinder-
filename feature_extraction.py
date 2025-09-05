#!/usr/bin/env python3
"""
Feature Extraction for AI_TraceFinder - Objective 1: Scanner Identification
Extracts comprehensive features from preprocessed images for machine learning
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import pywt
from scipy.fft import fft2
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureExtractor:
    def __init__(self, preprocessed_path="preprocessed", prnu_path="prnu_fingerprints", output_path="features"):
        self.preprocessed_path = preprocessed_path
        self.prnu_path = prnu_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Create output directories
        self.create_output_directories()
        
        # Load PRNU fingerprints
        self.load_prnu_fingerprints()
        
        # Initialize feature storage
        self.features = []
        self.failed_images = []
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/feature_vectors",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/metadata"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_prnu_fingerprints(self):
        """Load PRNU fingerprints for correlation computation"""
        logger.info("Loading PRNU fingerprints...")
        
        pickle_filepath = os.path.join(self.prnu_path, "fingerprints", "all_prnu_fingerprints.pkl")
        
        try:
            with open(pickle_filepath, 'rb') as f:
                self.prnu_fingerprints = pickle.load(f)
            logger.info(f"Loaded PRNU fingerprints for {len(self.prnu_fingerprints)} scanners")
        except Exception as e:
            logger.error(f"Failed to load PRNU fingerprints: {e}")
            self.prnu_fingerprints = {}
    
    def extract_noise_residuals(self, image):
        """Extract noise residuals using wavelet denoising (same as PRNU extraction)"""
        # Apply wavelet denoising
        coeffs = pywt.wavedec2(image, 'db8', level=3)
        
        # Threshold coefficients to remove noise
        threshold = np.std(coeffs[0]) * 0.1
        coeffs_thresholded = []
        
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep approximation coefficients
                coeffs_thresholded.append(coeff)
            else:
                # Threshold detail coefficients (coeff is a tuple for wavedec2)
                if isinstance(coeff, tuple):
                    coeff_thresholded = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff)
                else:
                    coeff_thresholded = pywt.threshold(coeff, threshold, mode='soft')
                coeffs_thresholded.append(coeff_thresholded)
        
        # Reconstruct denoised image
        denoised_image = pywt.waverec2(coeffs_thresholded, 'db8')
        
        # Ensure same size as original
        if denoised_image.shape != image.shape:
            denoised_image = denoised_image[:image.shape[0], :image.shape[1]]
        
        # Calculate residuals
        residuals = image - denoised_image
        
        # Normalize residuals (zero-mean, unit-std)
        residuals = residuals - np.mean(residuals)
        residuals = residuals / (np.std(residuals) + 1e-8)
        
        return residuals
    
    def compute_correlation_features(self, residuals, dpi):
        """Task 1: Compute correlation features with PRNU fingerprints"""
        correlation_features = {}
        
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            if dpi in dpi_fingerprints:
                fingerprint = dpi_fingerprints[dpi]
                
                # Compute Normalized Cross Correlation (NCC)
                # NCC = (residuals - mean) * (fingerprint - mean) / (std_residuals * std_fingerprint)
                mean_residuals = np.mean(residuals)
                mean_fingerprint = np.mean(fingerprint)
                std_residuals = np.std(residuals)
                std_fingerprint = np.std(fingerprint)
                
                if std_residuals > 0 and std_fingerprint > 0:
                    # Element-wise correlation
                    correlation = np.sum((residuals - mean_residuals) * (fingerprint - mean_fingerprint))
                    correlation = correlation / (std_residuals * std_fingerprint * residuals.size)
                else:
                    correlation = 0.0
                
                correlation_features[f'ncc_{scanner_id}_{dpi}dpi'] = correlation
        
        return correlation_features
    
    def extract_texture_features(self, image):
        """Task 1: Extract texture features (LBP, Haralick/GLCM)"""
        texture_features = {}
        
        # Local Binary Patterns (LBP)
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image, n_points, radius, method='uniform')
        
        # LBP histogram
        lbp_hist, _ = np.histogram(lbp, bins=n_points + 2, range=(0, n_points + 2), density=True)
        for i, val in enumerate(lbp_hist):
            texture_features[f'lbp_hist_{i}'] = val
        
        # Gray-Level Co-occurrence Matrix (GLCM) features
        # Convert to uint8 for GLCM
        image_uint8 = (image * 255).astype(np.uint8)
        
        # Compute GLCM for different directions
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        for distance in distances:
            for angle in angles:
                glcm = graycomatrix(image_uint8, [distance], [np.radians(angle)], 
                                  levels=256, symmetric=True, normed=True)
                
                # Extract Haralick features
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                ASM = graycoprops(glcm, 'ASM')[0, 0]
                
                texture_features[f'glcm_contrast_d{distance}_a{angle}'] = contrast
                texture_features[f'glcm_dissimilarity_d{distance}_a{angle}'] = dissimilarity
                texture_features[f'glcm_homogeneity_d{distance}_a{angle}'] = homogeneity
                texture_features[f'glcm_energy_d{distance}_a{angle}'] = energy
                texture_features[f'glcm_correlation_d{distance}_a{angle}'] = correlation
                texture_features[f'glcm_asm_d{distance}_a{angle}'] = ASM
        
        return texture_features
    
    def extract_frequency_features(self, image):
        """Task 1: Extract frequency domain features (FFT, DCT)"""
        frequency_features = {}
        
        # Fast Fourier Transform (FFT)
        fft_result = fft2(image)
        fft_magnitude = np.abs(fft_result)
        fft_phase = np.angle(fft_result)
        
        # FFT magnitude statistics
        frequency_features['fft_magnitude_mean'] = np.mean(fft_magnitude)
        frequency_features['fft_magnitude_std'] = np.std(fft_magnitude)
        frequency_features['fft_magnitude_skew'] = skew(fft_magnitude.flatten())
        frequency_features['fft_magnitude_kurtosis'] = kurtosis(fft_magnitude.flatten())
        
        # FFT phase statistics
        frequency_features['fft_phase_mean'] = np.mean(fft_phase)
        frequency_features['fft_phase_std'] = np.std(fft_phase)
        frequency_features['fft_phase_skew'] = skew(fft_phase.flatten())
        frequency_features['fft_phase_kurtosis'] = kurtosis(fft_phase.flatten())
        
        # Discrete Cosine Transform (DCT)
        dct_result = dct(dct(image, axis=0), axis=1)  # 2D DCT
        dct_magnitude = np.abs(dct_result)
        
        # DCT statistics
        frequency_features['dct_magnitude_mean'] = np.mean(dct_magnitude)
        frequency_features['dct_magnitude_std'] = np.std(dct_magnitude)
        frequency_features['dct_magnitude_skew'] = skew(dct_magnitude.flatten())
        frequency_features['dct_magnitude_kurtosis'] = kurtosis(dct_magnitude.flatten())
        
        # Energy distribution in frequency bands
        h, w = fft_magnitude.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency band (center region)
        low_freq = fft_magnitude[center_h-32:center_h+32, center_w-32:center_w+32]
        frequency_features['low_freq_energy'] = np.sum(low_freq**2)
        
        # Medium frequency band
        medium_freq = fft_magnitude[center_h-64:center_h+64, center_w-64:center_w+64]
        frequency_features['medium_freq_energy'] = np.sum(medium_freq**2)
        
        # High frequency band (edges) - use a different approach
        # Create a mask for high frequency regions
        mask = np.ones_like(fft_magnitude)
        mask[center_h-64:center_h+64, center_w-64:center_w+64] = 0
        high_freq = fft_magnitude * mask
        frequency_features['high_freq_energy'] = np.sum(high_freq**2)
        
        return frequency_features
    
    def extract_statistical_features(self, image):
        """Task 1: Extract statistical features"""
        statistical_features = {}
        
        # Basic statistics
        statistical_features['mean'] = np.mean(image)
        statistical_features['std'] = np.std(image)
        statistical_features['variance'] = np.var(image)
        statistical_features['skewness'] = skew(image.flatten())
        statistical_features['kurtosis'] = kurtosis(image.flatten())
        
        # Histogram moments
        hist, bins = np.histogram(image, bins=256, range=(0, 1), density=True)
        statistical_features['hist_mean'] = np.mean(hist)
        statistical_features['hist_std'] = np.std(hist)
        statistical_features['hist_skew'] = skew(hist)
        statistical_features['hist_kurtosis'] = kurtosis(hist)
        
        # Percentiles
        percentiles = [10, 25, 50, 75, 90]
        for p in percentiles:
            statistical_features[f'percentile_{p}'] = np.percentile(image, p)
        
        # Range and interquartile range
        statistical_features['range'] = np.max(image) - np.min(image)
        statistical_features['iqr'] = np.percentile(image, 75) - np.percentile(image, 25)
        
        # Edge density
        sobel_x = ndimage.sobel(image, axis=0)
        sobel_y = ndimage.sobel(image, axis=1)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        statistical_features['edge_density'] = np.mean(edge_magnitude)
        statistical_features['edge_variance'] = np.var(edge_magnitude)
        
        return statistical_features
    
    def extract_all_features(self, image_path, scanner_id, dpi, source):
        """Extract all features for a single image"""
        try:
            # Load image
            image = np.array(Image.open(image_path).convert('L')).astype(np.float32) / 255.0
            
            # Extract noise residuals
            residuals = self.extract_noise_residuals(image)
            
            # Compute all feature types
            correlation_features = self.compute_correlation_features(residuals, dpi)
            texture_features = self.extract_texture_features(image)
            frequency_features = self.extract_frequency_features(image)
            statistical_features = self.extract_statistical_features(image)
            
            # Combine all features
            all_features = {
                'filepath': image_path,
                'scanner_id': scanner_id,
                'dpi': dpi,
                'source': source
            }
            all_features.update(correlation_features)
            all_features.update(texture_features)
            all_features.update(frequency_features)
            all_features.update(statistical_features)
            
            return all_features
            
        except Exception as e:
            logger.error(f"Failed to extract features from {image_path}: {e}")
            self.failed_images.append({
                'filepath': image_path,
                'scanner_id': scanner_id,
                'dpi': dpi,
                'source': source,
                'error': str(e)
            })
            return None
    
    def process_dataset(self):
        """Task 2: Process all preprocessed images and extract features"""
        logger.info("Processing preprocessed dataset for feature extraction...")
        
        # Load metadata
        train_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_train.csv")
        test_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_test.csv")
        
        if not os.path.exists(train_metadata_path) or not os.path.exists(test_metadata_path):
            logger.error("Metadata files not found. Please run preprocessing first.")
            return
        
        train_df = pd.read_csv(train_metadata_path)
        test_df = pd.read_csv(test_metadata_path)
        
        # Process training data
        logger.info(f"Processing {len(train_df)} training images...")
        for idx, row in train_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing training image {idx}/{len(train_df)}")
            
            # Construct image path
            image_filename = os.path.basename(row['filepath']).replace('.tif', '.png')
            image_path = os.path.join(self.preprocessed_path, "train", row['scanner_id'], 
                                    str(row['dpi']), image_filename)
            
            if os.path.exists(image_path):
                features = self.extract_all_features(image_path, row['scanner_id'], 
                                                   row['dpi'], row['source'])
                if features:
                    features['split'] = 'train'
                    self.features.append(features)
        
        # Process test data
        logger.info(f"Processing {len(test_df)} test images...")
        for idx, row in test_df.iterrows():
            if idx % 50 == 0:
                logger.info(f"Processing test image {idx}/{len(test_df)}")
            
            # Construct image path
            image_filename = os.path.basename(row['filepath']).replace('.tif', '.png')
            image_path = os.path.join(self.preprocessed_path, "test", row['scanner_id'], 
                                    str(row['dpi']), image_filename)
            
            if os.path.exists(image_path):
                features = self.extract_all_features(image_path, row['scanner_id'], 
                                                   row['dpi'], row['source'])
                if features:
                    features['split'] = 'test'
                    self.features.append(features)
        
        logger.info(f"Feature extraction complete. Extracted features from {len(self.features)} images")
    
    def save_feature_dataset(self):
        """Task 3: Save feature dataset in CSV/NumPy format"""
        logger.info("Saving feature dataset...")
        
        if not self.features:
            logger.error("No features extracted. Please run feature extraction first.")
            return
        
        # Convert to DataFrame
        features_df = pd.DataFrame(self.features)
        
        # Save as CSV
        csv_path = os.path.join(self.output_path, "feature_vectors", "scanner_features.csv")
        features_df.to_csv(csv_path, index=False)
        logger.info(f"Saved features to {csv_path}")
        
        # Save as NumPy arrays
        # Separate features and labels
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = features_df[feature_columns].values
        y = features_df['scanner_id'].values
        split = features_df['split'].values
        
        # Save arrays
        np.save(os.path.join(self.output_path, "feature_vectors", "X_features.npy"), X)
        np.save(os.path.join(self.output_path, "feature_vectors", "y_labels.npy"), y)
        np.save(os.path.join(self.output_path, "feature_vectors", "split_indices.npy"), split)
        
        logger.info(f"Saved feature arrays: X shape {X.shape}, y shape {y.shape}")
        
        # Save feature names
        feature_names = pd.DataFrame({'feature_name': feature_columns})
        feature_names.to_csv(os.path.join(self.output_path, "feature_vectors", "feature_names.csv"), index=False)
        
        return features_df
    
    def generate_feature_report(self, features_df):
        """Task 4: Report feature distributions and summary statistics"""
        logger.info("Generating feature report...")
        
        # Basic statistics
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        print("\n" + "="*60)
        print("FEATURE EXTRACTION REPORT")
        print("="*60)
        
        print(f"\nDataset Summary:")
        print(f"Total images processed: {len(features_df)}")
        print(f"Training images: {len(features_df[features_df['split'] == 'train'])}")
        print(f"Test images: {len(features_df[features_df['split'] == 'test'])}")
        print(f"Failed images: {len(self.failed_images)}")
        print(f"Total features extracted: {len(feature_columns)}")
        
        print(f"\nFeature Categories:")
        ncc_features = [col for col in feature_columns if col.startswith('ncc_')]
        lbp_features = [col for col in feature_columns if col.startswith('lbp_')]
        glcm_features = [col for col in feature_columns if col.startswith('glcm_')]
        fft_features = [col for col in feature_columns if col.startswith('fft_')]
        dct_features = [col for col in feature_columns if col.startswith('dct_')]
        stat_features = [col for col in feature_columns if not any(col.startswith(prefix) 
                                                                 for prefix in ['ncc_', 'lbp_', 'glcm_', 'fft_', 'dct_'])]
        
        print(f"  Correlation features (NCC): {len(ncc_features)}")
        print(f"  Texture features (LBP): {len(lbp_features)}")
        print(f"  Texture features (GLCM): {len(glcm_features)}")
        print(f"  Frequency features (FFT): {len(fft_features)}")
        print(f"  Frequency features (DCT): {len(dct_features)}")
        print(f"  Statistical features: {len(stat_features)}")
        
        print(f"\nPer-Scanner Distribution:")
        scanner_counts = features_df['scanner_id'].value_counts()
        for scanner, count in scanner_counts.items():
            print(f"  {scanner}: {count}")
        
        print(f"\nPer-DPI Distribution:")
        dpi_counts = features_df['dpi'].value_counts()
        for dpi, count in dpi_counts.items():
            print(f"  {dpi} DPI: {count}")
        
        # Feature statistics
        print(f"\nFeature Statistics (first 10 features):")
        for i, feature in enumerate(feature_columns[:10]):
            values = features_df[feature].values
            print(f"  {feature}:")
            print(f"    Mean: {np.mean(values):.6f}")
            print(f"    Std: {np.std(values):.6f}")
            print(f"    Min: {np.min(values):.6f}")
            print(f"    Max: {np.max(values):.6f}")
        
        # Save detailed report
        report_path = os.path.join(self.output_path, "metadata", "feature_extraction_report.txt")
        with open(report_path, 'w') as f:
            f.write("FEATURE EXTRACTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total images processed: {len(features_df)}\n")
            f.write(f"Training images: {len(features_df[features_df['split'] == 'train'])}\n")
            f.write(f"Test images: {len(features_df[features_df['split'] == 'test'])}\n")
            f.write(f"Failed images: {len(self.failed_images)}\n")
            f.write(f"Total features extracted: {len(feature_columns)}\n\n")
            
            f.write("Feature Categories:\n")
            f.write(f"  Correlation features (NCC): {len(ncc_features)}\n")
            f.write(f"  Texture features (LBP): {len(lbp_features)}\n")
            f.write(f"  Texture features (GLCM): {len(glcm_features)}\n")
            f.write(f"  Frequency features (FFT): {len(fft_features)}\n")
            f.write(f"  Frequency features (DCT): {len(dct_features)}\n")
            f.write(f"  Statistical features: {len(stat_features)}\n\n")
            
            f.write("Per-Scanner Distribution:\n")
            for scanner, count in scanner_counts.items():
                f.write(f"  {scanner}: {count}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def visualize_feature_distributions(self, features_df):
        """Create visualizations of feature distributions"""
        logger.info("Creating feature distribution visualizations...")
        
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        # Select a few representative features for visualization
        sample_features = []
        
        # One correlation feature
        ncc_features = [col for col in feature_columns if col.startswith('ncc_')]
        if ncc_features:
            sample_features.append(ncc_features[0])
        
        # One texture feature
        lbp_features = [col for col in feature_columns if col.startswith('lbp_')]
        if lbp_features:
            sample_features.append(lbp_features[0])
        
        # One frequency feature
        fft_features = [col for col in feature_columns if col.startswith('fft_')]
        if fft_features:
            sample_features.append(fft_features[0])
        
        # One statistical feature
        stat_features = [col for col in feature_columns if not any(col.startswith(prefix) 
                                                                 for prefix in ['ncc_', 'lbp_', 'glcm_', 'fft_', 'dct_'])]
        if stat_features:
            sample_features.append(stat_features[0])
        
        if len(sample_features) >= 4:
            # Create distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, feature in enumerate(sample_features):
                # Plot distribution by scanner
                for scanner in features_df['scanner_id'].unique():
                    scanner_data = features_df[features_df['scanner_id'] == scanner][feature]
                    axes[i].hist(scanner_data, alpha=0.7, label=scanner, bins=30)
                
                axes[i].set_title(f'{feature} Distribution by Scanner')
                axes[i].set_xlabel('Feature Value')
                axes[i].set_ylabel('Frequency')
                axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "visualizations", "feature_distributions.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature distribution visualizations saved")
    
    def run_full_pipeline(self):
        """Run the complete feature extraction pipeline"""
        logger.info("Starting feature extraction pipeline...")
        
        # Task 1 & 2: Extract features from all images
        self.process_dataset()
        
        # Task 3: Save feature dataset
        features_df = self.save_feature_dataset()
        
        # Task 4: Generate report and visualizations
        if features_df is not None:
            self.generate_feature_report(features_df)
            self.visualize_feature_distributions(features_df)
        
        logger.info("Feature extraction pipeline complete!")

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.run_full_pipeline()
