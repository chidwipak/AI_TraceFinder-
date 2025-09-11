#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Feature Extraction
Extracts forensic features for tampered image detection
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
from scipy import ndimage
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from skimage import filters, feature, measure
from skimage.filters import gabor
import pywt
import mahotas as mh
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class ForensicFeatureExtractor:
    """Extract forensic features for tampered image detection"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.feature_names = []
        self.scaler = StandardScaler()
        
    def extract_all_features(self, image_path):
        """Extract all forensic features from an image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)
        
        features = {}
        
        # 1. Error Level Analysis (ELA)
        features.update(self._extract_ela_features(image))
        
        # 2. Noise Analysis
        features.update(self._extract_noise_features(image))
        
        # 3. Compression Artifacts
        features.update(self._extract_compression_features(image))
        
        # 4. Frequency Domain Features
        features.update(self._extract_frequency_features(image))
        
        # 5. Texture Features
        features.update(self._extract_texture_features(image))
        
        # 6. Statistical Features
        features.update(self._extract_statistical_features(image))
        
        # 7. Edge and Gradient Features
        features.update(self._extract_edge_features(image))
        
        # 8. Color Features
        features.update(self._extract_color_features(image))
        
        # 9. Wavelet Features
        features.update(self._extract_wavelet_features(image))
        
        # 10. Local Binary Pattern Features
        features.update(self._extract_lbp_features(image))
        
        return features
    
    def _extract_ela_features(self, image):
        """Extract Error Level Analysis features"""
        features = {}
        
        # Convert to grayscale for ELA
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # ELA calculation (simplified version)
        # Save image with different quality and compare
        temp_path = 'temp_ela.jpg'
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        compressed = cv2.imread(temp_path)
        compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
        compressed = cv2.resize(compressed, self.target_size)
        
        # Calculate difference
        ela_diff = np.abs(image.astype(np.float32) - compressed.astype(np.float32))
        
        # ELA features
        features['ela_mean'] = np.mean(ela_diff)
        features['ela_std'] = np.std(ela_diff)
        features['ela_max'] = np.max(ela_diff)
        features['ela_skew'] = skew(ela_diff.flatten())
        features['ela_kurtosis'] = kurtosis(ela_diff.flatten())
        
        # ELA variance across channels
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'ela_{channel}_mean'] = np.mean(ela_diff[:, :, i])
            features[f'ela_{channel}_std'] = np.std(ela_diff[:, :, i])
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return features
    
    def _extract_noise_features(self, image):
        """Extract noise analysis features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Gaussian noise estimation
        # Apply Gaussian filter and calculate difference
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = gray.astype(np.float32) - blurred.astype(np.float32)
        
        features['noise_mean'] = np.mean(noise)
        features['noise_std'] = np.std(noise)
        features['noise_variance'] = np.var(noise)
        features['noise_skew'] = skew(noise.flatten())
        features['noise_kurtosis'] = kurtosis(noise.flatten())
        
        # Noise level estimation using different methods
        # Method 1: Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features['laplacian_variance'] = laplacian_var
        
        # Method 2: Sobel variance
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        features['sobel_variance'] = np.var(sobel_magnitude)
        
        return features
    
    def _extract_compression_features(self, image):
        """Extract compression artifact features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # DCT-based compression analysis
        # Apply DCT to 8x8 blocks
        dct_features = []
        for i in range(0, gray.shape[0]-8, 8):
            for j in range(0, gray.shape[1]-8, 8):
                block = gray[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                dct_features.append(dct_block.flatten())
        
        dct_features = np.array(dct_features)
        
        # DCT coefficient statistics
        features['dct_mean'] = np.mean(dct_features)
        features['dct_std'] = np.std(dct_features)
        features['dct_energy'] = np.sum(dct_features**2)
        
        # High frequency DCT coefficients (compression artifacts)
        high_freq_coeffs = dct_features[:, 1:]  # Exclude DC component
        features['high_freq_dct_mean'] = np.mean(high_freq_coeffs)
        features['high_freq_dct_std'] = np.std(high_freq_coeffs)
        
        # Blocking artifacts detection
        # Calculate horizontal and vertical differences at block boundaries
        h_diff = np.abs(np.diff(gray, axis=1))
        v_diff = np.abs(np.diff(gray, axis=0))
        
        # Blocking artifacts at 8-pixel intervals
        h_blocking = []
        v_blocking = []
        for i in range(7, gray.shape[0]-1, 8):
            v_blocking.extend(v_diff[i, :])
        for j in range(7, gray.shape[1]-1, 8):
            h_blocking.extend(h_diff[:, j])
        
        features['blocking_artifacts_h'] = np.mean(h_blocking) if h_blocking else 0
        features['blocking_artifacts_v'] = np.mean(v_blocking) if v_blocking else 0
        
        return features
    
    def _extract_frequency_features(self, image):
        """Extract frequency domain features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # FFT analysis
        fft = fft2(gray)
        fft_shifted = fftshift(fft)
        magnitude = np.abs(fft_shifted)
        
        # Frequency domain statistics
        features['fft_mean'] = np.mean(magnitude)
        features['fft_std'] = np.std(magnitude)
        features['fft_energy'] = np.sum(magnitude**2)
        
        # High frequency energy
        h, w = magnitude.shape
        center_h, center_w = h//2, w//2
        high_freq_mask = np.zeros_like(magnitude)
        high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
        high_freq_mask = 1 - high_freq_mask
        
        features['high_freq_energy'] = np.sum(magnitude * high_freq_mask)
        features['high_freq_ratio'] = features['high_freq_energy'] / features['fft_energy']
        
        # Spectral centroid
        freq_x = np.arange(w) - center_w
        freq_y = np.arange(h) - center_h
        freq_mesh_x, freq_mesh_y = np.meshgrid(freq_x, freq_y)
        freq_magnitude = np.sqrt(freq_mesh_x**2 + freq_mesh_y**2)
        
        features['spectral_centroid'] = np.sum(freq_magnitude * magnitude) / np.sum(magnitude)
        
        return features
    
    def _extract_texture_features(self, image):
        """Extract texture features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Gabor filter responses
        gabor_responses = []
        for theta in [0, 45, 90, 135]:
            for frequency in [0.1, 0.3, 0.5]:
                real, _ = gabor(gray, frequency=frequency, theta=np.radians(theta))
                gabor_responses.append(real)
        
        gabor_responses = np.array(gabor_responses)
        features['gabor_mean'] = np.mean(gabor_responses)
        features['gabor_std'] = np.std(gabor_responses)
        features['gabor_energy'] = np.sum(gabor_responses**2)
        
        # Haralick texture features (simplified)
        # Calculate co-occurrence matrix for texture analysis
        glcm = feature.graycomatrix(gray.astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        
        # Contrast, dissimilarity, homogeneity, energy
        features['contrast'] = feature.graycoprops(glcm, 'contrast')[0, 0]
        features['dissimilarity'] = feature.graycoprops(glcm, 'dissimilarity')[0, 0]
        features['homogeneity'] = feature.graycoprops(glcm, 'homogeneity')[0, 0]
        features['energy'] = feature.graycoprops(glcm, 'energy')[0, 0]
        
        return features
    
    def _extract_statistical_features(self, image):
        """Extract statistical features"""
        features = {}
        
        # Per-channel statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = image[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_skew'] = skew(channel_data)
            features[f'{channel}_kurtosis'] = kurtosis(channel_data)
            features[f'{channel}_min'] = np.min(channel_data)
            features[f'{channel}_max'] = np.max(channel_data)
            features[f'{channel}_median'] = np.median(channel_data)
        
        # Overall image statistics
        features['overall_mean'] = np.mean(image)
        features['overall_std'] = np.std(image)
        features['overall_skew'] = skew(image.flatten())
        features['overall_kurtosis'] = kurtosis(image.flatten())
        
        return features
    
    def _extract_edge_features(self, image):
        """Extract edge and gradient features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features['edge_mean'] = np.mean(edges)
        
        # Sobel gradients
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        sobel_direction = np.arctan2(sobel_y, sobel_x)
        
        features['sobel_magnitude_mean'] = np.mean(sobel_magnitude)
        features['sobel_magnitude_std'] = np.std(sobel_magnitude)
        features['sobel_direction_mean'] = np.mean(sobel_direction)
        features['sobel_direction_std'] = np.std(sobel_direction)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_mean'] = np.mean(laplacian)
        features['laplacian_std'] = np.std(laplacian)
        features['laplacian_energy'] = np.sum(laplacian**2)
        
        return features
    
    def _extract_color_features(self, image):
        """Extract color-based features"""
        features = {}
        
        # RGB to HSV conversion
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # HSV statistics
        for i, channel in enumerate(['H', 'S', 'V']):
            channel_data = hsv[:, :, i].flatten()
            features[f'hsv_{channel}_mean'] = np.mean(channel_data)
            features[f'hsv_{channel}_std'] = np.std(channel_data)
            features[f'hsv_{channel}_skew'] = skew(channel_data)
        
        # Color moments
        # First moment (mean)
        features['color_moment_1'] = np.mean(image)
        
        # Second moment (variance)
        features['color_moment_2'] = np.var(image)
        
        # Third moment (skewness)
        features['color_moment_3'] = skew(image.flatten())
        
        # Color histogram features
        hist_r = cv2.calcHist([image], [0], None, [32], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [32], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [32], [0, 256])
        
        features['hist_r_entropy'] = -np.sum(hist_r * np.log(hist_r + 1e-10))
        features['hist_g_entropy'] = -np.sum(hist_g * np.log(hist_g + 1e-10))
        features['hist_b_entropy'] = -np.sum(hist_b * np.log(hist_b + 1e-10))
        
        return features
    
    def _extract_wavelet_features(self, image):
        """Extract wavelet transform features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Wavelet decomposition
        coeffs = pywt.dwt2(gray, 'db4')
        cA, (cH, cV, cD) = coeffs
        
        # Wavelet coefficient statistics
        features['wavelet_LL_mean'] = np.mean(cA)
        features['wavelet_LL_std'] = np.std(cA)
        features['wavelet_LL_energy'] = np.sum(cA**2)
        
        features['wavelet_LH_mean'] = np.mean(cH)
        features['wavelet_LH_std'] = np.std(cH)
        features['wavelet_LH_energy'] = np.sum(cH**2)
        
        features['wavelet_HL_mean'] = np.mean(cV)
        features['wavelet_HL_std'] = np.std(cV)
        features['wavelet_HL_energy'] = np.sum(cV**2)
        
        features['wavelet_HH_mean'] = np.mean(cD)
        features['wavelet_HH_std'] = np.std(cD)
        features['wavelet_HH_energy'] = np.sum(cD**2)
        
        # Energy ratios
        total_energy = features['wavelet_LL_energy'] + features['wavelet_LH_energy'] + features['wavelet_HL_energy'] + features['wavelet_HH_energy']
        features['wavelet_LL_ratio'] = features['wavelet_LL_energy'] / total_energy
        features['wavelet_HH_ratio'] = features['wavelet_HH_energy'] / total_energy
        
        return features
    
    def _extract_lbp_features(self, image):
        """Extract Local Binary Pattern features"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # LBP calculation (simplified version)
        # Using uniform LBP with radius=1, n_points=8
        lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
        
        # LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-10)
        
        # LBP features
        features['lbp_mean'] = np.mean(lbp)
        features['lbp_std'] = np.std(lbp)
        features['lbp_energy'] = np.sum(hist**2)
        features['lbp_entropy'] = -np.sum(hist * np.log(hist + 1e-10))
        
        return features

class SUPATLANTIQUEFeatureExtractor:
    """Main feature extraction class for SUPATLANTIQUE dataset"""
    
    def __init__(self, target_size=(224, 224)):
        self.extractor = ForensicFeatureExtractor(target_size)
        self.features_df = None
        
    def extract_features_from_inventory(self, inventory_path='objective2_data_inventory.csv'):
        """Extract features from all images in the inventory"""
        print("=== Extracting Forensic Features ===")
        
        # Load inventory
        inventory_df = pd.read_csv(inventory_path)
        print(f"Processing {len(inventory_df)} images...")
        
        all_features = []
        failed_images = []
        
        for idx, row in inventory_df.iterrows():
            try:
                print(f"Processing {idx+1}/{len(inventory_df)}: {Path(row['image_path']).name}")
                
                # Extract features
                features = self.extractor.extract_all_features(row['image_path'])
                
                # Add metadata
                features['image_path'] = row['image_path']
                features['label'] = row['label']
                features['tamper_type'] = row['tamper_type']
                features['category'] = row['category']
                
                all_features.append(features)
                
            except Exception as e:
                print(f"Error processing {row['image_path']}: {e}")
                failed_images.append(row['image_path'])
        
        # Create features DataFrame
        self.features_df = pd.DataFrame(all_features)
        
        print(f"\nFeature extraction complete:")
        print(f"  - Successfully processed: {len(all_features)} images")
        print(f"  - Failed: {len(failed_images)} images")
        print(f"  - Total features extracted: {len(self.features_df.columns) - 4}")  # Exclude metadata columns
        
        if failed_images:
            print(f"Failed images: {failed_images}")
        
        return self.features_df
    
    def save_features(self, output_path='objective2_forensic_features.csv'):
        """Save extracted features to CSV"""
        if self.features_df is None:
            raise ValueError("No features extracted. Run extract_features_from_inventory() first.")
        
        self.features_df.to_csv(output_path, index=False)
        print(f"Features saved to: {output_path}")
        
        return output_path
    
    def analyze_features(self):
        """Analyze extracted features"""
        if self.features_df is None:
            raise ValueError("No features extracted. Run extract_features_from_inventory() first.")
        
        print("\n=== Feature Analysis ===")
        
        # Separate features and metadata
        feature_cols = [col for col in self.features_df.columns if col not in ['image_path', 'label', 'tamper_type', 'category']]
        X = self.features_df[feature_cols]
        y = self.features_df['label']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution:")
        print(f"  - Original: {sum(y == 0)}")
        print(f"  - Tampered: {sum(y == 1)}")
        
        # Feature statistics
        print(f"\nFeature statistics:")
        print(f"  - Mean values range: {X.mean().min():.4f} to {X.mean().max():.4f}")
        print(f"  - Std values range: {X.std().min():.4f} to {X.std().max():.4f}")
        
        # Check for missing values
        missing_values = X.isnull().sum()
        if missing_values.sum() > 0:
            print(f"  - Missing values: {missing_values.sum()}")
            print(f"  - Features with missing values: {missing_values[missing_values > 0].index.tolist()}")
        else:
            print(f"  - No missing values")
        
        # Feature correlation analysis
        correlation_matrix = X.corr()
        high_corr_pairs = []
        
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_val = correlation_matrix.iloc[i, j]
                if abs(corr_val) > 0.9:
                    high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
        
        if high_corr_pairs:
            print(f"  - High correlation pairs (>0.9): {len(high_corr_pairs)}")
        else:
            print(f"  - No high correlation pairs found")
        
        return X, y
    
    def visualize_features(self):
        """Create feature visualizations"""
        if self.features_df is None:
            raise ValueError("No features extracted. Run extract_features_from_inventory() first.")
        
        print("\n=== Creating Feature Visualizations ===")
        
        # Create output directory
        os.makedirs('objective2_features', exist_ok=True)
        
        # Separate features and metadata
        feature_cols = [col for col in self.features_df.columns if col not in ['image_path', 'label', 'tamper_type', 'category']]
        X = self.features_df[feature_cols]
        y = self.features_df['label']
        
        # 1. Feature distribution by class
        plt.figure(figsize=(20, 15))
        
        # Select top 12 features for visualization
        feature_importance = X.std().sort_values(ascending=False)
        top_features = feature_importance.head(12).index
        
        for i, feature in enumerate(top_features):
            plt.subplot(3, 4, i + 1)
            
            # Plot distributions for each class
            original_data = X[y == 0][feature]
            tampered_data = X[y == 1][feature]
            
            plt.hist(original_data, alpha=0.7, label='Original', bins=20, color='green')
            plt.hist(tampered_data, alpha=0.7, label='Tampered', bins=20, color='red')
            
            plt.title(f'{feature}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('objective2_features/feature_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Correlation heatmap
        plt.figure(figsize=(15, 12))
        correlation_matrix = X.corr()
        
        # Select subset of features for correlation matrix
        subset_features = feature_importance.head(20).index
        subset_corr = correlation_matrix.loc[subset_features, subset_features]
        
        sns.heatmap(subset_corr, annot=False, cmap='coolwarm', center=0, square=True)
        plt.title('Feature Correlation Matrix (Top 20 Features)')
        plt.tight_layout()
        plt.savefig('objective2_features/feature_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance (variance-based)
        plt.figure(figsize=(12, 8))
        feature_importance.head(20).plot(kind='bar')
        plt.title('Top 20 Features by Standard Deviation')
        plt.xlabel('Features')
        plt.ylabel('Standard Deviation')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('objective2_features/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Feature visualizations saved to: objective2_features/")

def main():
    """Main feature extraction function"""
    extractor = SUPATLANTIQUEFeatureExtractor(target_size=(224, 224))
    
    # Extract features from all images
    features_df = extractor.extract_features_from_inventory()
    
    # Save features
    extractor.save_features()
    
    # Analyze features
    X, y = extractor.analyze_features()
    
    # Create visualizations
    extractor.visualize_features()
    
    print("\n=== Feature Extraction Complete ===")
    print("Next steps:")
    print("1. Review feature analysis and visualizations")
    print("2. Proceed with baseline ML models")
    print("3. Implement deep learning models")

if __name__ == "__main__":
    main()
