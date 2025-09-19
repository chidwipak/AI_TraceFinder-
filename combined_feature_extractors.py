#!/usr/bin/env python3
"""
Combined Feature Extractors for AI TraceFinder
Extracts features for both Objective 1 (Scanner ID) and Objective 2 (Tampered Detection)
Ensures perfect consistency with training pipelines
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import warnings
warnings.filterwarnings('ignore')

# Scientific computing
from scipy import ndimage, stats
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift, dct
from scipy.fftpack import dct as dct_legacy
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

class CombinedFeatureExtractor:
    """Unified feature extractor for both objectives"""
    
    def __init__(self):
        self.prnu_fingerprints = {}
        self.load_prnu_fingerprints()
        
    def load_prnu_fingerprints(self):
        """Load PRNU fingerprints for Objective 1"""
        try:
            pickle_filepath = "prnu_fingerprints/fingerprints/all_prnu_fingerprints.pkl"
            if os.path.exists(pickle_filepath):
                with open(pickle_filepath, 'rb') as f:
                    self.prnu_fingerprints = pickle.load(f)
                print(f"✅ Loaded PRNU fingerprints for {len(self.prnu_fingerprints)} scanners")
            else:
                print("⚠️ PRNU fingerprints not found - using fallback features")
                self.prnu_fingerprints = {}
        except Exception as e:
            print(f"❌ Error loading PRNU fingerprints: {e}")
            self.prnu_fingerprints = {}
    
    def extract_scanner_features(self, image):
        """Extract 153 features for Objective 1 (Scanner ID) - exact match with training"""
        try:
            # Ensure image is grayscale and normalized
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            if image.dtype != np.float64:
                image = image.astype(np.float64) / 255.0
            
            # Extract noise residuals for PRNU correlation
            residuals = self.extract_noise_residuals(image)
            if residuals is None:
                print("⚠️ Cannot extract noise residuals - using fallback")
                return self.extract_scanner_fallback_features(image)
            
            # Compute PRNU correlation features for both DPI
            correlation_features_150 = self.compute_correlation_features(residuals, 150)
            correlation_features_300 = self.compute_correlation_features(residuals, 300)
            
            # Extract other features
            texture_features = self.extract_texture_features(image)
            frequency_features = self.extract_frequency_features(image)
            statistical_features = self.extract_statistical_features(image)
            
            # Combine all features in the exact same order as training
            all_features = {}
            all_features.update(correlation_features_150)
            all_features.update(texture_features)
            all_features.update(frequency_features)
            all_features.update(statistical_features)
            all_features.update(correlation_features_300)
            
            # Convert to array in the exact same order as training
            feature_names = [
                # 1. NCC 150 DPI features (11 features)
                'ncc_Canon120-1_150dpi', 'ncc_Canon120-2_150dpi', 'ncc_Canon220_150dpi',
                'ncc_Canon9000-1_150dpi', 'ncc_Canon9000-2_150dpi', 'ncc_EpsonV39-1_150dpi',
                'ncc_EpsonV39-2_150dpi', 'ncc_EpsonV370-1_150dpi', 'ncc_EpsonV370-2_150dpi',
                'ncc_EpsonV550_150dpi', 'ncc_HP_150dpi'
            ]
            
            # 2. LBP histogram features (26 features)
            for i in range(26):
                feature_names.append(f'lbp_hist_{i}')
            
            # 3. GLCM features (72 features: 3 distances × 4 angles × 6 properties)
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
            for distance in distances:
                for angle in angles:
                    for prop in glcm_props:
                        feature_names.append(f'glcm_{prop}_d{distance}_a{angle}')
            
            # 4. Frequency features (15 features)
            freq_features = ['fft_magnitude_mean', 'fft_magnitude_std', 'fft_magnitude_skew', 'fft_magnitude_kurtosis',
                           'fft_phase_mean', 'fft_phase_std', 'fft_phase_skew', 'fft_phase_kurtosis',
                           'dct_magnitude_mean', 'dct_magnitude_std', 'dct_magnitude_skew', 'dct_magnitude_kurtosis',
                           'low_freq_energy', 'medium_freq_energy', 'high_freq_energy']
            feature_names.extend(freq_features)
            
            # 5. Statistical features (18 features)
            stat_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'hist_mean', 'hist_std', 'hist_skew', 'hist_kurtosis',
                           'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'range', 'iqr',
                           'edge_density', 'edge_variance']
            feature_names.extend(stat_features)
            
            # 6. NCC 300 DPI features (11 features)
            for scanner in ['Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2', 'EpsonV39-1',
                          'EpsonV39-2', 'EpsonV370-1', 'EpsonV370-2', 'EpsonV550', 'HP']:
                feature_names.append(f'ncc_{scanner}_300dpi')
            
            # Extract features in the correct order
            features = []
            for name in feature_names:
                if name in all_features:
                    features.append(all_features[name])
                else:
                    features.append(0.0)  # Default value for missing features
            
            # Ensure we have exactly 153 features
            while len(features) < 153:
                features.append(0.0)
            features = features[:153]
            
            return np.array(features)
            
        except Exception as e:
            print(f"❌ Error extracting scanner features: {e}")
            return self.extract_scanner_fallback_features(image)
    
    def extract_tampered_features(self, image):
        """Extract 84 features for Objective 2 (Tampered Detection) - exact match with training"""
        try:
            # Ensure image is RGB and proper size
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            
            # Resize to target size
            image = cv2.resize(image, (224, 224))
            
            # Extract features in exact order from CSV header
            features = {}
            
            # 1. Basic Statistical Features (20 features)
            features.update(self._extract_basic_statistical_features(image))
            
            # 2. HSV Features (3 features)
            features.update(self._extract_hsv_features(image))
            
            # 3. Texture Features (7 features)
            features.update(self._extract_texture_features_tampered(image))
            
            # 4. Edge Features (8 features)
            features.update(self._extract_edge_features_tampered(image))
            
            # 5. Frequency Features (6 features)
            features.update(self._extract_frequency_features_tampered(image))
            
            # 6. Wavelet Features (8 features)
            features.update(self._extract_wavelet_features_tampered(image))
            
            # 7. ELA Features (6 features)
            features.update(self._extract_ela_features_tampered(image))
            
            # 8. Noise Features (6 features)
            features.update(self._extract_noise_features_tampered(image))
            
            # 9. Tampering Features (6 features)
            features.update(self._extract_tampering_features_tampered(image))
            
            # 10. Advanced Features (10 features)
            features.update(self._extract_advanced_features_tampered(image))
            
            # Convert to array in exact order from CSV header
            feature_names = [
                'mean_r', 'mean_g', 'mean_b', 'mean_gray', 'std_r', 'std_g', 'std_b', 'std_gray',
                'skew_r', 'skew_g', 'skew_b', 'skew_gray', 'kurtosis_r', 'kurtosis_g', 'kurtosis_b',
                'kurtosis_gray', 'hist_entropy_r', 'hist_entropy_g', 'hist_entropy_b', 'mean_h', 'mean_s',
                'mean_v', 'lbp_hist_entropy', 'glcm_contrast', 'glcm_dissimilarity', 'glcm_homogeneity',
                'glcm_energy', 'glcm_correlation', 'glcm_asm', 'gabor_mean', 'gabor_std', 'sobel_mean',
                'sobel_std', 'sobel_max', 'roberts_mean', 'roberts_std', 'scharr_mean', 'scharr_std',
                'canny_edge_density', 'hog_mean', 'hog_std', 'hog_max', 'fft_mean', 'fft_std', 'fft_max',
                'high_freq_energy', 'dct_mean', 'dct_std', 'wavelet_energy_approx', 'wavelet_energy_horizontal',
                'wavelet_energy_vertical', 'wavelet_energy_diagonal', 'wavelet_entropy_approx', 'wavelet_entropy_horizontal',
                'wavelet_entropy_vertical', 'wavelet_entropy_diagonal', 'ela_mean', 'ela_std', 'ela_max',
                'ela_center_mean', 'ela_center_std', 'noise_mean', 'noise_std', 'noise_variance', 'high_freq_noise_std',
                'high_freq_noise_mean', 'block_similarity_max', 'block_similarity_mean', 'block_similarity_std',
                'gradient_mean', 'gradient_std', 'gradient_max', 'moment_00', 'moment_01', 'moment_10', 'moment_11',
                'haralick_contrast', 'haralick_dissimilarity', 'haralick_homogeneity', 'haralick_energy',
                'haralick_correlation', 'tamura_coarseness', 'tamura_contrast', 'tamura_directionality'
            ]
            
            # Extract features in the correct order
            feature_array = []
            for name in feature_names:
                if name in features:
                    feature_array.append(features[name])
                else:
                    feature_array.append(0.0)  # Default value for missing features
            
            return np.array(feature_array)
            
        except Exception as e:
            print(f"❌ Error extracting tampered features: {e}")
            return np.zeros(84)  # Return zero features if extraction fails
    
    def extract_noise_residuals(self, image):
        """Extract noise residuals using wavelet denoising (same as PRNU extraction)"""
        try:
            # Resize image to match PRNU fingerprint size (256x256)
            if image.shape != (256, 256):
                import cv2
                image_resized = cv2.resize(image, (256, 256))
            else:
                image_resized = image
            
            # Apply wavelet denoising
            coeffs = pywt.wavedec2(image_resized, 'db8', level=3)
            
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
            if denoised_image.shape != image_resized.shape:
                denoised_image = denoised_image[:image_resized.shape[0], :image_resized.shape[1]]
            
            # Calculate residuals
            residuals = image_resized - denoised_image
            
            # Normalize residuals (zero-mean, unit-std)
            residuals = residuals - np.mean(residuals)
            residuals = residuals / (np.std(residuals) + 1e-8)
            
            return residuals
        except Exception as e:
            print(f"❌ Error extracting noise residuals: {e}")
            return None
    
    def compute_correlation_features(self, residuals, dpi):
        """Compute correlation features with PRNU fingerprints"""
        correlation_features = {}
        
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            if dpi in dpi_fingerprints:
                fingerprint = dpi_fingerprints[dpi]
                
                # Compute Normalized Cross Correlation (NCC)
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
        """Extract texture features (LBP, GLCM) for scanner ID"""
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
                try:
                    glcm = graycomatrix(image_uint8, [distance], [np.radians(angle)], 
                                      levels=256, symmetric=True, normed=True)
                    
                    # GLCM properties
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']:
                        texture_features[f'glcm_{prop}_d{distance}_a{angle}'] = graycoprops(glcm, prop)[0, 0]
                except:
                    # If GLCM fails, use default values
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']:
                        texture_features[f'glcm_{prop}_d{distance}_a{angle}'] = 0.0
        
        return texture_features
    
    def extract_frequency_features(self, image):
        """Extract frequency domain features for scanner ID"""
        frequency_features = {}
        
        # FFT features
        fft_image = fft2(image)
        fft_magnitude = np.abs(fft_image)
        fft_phase = np.angle(fft_image)
        
        frequency_features['fft_magnitude_mean'] = np.mean(fft_magnitude)
        frequency_features['fft_magnitude_std'] = np.std(fft_magnitude)
        frequency_features['fft_magnitude_skew'] = skew(fft_magnitude.flatten())
        frequency_features['fft_magnitude_kurtosis'] = kurtosis(fft_magnitude.flatten())
        
        frequency_features['fft_phase_mean'] = np.mean(fft_phase)
        frequency_features['fft_phase_std'] = np.std(fft_phase)
        frequency_features['fft_phase_skew'] = skew(fft_phase.flatten())
        frequency_features['fft_phase_kurtosis'] = kurtosis(fft_phase.flatten())
        
        # DCT features
        dct_image = np.abs(dct_legacy(dct_legacy(image.T, norm='ortho').T, norm='ortho'))
        frequency_features['dct_magnitude_mean'] = np.mean(dct_image)
        frequency_features['dct_magnitude_std'] = np.std(dct_image)
        frequency_features['dct_magnitude_skew'] = skew(dct_image.flatten())
        frequency_features['dct_magnitude_kurtosis'] = kurtosis(dct_image.flatten())
        
        # Frequency energy features
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency energy (center region)
        low_freq_region = fft_magnitude[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        frequency_features['low_freq_energy'] = np.sum(low_freq_region)
        
        # Medium frequency energy
        medium_freq_region = fft_magnitude[center_h-h//3:center_h+h//3, center_w-w//3:center_w+w//3]
        frequency_features['medium_freq_energy'] = np.sum(medium_freq_region)
        
        # High frequency energy (entire spectrum)
        frequency_features['high_freq_energy'] = np.sum(fft_magnitude)
        
        return frequency_features
    
    def extract_statistical_features(self, image):
        """Extract statistical features for scanner ID"""
        statistical_features = {}
        
        # Basic statistics
        statistical_features['mean'] = np.mean(image)
        statistical_features['std'] = np.std(image)
        statistical_features['variance'] = np.var(image)
        statistical_features['skewness'] = skew(image.flatten())
        statistical_features['kurtosis'] = kurtosis(image.flatten())
        
        # Histogram statistics
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)  # Normalize
        statistical_features['hist_mean'] = np.mean(hist)
        statistical_features['hist_std'] = np.std(hist)
        statistical_features['hist_skew'] = skew(hist)
        statistical_features['hist_kurtosis'] = kurtosis(hist)
        
        # Percentiles
        statistical_features['percentile_10'] = np.percentile(image, 10)
        statistical_features['percentile_25'] = np.percentile(image, 25)
        statistical_features['percentile_50'] = np.percentile(image, 50)
        statistical_features['percentile_75'] = np.percentile(image, 75)
        statistical_features['percentile_90'] = np.percentile(image, 90)
        
        # Range and IQR
        statistical_features['range'] = np.max(image) - np.min(image)
        statistical_features['iqr'] = statistical_features['percentile_75'] - statistical_features['percentile_25']
        
        # Edge features
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        statistical_features['edge_density'] = np.sum(edge_magnitude > 0.1) / edge_magnitude.size
        statistical_features['edge_variance'] = np.var(edge_magnitude)
        
        return statistical_features
    
    def extract_scanner_fallback_features(self, image):
        """Fallback feature extraction for scanner ID if main extraction fails"""
        try:
            features = []
            
            # Basic features to reach 153
            # Statistical features
            features.extend([
                np.mean(image), np.std(image), np.var(image), np.median(image),
                np.min(image), np.max(image), np.ptp(image), skew(image.flatten()), kurtosis(image.flatten())
            ])
            
            # Fill remaining features with zeros to reach 153
            while len(features) < 153:
                features.append(0.0)
            
            return np.array(features[:153])
        except:
            return np.zeros(153)
    
    # Tampered Detection Feature Extraction Methods
    def _extract_basic_statistical_features(self, image):
        """Extract basic statistical features for tampered detection"""
        features = {}
        
        # Convert to grayscale for some features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic statistics for each channel
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
        
        # Histogram entropy for each channel
        for i, channel in enumerate(['r', 'g', 'b']):
            hist, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
            hist = hist / np.sum(hist)
            hist = hist[hist > 0]  # Remove zeros
            features[f'hist_entropy_{channel}'] = -np.sum(hist * np.log2(hist))
        
        return features
    
    def _extract_hsv_features(self, image):
        """Extract HSV features"""
        features = {}
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        features['mean_h'] = np.mean(hsv[:, :, 0])
        features['mean_s'] = np.mean(hsv[:, :, 1])
        features['mean_v'] = np.mean(hsv[:, :, 2])
        return features
    
    def _extract_texture_features_tampered(self, image):
        """Extract texture features for tampered detection"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # LBP entropy
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
        features['lbp_hist_entropy'] = shannon_entropy(lbp.astype(int))
        
        # GLCM features
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features['glcm_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features['glcm_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features['glcm_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features['glcm_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features['glcm_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
            features['glcm_asm'] = graycoprops(glcm, 'ASM')[0, 0]
        except:
            features.update({f'glcm_{prop}': 0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']})
        
        # Gabor features
        try:
            gabor_responses = []
            for theta in [0, 45, 90, 135]:
                for frequency in [0.1, 0.3]:
                    real, _ = gabor(gray, frequency=frequency, theta=np.deg2rad(theta))
                    gabor_responses.extend([np.mean(real), np.std(real)])
            features['gabor_mean'] = np.mean(gabor_responses)
            features['gabor_std'] = np.std(gabor_responses)
        except:
            features['gabor_mean'] = 0
            features['gabor_std'] = 0
        
        return features
    
    def _extract_edge_features_tampered(self, image):
        """Extract edge features for tampered detection"""
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
        except:
            features['hog_mean'] = 0
            features['hog_std'] = 0
            features['hog_max'] = 0
        
        return features
    
    def _extract_frequency_features_tampered(self, image):
        """Extract frequency features for tampered detection"""
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
    
    def _extract_wavelet_features_tampered(self, image):
        """Extract wavelet features for tampered detection"""
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
        except:
            features.update({f'wavelet_energy_{band}': 0 for band in ['approx', 'horizontal', 'vertical', 'diagonal']})
            features.update({f'wavelet_entropy_{band}': 0 for band in ['approx', 'horizontal', 'vertical', 'diagonal']})
        
        return features
    
    def _extract_ela_features_tampered(self, image):
        """Extract Error Level Analysis features"""
        features = {}
        
        try:
            # Convert to JPEG and back to detect compression artifacts
            temp_path = '/tmp/temp_image.jpg'
            cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            compressed = cv2.imread(temp_path)
            os.remove(temp_path)
            
            if compressed is not None:
                ela = cv2.absdiff(image, compressed)
                features['ela_mean'] = np.mean(ela)
                features['ela_std'] = np.std(ela)
                features['ela_max'] = np.max(ela)
                
                # Center region ELA
                h, w = ela.shape[:2]
                center_ela = ela[h//4:3*h//4, w//4:3*w//4]
                features['ela_center_mean'] = np.mean(center_ela)
                features['ela_center_std'] = np.std(center_ela)
            else:
                features.update({f'ela_{prop}': 0 for prop in ['mean', 'std', 'max', 'center_mean', 'center_std']})
        except:
            features.update({f'ela_{prop}': 0 for prop in ['mean', 'std', 'max', 'center_mean', 'center_std']})
        
        return features
    
    def _extract_noise_features_tampered(self, image):
        """Extract noise analysis features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Basic noise statistics
        noise = cv2.Laplacian(gray, cv2.CV_64F)
        features['noise_mean'] = np.mean(noise)
        features['noise_std'] = np.std(noise)
        features['noise_variance'] = np.var(noise)
        
        # High frequency noise
        f_transform = fft2(gray)
        high_freq = f_transform.copy()
        h, w = high_freq.shape
        high_freq[h//4:3*h//4, w//4:3*w//4] = 0
        high_freq_noise = np.abs(np.fft.ifft2(high_freq))
        features['high_freq_noise_std'] = np.std(high_freq_noise)
        features['high_freq_noise_mean'] = np.mean(high_freq_noise)
        
        return features
    
    def _extract_tampering_features_tampered(self, image):
        """Extract tampering-specific features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Block similarity (copy-move detection)
        block_size = 8
        h, w = gray.shape
        similarities = []
        for i in range(0, h-block_size, block_size):
            for j in range(0, w-block_size, block_size):
                block1 = gray[i:i+block_size, j:j+block_size]
                for ii in range(i+block_size, h-block_size, block_size):
                    for jj in range(0, w-block_size, block_size):
                        block2 = gray[ii:ii+block_size, jj:jj+block_size]
                        similarity = np.corrcoef(block1.flatten(), block2.flatten())[0, 1]
                        if not np.isnan(similarity):
                            similarities.append(similarity)
        
        if similarities:
            features['block_similarity_max'] = np.max(similarities)
            features['block_similarity_mean'] = np.mean(similarities)
            features['block_similarity_std'] = np.std(similarities)
        else:
            features.update({f'block_similarity_{prop}': 0 for prop in ['max', 'mean', 'std']})
        
        # Gradient features
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        features['gradient_mean'] = np.mean(gradient_magnitude)
        features['gradient_std'] = np.std(gradient_magnitude)
        features['gradient_max'] = np.max(gradient_magnitude)
        
        return features
    
    def _extract_advanced_features_tampered(self, image):
        """Extract advanced forensic features"""
        features = {}
        
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Moments
        moments = cv2.moments(gray)
        features['moment_00'] = moments['m00']
        features['moment_01'] = moments['m01']
        features['moment_10'] = moments['m10']
        features['moment_11'] = moments['m11']
        
        # Haralick features (simplified)
        try:
            glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
            features['haralick_contrast'] = graycoprops(glcm, 'contrast')[0, 0]
            features['haralick_dissimilarity'] = graycoprops(glcm, 'dissimilarity')[0, 0]
            features['haralick_homogeneity'] = graycoprops(glcm, 'homogeneity')[0, 0]
            features['haralick_energy'] = graycoprops(glcm, 'energy')[0, 0]
            features['haralick_correlation'] = graycoprops(glcm, 'correlation')[0, 0]
        except:
            features.update({f'haralick_{prop}': 0 for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']})
        
        # Tamura features (simplified)
        features['tamura_coarseness'] = np.std(gray)  # Simplified
        features['tamura_contrast'] = np.std(gray) / np.mean(gray) if np.mean(gray) > 0 else 0
        features['tamura_directionality'] = np.std(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
        
        return features
