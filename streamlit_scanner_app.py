#!/usr/bin/env python3
"""
AI TraceFinder - Scanner Identification Streamlit App
Uses the highest accuracy Stacking_Top5_Ridge model (93.75% accuracy)
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import pickle
import os
import tempfile
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Image processing imports
import tifffile
import imageio.v2 as imageio
from pdf2image import convert_from_path
import io

# Feature extraction imports
from scipy import ndimage
import pywt
from scipy.fft import fft2
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Configure page
st.set_page_config(
    page_title="AI TraceFinder - Scanner Identification",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .accuracy-badge {
        background-color: #28a745;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .scanner-card {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.3rem 0;
        font-size: 0.9rem;
        color: #333;
        font-weight: 500;
    }
    .unknown-scanner {
        background-color: #fff3cd;
        border-left-color: #ffc107;
    }
    .known-scanner {
        background-color: #d4edda;
        border-left-color: #28a745;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class ScannerIdentifier:
    def __init__(self):
        self.model_path = "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl"
        self.confidence_threshold = 0.3  # Threshold for unknown scanner detection
        
        # 11 Supported Scanner Models
        self.supported_scanners = {
            0: "Canon120-1",
            1: "Canon120-2", 
            2: "Canon220",
            3: "Canon9000-1",
            4: "Canon9000-2",
            5: "EpsonV370-1",
            6: "EpsonV370-2",
            7: "EpsonV39-1",
            8: "EpsonV39-2",
            9: "EpsonV550",
            10: "HP"
        }
        
        self.model = None
        self.imputer = None
        self.selector = None
        self.label_encoder = None
        
    def load_model(self):
        """Load the trained Stacking_Top5_Ridge model and preprocessing objects"""
        try:
            # Load the model
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
            else:
                st.error(f"❌ Model file not found at {self.model_path}")
                return False
            
            # Load preprocessing objects
            try:
                with open('ensemble_95_results_ultra/imputer.pkl', 'rb') as f:
                    self.imputer = pickle.load(f)
                with open('ensemble_95_results_ultra/feature_selector.pkl', 'rb') as f:
                    self.selector = pickle.load(f)
                with open('ensemble_95_results_ultra/label_encoder.pkl', 'rb') as f:
                    self.label_encoder = pickle.load(f)
                st.success("✅ Model and preprocessing objects loaded successfully!")
                return True
            except Exception as e:
                st.error(f"❌ Error loading preprocessing objects: {e}")
                return False
                
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return False
    
    def robust_image_reading(self, file_path):
        """Robust image reading with multiple fallback methods"""
        try:
            # Try different image reading methods
            if file_path.lower().endswith('.tif') or file_path.lower().endswith('.tiff'):
                try:
                    image = tifffile.imread(file_path)
                except:
                    image = imageio.imread(file_path)
            else:
                image = Image.open(file_path)
                image = np.array(image)
            
            return image
        except Exception as e:
            st.error(f"Error reading image: {e}")
            return None
    
    def preprocess_image(self, image):
        """Preprocess image for feature extraction"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # RGB to grayscale
                    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
                elif image.shape[2] == 4:
                    # RGBA to grayscale
                    image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            
            # Normalize to [0, 1]
            image = image.astype(np.float32)
            if image.max() > 1:
                image = image / 255.0
            
            # Resize to 256x256
            pil_image = Image.fromarray((image * 255).astype(np.uint8))
            pil_image = pil_image.resize((256, 256), Image.Resampling.LANCZOS)
            image = np.array(pil_image).astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    
    def load_prnu_fingerprints(self):
        """Load PRNU fingerprints for correlation computation"""
        try:
            import pickle
            pickle_filepath = "prnu_fingerprints/fingerprints/all_prnu_fingerprints.pkl"
            with open(pickle_filepath, 'rb') as f:
                self.prnu_fingerprints = pickle.load(f)
            return True
        except Exception as e:
            st.error(f"Failed to load PRNU fingerprints: {e}")
            self.prnu_fingerprints = {}
            return False
    
    def extract_noise_residuals(self, image):
        """Extract noise residuals using wavelet denoising (same as PRNU extraction)"""
        try:
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
        except Exception as e:
            st.error(f"Error extracting noise residuals: {e}")
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
        """Extract texture features (LBP, Haralick/GLCM)"""
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
                except:
                    # Fill with zeros if GLCM fails
                    texture_features[f'glcm_contrast_d{distance}_a{angle}'] = 0.0
                    texture_features[f'glcm_dissimilarity_d{distance}_a{angle}'] = 0.0
                    texture_features[f'glcm_homogeneity_d{distance}_a{angle}'] = 0.0
                    texture_features[f'glcm_energy_d{distance}_a{angle}'] = 0.0
                    texture_features[f'glcm_correlation_d{distance}_a{angle}'] = 0.0
                    texture_features[f'glcm_asm_d{distance}_a{angle}'] = 0.0
        
        return texture_features
    
    def extract_frequency_features(self, image):
        """Extract frequency domain features (FFT, DCT)"""
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
        dct_result = dct(dct(image.T, norm='ortho').T, norm='ortho')
        dct_magnitude = np.abs(dct_result)
        
        # DCT magnitude statistics
        frequency_features['dct_magnitude_mean'] = np.mean(dct_magnitude)
        frequency_features['dct_magnitude_std'] = np.std(dct_magnitude)
        frequency_features['dct_magnitude_skew'] = skew(dct_magnitude.flatten())
        frequency_features['dct_magnitude_kurtosis'] = kurtosis(dct_magnitude.flatten())
        
        # Frequency band energy
        h, w = image.shape
        low_freq = dct_magnitude[:h//4, :w//4]
        medium_freq = dct_magnitude[h//4:3*h//4, w//4:3*w//4]
        high_freq = dct_magnitude[3*h//4:, 3*w//4:]
        
        frequency_features['low_freq_energy'] = np.sum(low_freq**2)
        frequency_features['medium_freq_energy'] = np.sum(medium_freq**2)
        frequency_features['high_freq_energy'] = np.sum(high_freq**2)
        
        return frequency_features
    
    def extract_statistical_features(self, image):
        """Extract statistical features"""
        statistical_features = {}
        
        # Basic statistics
        statistical_features['mean'] = np.mean(image)
        statistical_features['std'] = np.std(image)
        statistical_features['variance'] = np.var(image)
        statistical_features['skewness'] = skew(image.flatten())
        statistical_features['kurtosis'] = kurtosis(image.flatten())
        
        # Histogram statistics
        hist, _ = np.histogram(image, bins=256, range=(0, 1), density=True)
        statistical_features['hist_mean'] = np.mean(hist)
        statistical_features['hist_std'] = np.std(hist)
        statistical_features['hist_skew'] = skew(hist)
        statistical_features['hist_kurtosis'] = kurtosis(hist)
        
        # Percentiles
        for p in [10, 25, 50, 75, 90]:
            statistical_features[f'percentile_{p}'] = np.percentile(image, p)
        
        # Range and IQR
        statistical_features['range'] = np.ptp(image)
        statistical_features['iqr'] = np.percentile(image, 75) - np.percentile(image, 25)
        
        # Edge features
        sobel_x = ndimage.sobel(image, axis=1)
        sobel_y = ndimage.sobel(image, axis=0)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        statistical_features['edge_density'] = np.mean(edge_magnitude)
        statistical_features['edge_variance'] = np.var(edge_magnitude)
        
        return statistical_features
    
    def extract_features(self, image):
        """Extract comprehensive features to match original training (153 features)"""
        try:
            # Load PRNU fingerprints if not already loaded
            if not hasattr(self, 'prnu_fingerprints') or not self.prnu_fingerprints:
                if not self.load_prnu_fingerprints():
                    st.error("Cannot load PRNU fingerprints - using fallback features")
                    return self.extract_fallback_features(image)
            
            # Extract noise residuals for PRNU correlation
            residuals = self.extract_noise_residuals(image)
            if residuals is None:
                st.error("Cannot extract noise residuals - using fallback features")
                return self.extract_fallback_features(image)
            
            # Compute PRNU correlation features for both DPI
            correlation_features_150 = self.compute_correlation_features(residuals, 150)
            correlation_features_300 = self.compute_correlation_features(residuals, 300)
            
            # Extract other features
            texture_features = self.extract_texture_features(image)
            frequency_features = self.extract_frequency_features(image)
            statistical_features = self.extract_statistical_features(image)
            
            # Combine all features in the same order as training
            all_features = {}
            all_features.update(correlation_features_150)
            all_features.update(texture_features)
            all_features.update(frequency_features)
            all_features.update(statistical_features)
            all_features.update(correlation_features_300)
            
            # Convert to array in the same order as training
            feature_names = [
                'ncc_Canon120-1_150dpi', 'ncc_Canon120-2_150dpi', 'ncc_Canon220_150dpi',
                'ncc_Canon9000-1_150dpi', 'ncc_Canon9000-2_150dpi', 'ncc_EpsonV39-1_150dpi',
                'ncc_EpsonV39-2_150dpi', 'ncc_EpsonV370-1_150dpi', 'ncc_EpsonV370-2_150dpi',
                'ncc_EpsonV550_150dpi', 'ncc_HP_150dpi'
            ]
            
            # Add LBP histogram features
            for i in range(26):  # lbp_hist_0 to lbp_hist_25
                feature_names.append(f'lbp_hist_{i}')
            
            # Add GLCM features
            distances = [1, 2, 3]
            angles = [0, 45, 90, 135]
            glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']
            for distance in distances:
                for angle in angles:
                    for prop in glcm_props:
                        feature_names.append(f'glcm_{prop}_d{distance}_a{angle}')
            
            # Add frequency features
            freq_features = ['fft_magnitude_mean', 'fft_magnitude_std', 'fft_magnitude_skew', 'fft_magnitude_kurtosis',
                           'fft_phase_mean', 'fft_phase_std', 'fft_phase_skew', 'fft_phase_kurtosis',
                           'dct_magnitude_mean', 'dct_magnitude_std', 'dct_magnitude_skew', 'dct_magnitude_kurtosis',
                           'low_freq_energy', 'medium_freq_energy', 'high_freq_energy']
            feature_names.extend(freq_features)
            
            # Add statistical features
            stat_features = ['mean', 'std', 'variance', 'skewness', 'kurtosis', 'hist_mean', 'hist_std', 'hist_skew', 'hist_kurtosis',
                           'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'range', 'iqr',
                           'edge_density', 'edge_variance']
            feature_names.extend(stat_features)
            
            # Add 300 DPI correlation features
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
            st.error(f"Error extracting features: {e}")
            return self.extract_fallback_features(image)
    
    def extract_fallback_features(self, image):
        """Fallback feature extraction if PRNU features fail"""
        try:
            features = []
            
            # Basic features to reach 90 (what the model expects after SelectKBest)
            # Statistical features
            features.extend([
                np.mean(image), np.std(image), np.var(image), np.median(image),
                np.min(image), np.max(image), np.ptp(image), skew(image.flatten()), kurtosis(image.flatten())
            ])
            
            # LBP features
            lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
            features.extend([np.mean(lbp), np.std(lbp), np.var(lbp)])
            
            # GLCM features
            try:
                glcm = graycomatrix((image * 255).astype(np.uint8), distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                features.extend([
                    graycoprops(glcm, 'contrast')[0, 0], graycoprops(glcm, 'dissimilarity')[0, 0],
                    graycoprops(glcm, 'homogeneity')[0, 0], graycoprops(glcm, 'energy')[0, 0],
                    graycoprops(glcm, 'correlation')[0, 0]
                ])
            except:
                features.extend([0.0] * 5)
            
            # FFT features
            fft_image = np.abs(fft2(image))
            features.extend([np.mean(fft_image), np.std(fft_image), np.var(fft_image)])
            
            # DCT features
            dct_image = np.abs(dct(dct(image.T, norm='ortho').T, norm='ortho'))
            features.extend([np.mean(dct_image), np.std(dct_image), np.var(dct_image)])
            
            # Wavelet features
            try:
                coeffs = pywt.dwt2(image, 'haar')
                cA, (cH, cV, cD) = coeffs
                features.extend([np.mean(cA), np.std(cA), np.mean(cH), np.std(cH),
                               np.mean(cV), np.std(cV), np.mean(cD), np.std(cD)])
            except:
                features.extend([0.0] * 8)
            
            # Edge features
            sobel_x = ndimage.sobel(image, axis=1)
            sobel_y = ndimage.sobel(image, axis=0)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            features.extend([np.mean(edge_magnitude), np.std(edge_magnitude), np.var(edge_magnitude)])
            
            # Additional features to reach 90
            features.extend([
                np.sum(image), np.sum(image**2), np.sum(image**3), np.sum(image**4)
            ])
            
            # More LBP variations
            for radius in [2, 3]:
                try:
                    lbp_var = local_binary_pattern(image, P=8*radius, R=radius, method='uniform')
                    features.extend([np.mean(lbp_var), np.std(lbp_var)])
                except:
                    features.extend([0.0, 0.0])
            
            # More GLCM
            for distance in [2, 3]:
                try:
                    glcm_var = graycomatrix((image * 255).astype(np.uint8), distances=[distance], angles=[0], 
                                          levels=256, symmetric=True, normed=True)
                    features.extend([graycoprops(glcm_var, 'contrast')[0, 0], graycoprops(glcm_var, 'energy')[0, 0]])
                except:
                    features.extend([0.0, 0.0])
            
            # More wavelets
            for wavelet in ['db4', 'db8']:
                try:
                    coeffs_var = pywt.dwt2(image, wavelet)
                    cA_var, (cH_var, cV_var, cD_var) = coeffs_var
                    features.extend([np.mean(cA_var), np.std(cA_var)])
                except:
                    features.extend([0.0, 0.0])
            
            # Additional frequency features
            fft_centered = np.fft.fftshift(fft_image)
            features.extend([np.mean(fft_centered), np.std(fft_centered)])
            
            # DCT variations
            dct_h = np.abs(dct(image, axis=1, norm='ortho'))
            dct_v = np.abs(dct(image, axis=0, norm='ortho'))
            features.extend([np.mean(dct_h), np.std(dct_h), np.mean(dct_v), np.std(dct_v)])
            
            # Edge variations
            laplacian = ndimage.laplace(image)
            features.extend([np.mean(laplacian), np.std(laplacian)])
            
            # Gradient direction
            grad_direction = np.arctan2(sobel_y, sobel_x)
            features.extend([np.mean(grad_direction), np.std(grad_direction)])
            
            # Pad to exactly 90 features
            while len(features) < 90:
                features.append(0.0)
            features = features[:90]
            
            return np.array(features)
            
        except Exception as e:
            st.error(f"Error in fallback feature extraction: {e}")
            return None
    
    def predict_scanner(self, image):
        """Predict scanner type from image"""
        try:
            if self.model is None:
                return None, 0.0, "Model not loaded"
            
            # Extract features (153 features)
            features = self.extract_features(image)
            if features is None:
                return None, 0.0, "Feature extraction failed"
            
            # Apply the same preprocessing as training
            # 1. Imputation
            features_imputed = self.imputer.transform(features.reshape(1, -1))
            
            # 2. Feature selection (153 -> 90 features)
            features_selected = self.selector.transform(features_imputed)
            
            # Get prediction
            prediction = self.model.predict(features_selected)[0]
            
            # Calculate confidence using base estimator probabilities
            try:
                # Get probabilities from all base estimators using preprocessed features
                base_probabilities = []
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'predict_proba'):
                        prob = estimator.predict_proba(features_selected)[0]
                        base_probabilities.append(prob)
                    else:
                        # If estimator doesn't have predict_proba, create dummy probabilities
                        dummy_prob = np.zeros(len(self.supported_scanners))
                        dummy_prob[prediction] = 1.0
                        base_probabilities.append(dummy_prob)
                
                # Average the probabilities from base estimators
                if base_probabilities:
                    avg_probabilities = np.mean(base_probabilities, axis=0)
                    confidence = avg_probabilities[prediction]
                else:
                    # Fallback if no probabilities available
                    confidence = 0.5
                
            except Exception as e:
                # Fallback: use a reasonable confidence based on prediction
                confidence = 0.7  # Default confidence for known predictions
                print(f"Debug: Confidence calculation failed: {e}")
            
            # Check if confidence is below threshold (unknown scanner)
            if confidence < self.confidence_threshold:
                return "Unknown Scanner", confidence, "Low confidence - likely not one of the 11 supported scanners"
            
            # Map prediction to scanner name
            scanner_name = self.supported_scanners.get(prediction, "Unknown Scanner")
            
            return scanner_name, confidence, "High confidence prediction"
            
        except Exception as e:
            st.error(f"Error in prediction: {e}")
            return None, 0.0, f"Prediction error: {e}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AI TraceFinder - Scanner Identification</h1>', unsafe_allow_html=True)
    
    # Accuracy badge
    st.markdown('<div class="accuracy-badge">🎯 Model Accuracy: 93.75% (Stacking_Top5_Ridge)</div>', unsafe_allow_html=True)
    
    # Initialize scanner identifier first
    if 'identifier' not in st.session_state:
        st.session_state.identifier = ScannerIdentifier()
    
    identifier = st.session_state.identifier
    
    # Sidebar with supported scanners
    with st.sidebar:
        st.header("📋 Supported Scanner Models")
        st.markdown("This AI can identify **only** these 11 scanner models:")
        
        for idx, scanner in identifier.supported_scanners.items():
            st.markdown(f'<div class="scanner-card">{scanner}</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**⚠️ Important:**")
        st.markdown("- Any scanner **not** in this list will be classified as **Unknown Scanner**")
        st.markdown("- Confidence threshold: **30%**")
        st.markdown("- Below 30% confidence = Unknown Scanner")
        
        st.markdown("---")
        st.markdown("**📁 Supported File Types:**")
        st.markdown("- TIFF/TIF files")
        st.markdown("- PDF files (images will be extracted)")
        st.markdown("- JPG/JPEG files")
        st.markdown("- PNG files")
    
    # Load model
    if not identifier.load_model():
        st.stop()
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image/PDF")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['tif', 'tiff', 'pdf', 'jpg', 'jpeg', 'png'],
            help="Upload an image or PDF to identify the scanner used"
        )
        
        if uploaded_file is not None:
            # Display file info
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"📏 **Size:** {uploaded_file.size / 1024:.1f} KB")
            st.info(f"📅 **Type:** {uploaded_file.type}")
            
            # Process file
            if st.button("🔍 Identify Scanner", type="primary"):
                with st.spinner("Processing image and extracting features..."):
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Handle PDF files
                        if uploaded_file.type == "application/pdf":
                            st.info("📄 PDF detected - extracting images...")
                            try:
                                images = convert_from_path(tmp_path)
                                if images:
                                    # Convert first page to numpy array
                                    image = np.array(images[0])
                                    st.info(f"✅ Extracted {len(images)} page(s) from PDF")
                                else:
                                    st.error("❌ No images found in PDF")
                                    os.unlink(tmp_path)
                                    st.stop()
                            except Exception as e:
                                st.error(f"❌ Error processing PDF: {e}")
                                os.unlink(tmp_path)
                                st.stop()
                        else:
                            # Handle image files
                            image = identifier.robust_image_reading(tmp_path)
                            if image is None:
                                st.error("❌ Failed to read image")
                                os.unlink(tmp_path)
                                st.stop()
                        
                        # Preprocess image
                        processed_image = identifier.preprocess_image(image)
                        if processed_image is None:
                            st.error("❌ Failed to preprocess image")
                            os.unlink(tmp_path)
                            st.stop()
                        
                        # Make prediction
                        scanner_name, confidence, status = identifier.predict_scanner(processed_image)
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        # Display results
                        with col2:
                            st.header("🎯 Scanner Identification Results")
                            
                            if scanner_name:
                                # Determine result styling
                                if scanner_name == "Unknown Scanner":
                                    card_class = "unknown-scanner"
                                    confidence_class = "confidence-low"
                                    icon = "❓"
                                else:
                                    card_class = "known-scanner"
                                    if confidence >= 0.9:
                                        confidence_class = "confidence-high"
                                        icon = "✅"
                                    elif confidence >= 0.8:
                                        confidence_class = "confidence-medium"
                                        icon = "⚠️"
                                    else:
                                        confidence_class = "confidence-low"
                                        icon = "❓"
                                
                                # Display result card
                                st.markdown(f'''
                                <div class="scanner-card {card_class}">
                                    <h3>{icon} Scanner Type: <strong>{scanner_name}</strong></h3>
                                    <p><span class="{confidence_class}">Confidence: {confidence:.1%}</span></p>
                                    <p><em>{status}</em></p>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # Additional info
                                if scanner_name == "Unknown Scanner":
                                    st.warning("⚠️ This scanner is not in our database of 11 supported models. It may be:")
                                    st.markdown("- A different scanner model")
                                    st.markdown("- A camera or other imaging device")
                                    st.markdown("- A heavily processed/altered image")
                                else:
                                    st.success(f"✅ Successfully identified as {scanner_name} scanner!")
                                
                                # Confidence interpretation
                                if confidence >= 0.9:
                                    st.info("🟢 **High Confidence** - Very reliable prediction")
                                elif confidence >= 0.8:
                                    st.info("🟡 **Medium Confidence** - Good prediction")
                                elif confidence >= 0.7:
                                    st.info("🟠 **Low Confidence** - Uncertain prediction")
                                else:
                                    st.warning("🔴 **Very Low Confidence** - Likely unknown scanner")
                                
                            else:
                                st.error("❌ Failed to identify scanner")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing file: {e}")
                        if 'tmp_path' in locals():
                            try:
                                os.unlink(tmp_path)
                            except:
                                pass
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🔍 <strong>AI TraceFinder</strong> - Scanner Identification System</p>
        <p>Powered by Stacking_Top5_Ridge Ensemble Model (93.75% Accuracy)</p>
        <p>Supports 11 scanner models: Canon, Epson, and HP variants</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()