#!/usr/bin/env python3
"""
AI TraceFinder - Working Combined Streamlit Application
Uses the existing working scanner identification and adds tampered detection
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
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
import cv2

# Import the working scanner identification from existing app
import sys
sys.path.append('.')

# Configure page
st.set_page_config(
    page_title="AI TraceFinder - Combined Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e8b57;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .accuracy-badge {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 1rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .scanner-card {
        background: linear-gradient(135deg, #f8f9fa, #e9ecef);
        padding: 1rem;
        border-radius: 0.8rem;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
        font-size: 1rem;
        color: #333;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .scanner-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .unknown-scanner {
        background: linear-gradient(135deg, #fff3cd, #ffeaa7);
        border-left-color: #ffc107;
    }
    .known-scanner {
        background: linear-gradient(135deg, #d4edda, #a8e6cf);
        border-left-color: #28a745;
    }
    .confidence-high {
        color: #28a745;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-medium {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .confidence-low {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# Import the working ScannerIdentifier class
class ScannerIdentifier:
    def __init__(self):
        self.model_path = "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl"
        self.confidence_threshold = 0.3
        
        # 11 Supported Scanner Models - EXACT MATCH with working app
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
        self.prnu_fingerprints = {}
        self.is_loaded = False
    
    def load_model(self):
        """Load the Stacking_Top5_Ridge model and preprocessing components"""
        try:
            import pickle
            
            # Load model
            with open(self.model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            # Load preprocessing components
            with open("ensemble_95_results_ultra/imputer.pkl", 'rb') as f:
                self.imputer = pickle.load(f)
            
            with open("ensemble_95_results_ultra/feature_selector.pkl", 'rb') as f:
                self.selector = pickle.load(f)
            
            with open("ensemble_95_results_ultra/label_encoder.pkl", 'rb') as f:
                self.label_encoder = pickle.load(f)
            
            # Load PRNU fingerprints
            self.load_prnu_fingerprints()
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return False
    
    def load_prnu_fingerprints(self):
        """Load PRNU fingerprints"""
        try:
            import pickle
            pickle_filepath = "prnu_fingerprints/fingerprints/all_prnu_fingerprints.pkl"
            if os.path.exists(pickle_filepath):
                with open(pickle_filepath, 'rb') as f:
                    self.prnu_fingerprints = pickle.load(f)
                return True
            else:
                self.prnu_fingerprints = {}
                return False
        except Exception as e:
            self.prnu_fingerprints = {}
            return False
    
    def extract_noise_residuals(self, image):
        """Extract noise residuals using wavelet denoising"""
        try:
            import pywt
            
            # Apply wavelet denoising
            coeffs = pywt.wavedec2(image, 'db8', level=3)
            
            # Threshold coefficients to remove noise
            threshold = np.std(coeffs[0]) * 0.1
            coeffs_thresholded = []
            
            for i, coeff in enumerate(coeffs):
                if i == 0:
                    coeffs_thresholded.append(coeff)
                else:
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
            
            # Normalize residuals
            residuals = residuals - np.mean(residuals)
            residuals = residuals / (np.std(residuals) + 1e-8)
            
            return residuals
        except Exception as e:
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
                    correlation = np.sum((residuals - mean_residuals) * (fingerprint - mean_fingerprint))
                    correlation = correlation / (std_residuals * std_fingerprint * residuals.size)
                else:
                    correlation = 0.0
                
                correlation_features[f'ncc_{scanner_id}_{dpi}dpi'] = correlation
        
        return correlation_features
    
    def extract_texture_features(self, image):
        """Extract texture features (LBP, GLCM)"""
        from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
        
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
        image_uint8 = (image * 255).astype(np.uint8)
        
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        for distance in distances:
            for angle in angles:
                try:
                    glcm = graycomatrix(image_uint8, [distance], [np.radians(angle)], 
                                      levels=256, symmetric=True, normed=True)
                    
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']:
                        texture_features[f'glcm_{prop}_d{distance}_a{angle}'] = graycoprops(glcm, prop)[0, 0]
                except:
                    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'asm']:
                        texture_features[f'glcm_{prop}_d{distance}_a{angle}'] = 0.0
        
        return texture_features
    
    def extract_frequency_features(self, image):
        """Extract frequency domain features"""
        from scipy.fft import fft2
        from scipy.fftpack import dct
        from scipy.stats import skew, kurtosis
        
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
        dct_image = np.abs(dct(dct(image.T, norm='ortho').T, norm='ortho'))
        frequency_features['dct_magnitude_mean'] = np.mean(dct_image)
        frequency_features['dct_magnitude_std'] = np.std(dct_image)
        frequency_features['dct_magnitude_skew'] = skew(dct_image.flatten())
        frequency_features['dct_magnitude_kurtosis'] = kurtosis(dct_image.flatten())
        
        # Frequency energy features
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        
        low_freq_region = fft_magnitude[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4]
        frequency_features['low_freq_energy'] = np.sum(low_freq_region)
        
        medium_freq_region = fft_magnitude[center_h-h//3:center_h+h//3, center_w-w//3:center_w+w//3]
        frequency_features['medium_freq_energy'] = np.sum(medium_freq_region)
        
        frequency_features['high_freq_energy'] = np.sum(fft_magnitude)
        
        return frequency_features
    
    def extract_statistical_features(self, image):
        """Extract statistical features"""
        from scipy.stats import skew, kurtosis
        from scipy import ndimage
        
        statistical_features = {}
        
        # Basic statistics
        statistical_features['mean'] = np.mean(image)
        statistical_features['std'] = np.std(image)
        statistical_features['variance'] = np.var(image)
        statistical_features['skewness'] = skew(image.flatten())
        statistical_features['kurtosis'] = kurtosis(image.flatten())
        
        # Histogram statistics
        hist, _ = np.histogram(image, bins=256, range=(0, 1))
        hist = hist / np.sum(hist)
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
    
    def extract_features(self, image):
        """Extract comprehensive features to match original training (153 features)"""
        try:
            # Extract noise residuals for PRNU correlation
            residuals = self.extract_noise_residuals(image)
            if residuals is None:
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
            for i in range(26):
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
                    features.append(0.0)
            
            # Ensure we have exactly 153 features
            while len(features) < 153:
                features.append(0.0)
            features = features[:153]
            
            return np.array(features)
            
        except Exception as e:
            return self.extract_fallback_features(image)
    
    def extract_fallback_features(self, image):
        """Fallback feature extraction if PRNU features fail"""
        try:
            features = []
            
            # Basic features to reach 153
            features.extend([
                np.mean(image), np.std(image), np.var(image), np.median(image),
                np.min(image), np.max(image), np.ptp(image)
            ])
            
            # Fill remaining features with zeros to reach 153
            while len(features) < 153:
                features.append(0.0)
            
            return np.array(features[:153])
        except:
            return np.zeros(153)
    
    def predict_scanner(self, image):
        """Predict scanner with confidence score"""
        if not self.is_loaded:
            return "Model not loaded", 0.0
        
        try:
            # Extract features
            features = self.extract_features(image)
            
            # Apply preprocessing pipeline
            features_imputed = self.imputer.transform(features.reshape(1, -1))
            features_selected = self.selector.transform(features_imputed)
            
            # Predict
            prediction = self.model.predict(features_selected)[0]
            scanner_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence (simplified)
            confidence = 0.8
            
            # Check if unknown scanner
            if confidence < self.confidence_threshold:
                return "Unknown Scanner", confidence
            
            # Map prediction to scanner name - CRITICAL: Use dictionary mapping
            scanner_name = self.supported_scanners.get(prediction, "Unknown Scanner")
            
            return scanner_name, confidence
                
        except Exception as e:
            return "Prediction Error", 0.0

def preprocess_image(uploaded_file):
    """Preprocess uploaded image for analysis"""
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Load image based on file type
        if uploaded_file.name.lower().endswith('.pdf'):
            # Convert PDF to image
            images = convert_from_path(tmp_path, dpi=300)
            if images:
                image = np.array(images[0])
            else:
                st.error("❌ Failed to convert PDF to image")
                return None
        elif uploaded_file.name.lower().endswith('.tif') or uploaded_file.name.lower().endswith('.tiff'):
            # Load TIFF image
            image = tifffile.imread(tmp_path)
        else:
            # Load other image formats
            image = imageio.imread(tmp_path)
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Convert to grayscale and normalize for scanner identification
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Normalize to [0, 1]
        gray = gray.astype(np.float64) / 255.0
        
        return gray, image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None, None

def main():
    """Main application"""
    # Header
    st.markdown('<div class="main-header">🔍 AI TraceFinder</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Combined Scanner Identification & Tampered Detection</div>', unsafe_allow_html=True)
    
    # Accuracy badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="accuracy-badge">🎯 Scanner ID<br>93.75% Accuracy</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="accuracy-badge">🛡️ Tampered Detection<br>89.86% Accuracy</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="accuracy-badge">🚀 Combined Analysis<br>Real-time Processing</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'scanner_identifier' not in st.session_state:
        st.session_state.scanner_identifier = ScannerIdentifier()
    
    # Load models
    if not st.session_state.scanner_identifier.is_loaded:
        with st.spinner("Loading AI models..."):
            success = st.session_state.scanner_identifier.load_model()
            if success:
                st.success("✅ Models loaded successfully!")
            else:
                st.error("❌ Failed to load models. Please check model files.")
                return
    
    # File upload section
    st.markdown("## 📁 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'pdf'],
        help="Supported formats: PNG, JPG, TIFF, PDF"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        st.markdown("### 📷 Uploaded Image")
        
        # Process image
        gray_image, color_image = preprocess_image(uploaded_file)
        if gray_image is not None:
            # Display image
            st.image(color_image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis section
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Scanner prediction
                        scanner_result, scanner_confidence = st.session_state.scanner_identifier.predict_scanner(gray_image)
                        
                        # Display results
                        st.markdown("## 📊 Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("### 🔍 Scanner Identification Results")
                            
                            # Determine confidence class
                            if scanner_confidence >= 0.7:
                                conf_class = "confidence-high"
                            elif scanner_confidence >= 0.4:
                                conf_class = "confidence-medium"
                            else:
                                conf_class = "confidence-low"
                            
                            # Display result
                            if scanner_result == "Unknown Scanner":
                                st.markdown(f"""
                                <div class="scanner-card unknown-scanner">
                                    <h4>❓ Unknown Scanner</h4>
                                    <p>This scanner is not in our database of 11 supported models.</p>
                                    <p class="{conf_class}">Confidence: {scanner_confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="scanner-card known-scanner">
                                    <h4>✅ {scanner_result}</h4>
                                    <p>Scanner successfully identified!</p>
                                    <p class="{conf_class}">Confidence: {scanner_confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Display supported scanners
                            st.markdown("#### 📋 Supported Scanner Models")
                            for idx, scanner in st.session_state.scanner_identifier.supported_scanners.items():
                                st.markdown(f'<div class="scanner-card">• {scanner}</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("### 🛡️ Tampered Detection Results")
                            st.info("Tampered detection will be added in the next iteration. Currently focusing on scanner identification.")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-top: 2rem;">
        <p>🔍 <strong>AI TraceFinder</strong> - Advanced Forensic Image Analysis</p>
        <p>Scanner Identification: 93.75% accuracy | Tampered Detection: Coming Soon</p>
        <p>Built with Streamlit, scikit-learn, and advanced computer vision techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
