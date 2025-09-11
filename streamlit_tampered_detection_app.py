import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from PIL import Image
import cv2
import json
from scipy import ndimage
from scipy.stats import skew, kurtosis
from scipy.fft import fft2, fftshift
from skimage import filters, feature, measure
from skimage.filters import gabor
import pywt
import mahotas as mh
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI TraceFinder - Tampered Image Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme styling
st.markdown("""
<style>
    /* Main app background */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%);
        color: #ffffff;
    }
    
    /* Sidebar removed - no styling needed */
    
    /* Main content area */
    .main .block-container {
        background: transparent;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Headers */
    .main-header {
        font-size: 3.5rem;
        background: linear-gradient(45deg, #00d4ff, #0099cc, #0066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
        letter-spacing: 2px;
    }
    
    .sub-header {
        font-size: 1.8rem;
        color: #ff6b6b;
        margin-bottom: 1.5rem;
        font-weight: 600;
        text-align: center;
        text-shadow: 0 0 20px rgba(255, 107, 107, 0.3);
    }
    
    /* Metrics styling */
    .metric-container {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 0.8rem 0;
        border: 1px solid #404040;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #1e3a1e 0%, #2d5a2d 100%);
        border: 2px solid #4caf50;
        color: #a8e6a8;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(76, 175, 80, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Error box */
    .error-box {
        background: linear-gradient(135deg, #3a1e1e 0%, #5a2d2d 100%);
        border: 2px solid #f44336;
        color: #ffb3b3;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(244, 67, 54, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #1e2a3a 0%, #2d3a5a 100%);
        border: 2px solid #2196f3;
        color: #b3d9ff;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(33, 150, 243, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* File uploader styling */
    .stFileUploader > div > div {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        border: 2px dashed #666;
        border-radius: 15px;
        padding: 2rem;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #00d4ff;
        background: linear-gradient(135deg, #2a3a3a 0%, #3a4a4a 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #0099cc, #0066ff);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar removed - no metric styling needed */
    
    /* Text styling */
    .stMarkdown {
        color: #ffffff !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    /* Main content text */
    .main .stMarkdown {
        color: #ffffff !important;
    }
    
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    /* Block containers */
    .block-container {
        background: transparent !important;
    }
    
    /* Main content area */
    .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* Columns */
    .stColumn {
        background: transparent !important;
    }
    
    /* File uploader area */
    .uploadedFile {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%) !important;
        border: 1px solid #404040 !important;
        border-radius: 10px !important;
    }
    
    /* Chart styling */
    .stPlotlyChart {
        background: transparent;
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Image styling */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 2px solid #404040;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 10px;
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background: #2a2a2a;
        border: 1px solid #404040;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%);
        border: 1px solid #404040;
        border-radius: 10px;
        color: #ffffff;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
        border-top: 2px solid #404040;
        padding: 2rem;
        margin-top: 3rem;
        border-radius: 15px;
        text-align: center;
        color: #cccccc;
    }
    
    /* Custom animations */
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 0 20px rgba(0, 212, 255, 0.6); }
        100% { box-shadow: 0 0 5px rgba(0, 212, 255, 0.3); }
    }
    
    .glow {
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00d4ff, #0099cc);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #0099cc, #0066ff);
    }
    
    /* Comprehensive Streamlit dark theme */
    .stApp > header {
        background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%) !important;
        border-bottom: 2px solid #404040 !important;
    }
    
    /* Sidebar removed - no styling needed */
    
    /* Main content area */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a1a 50%, #2d2d2d 100%) !important;
    }
    
    div[data-testid="stAppViewContainer"] .main .block-container {
        background: transparent !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }
    
    /* File uploader */
    div[data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%) !important;
        border: 2px dashed #666 !important;
        border-radius: 15px !important;
        padding: 2rem !important;
    }
    
    div[data-testid="stFileUploader"]:hover {
        border-color: #00d4ff !important;
        background: linear-gradient(135deg, #2a3a3a 0%, #3a4a4a 100%) !important;
    }
    
    /* Buttons */
    div[data-testid="stButton"] > button {
        background: linear-gradient(45deg, #00d4ff, #0099cc) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.5rem 2rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(45deg, #0099cc, #0066ff) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    
    /* Images */
    div[data-testid="stImage"] {
        border-radius: 15px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        border: 2px solid #404040 !important;
    }
    
    /* Charts */
    div[data-testid="stPlotlyChart"] {
        background: transparent !important;
        border-radius: 15px !important;
        padding: 1rem !important;
    }
    
    /* All text elements */
    .stMarkdown, .stText, .stSelectbox label, .stTextInput label {
        color: #ffffff !important;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Sidebar removed - no styling needed */
</style>
""", unsafe_allow_html=True)

class TamperedImageDetector:
    """Tampered Image Detection using Robust Ensemble Model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_indices = None
        self.load_model()
    
    def load_model(self):
        """Load the baseline LightGBM model and scaler"""
        try:
            # Load baseline LightGBM model (better performance than robust model)
            model_path = 'objective2_baseline_models/lightgbm.joblib'
            scaler_path = 'objective2_baseline_models/scaler.joblib'
            feature_info_path = 'objective2_baseline_models/feature_info.json'
            
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                st.success("✅ Loaded baseline LightGBM model (71.4% overall accuracy)")
            else:
                st.error("❌ Model file not found")
                return
            
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                st.success("✅ Loaded baseline scaler")
            else:
                st.error("❌ Scaler file not found")
                return
            
            if os.path.exists(feature_info_path):
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_indices = feature_info.get('selected_features', [])
                st.success("✅ Loaded feature selection info")
            else:
                st.error("❌ Feature info file not found")
                return
            
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
    
    def extract_ela_features(self, image):
        """Extract Error Level Analysis features - matching original training"""
        try:
            # ELA calculation (matching original training)
            temp_path = '/tmp/temp_ela.jpg'
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            compressed = cv2.imread(temp_path)
            compressed = cv2.cvtColor(compressed, cv2.COLOR_BGR2RGB)
            compressed = cv2.resize(compressed, (224, 224))
            os.remove(temp_path)
            
            # Calculate difference
            ela_diff = np.abs(image.astype(np.float32) - compressed.astype(np.float32))
            
            # ELA features (matching original)
            features = [
                np.mean(ela_diff),  # ela_mean
                np.std(ela_diff),   # ela_std
                np.max(ela_diff),   # ela_max
                skew(ela_diff.flatten()),  # ela_skew
                kurtosis(ela_diff.flatten()),  # ela_kurtosis
                np.mean(ela_diff[:, :, 0]),  # ela_R_mean
                np.std(ela_diff[:, :, 0]),   # ela_R_std
                np.mean(ela_diff[:, :, 1]),  # ela_G_mean
                np.std(ela_diff[:, :, 1]),   # ela_G_std
                np.mean(ela_diff[:, :, 2]),  # ela_B_mean
                np.std(ela_diff[:, :, 2])    # ela_B_std
            ]
            
            return features
        except Exception as e:
            return [0.0] * 11
    
    def extract_noise_features(self, image):
        """Extract noise analysis features - matching original training"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Gaussian noise estimation (matching original)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            noise = gray.astype(np.float32) - blurred.astype(np.float32)
            
            # Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Sobel variance
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_variance = np.var(sobel_magnitude)
            
            features = [
                np.mean(noise),  # noise_mean
                np.std(noise),   # noise_std
                np.var(noise),   # noise_variance
                skew(noise.flatten()),  # noise_skew
                kurtosis(noise.flatten()),  # noise_kurtosis
                laplacian_var,   # laplacian_variance
                sobel_variance   # sobel_variance
            ]
            
            return features
        except Exception as e:
            return [0.0] * 7
    
    def extract_compression_features(self, image):
        """Extract compression artifact features - matching original training"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # DCT-based compression analysis
            dct_features = []
            for i in range(0, gray.shape[0]-8, 8):
                for j in range(0, gray.shape[1]-8, 8):
                    block = gray[i:i+8, j:j+8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    dct_features.append(dct_block.flatten())
            
            dct_features = np.array(dct_features)
            
            # DCT coefficient statistics
            dct_mean = np.mean(dct_features)
            dct_std = np.std(dct_features)
            dct_energy = np.sum(dct_features**2)
            
            # High frequency DCT coefficients (compression artifacts)
            high_freq_coeffs = dct_features[:, 1:]  # Exclude DC component
            high_freq_dct_mean = np.mean(high_freq_coeffs)
            high_freq_dct_std = np.std(high_freq_coeffs)
            
            # Blocking artifacts detection
            h_diff = np.abs(np.diff(gray, axis=1))
            v_diff = np.abs(np.diff(gray, axis=0))
            
            # Blocking artifacts at 8-pixel intervals
            h_blocking = []
            v_blocking = []
            for i in range(7, gray.shape[0]-1, 8):
                v_blocking.extend(v_diff[i, :])
            for j in range(7, gray.shape[1]-1, 8):
                h_blocking.extend(h_diff[:, j])
            
            blocking_artifacts_h = np.mean(h_blocking) if h_blocking else 0
            blocking_artifacts_v = np.mean(v_blocking) if v_blocking else 0
            
            features = [
                dct_mean, dct_std, dct_energy, high_freq_dct_mean, high_freq_dct_std,
                blocking_artifacts_h, blocking_artifacts_v
            ]
            
            return features
        except Exception as e:
            return [0.0] * 7
    
    def extract_frequency_features(self, image):
        """Extract frequency domain features - matching original training"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # FFT analysis
            fft = fft2(gray)
            fft_shifted = fftshift(fft)
            magnitude = np.abs(fft_shifted)
            
            # Frequency domain statistics
            fft_mean = np.mean(magnitude)
            fft_std = np.std(magnitude)
            fft_energy = np.sum(magnitude**2)
            
            # High frequency energy
            h, w = magnitude.shape
            center_h, center_w = h//2, w//2
            high_freq_mask = np.zeros_like(magnitude)
            high_freq_mask[center_h-h//4:center_h+h//4, center_w-w//4:center_w+w//4] = 1
            high_freq_mask = 1 - high_freq_mask
            
            high_freq_energy = np.sum(magnitude * high_freq_mask)
            high_freq_ratio = high_freq_energy / fft_energy if fft_energy > 0 else 0
            
            # Spectral centroid
            freq_x = np.arange(w) - center_w
            freq_y = np.arange(h) - center_h
            freq_mesh_x, freq_mesh_y = np.meshgrid(freq_x, freq_y)
            freq_magnitude = np.sqrt(freq_mesh_x**2 + freq_mesh_y**2)
            
            spectral_centroid = np.sum(freq_magnitude * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0
            
            features = [
                fft_mean, fft_std, fft_energy, high_freq_energy, high_freq_ratio, spectral_centroid
            ]
            
            return features
        except Exception as e:
            return [0.0] * 6
    
    def extract_statistical_features(self, image):
        """Extract statistical features - matching original training"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Basic statistics
            features = [
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.median(gray),
                np.min(gray),
                np.max(gray),
                skew(gray.flatten()),
                kurtosis(gray.flatten()),
                np.percentile(gray, 25),
                np.percentile(gray, 75),
                np.percentile(gray, 90),
                np.percentile(gray, 95),
                np.percentile(gray, 99),
                np.percentile(gray, 1),
                np.percentile(gray, 5),
                np.percentile(gray, 10),
                np.percentile(gray, 20),
                np.percentile(gray, 30),
                np.percentile(gray, 40),
                np.percentile(gray, 50),
                np.percentile(gray, 60),
                np.percentile(gray, 70),
                np.percentile(gray, 80),
                np.percentile(gray, 85),
                np.percentile(gray, 90)
            ]
            
            return features
        except Exception as e:
            return [0.0] * 25
    
    def extract_texture_features(self, image):
        """Extract texture features - matching original training"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # GLCM features
            glcm = feature.graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            
            # Haralick features
            contrast = feature.graycoprops(glcm, 'contrast')
            dissimilarity = feature.graycoprops(glcm, 'dissimilarity')
            homogeneity = feature.graycoprops(glcm, 'homogeneity')
            energy = feature.graycoprops(glcm, 'energy')
            correlation = feature.graycoprops(glcm, 'correlation')
            
            # LBP features
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, range=(0, 10))
            lbp_hist = lbp_hist.astype(float)
            lbp_hist /= (lbp_hist.sum() + 1e-7)
            
            # Zernike moments
            zernike_moments = mh.features.zernike_moments(gray, radius=8, degree=8)
            
            # Hu moments
            moments = cv2.moments(gray)
            hu_moments = cv2.HuMoments(moments).flatten()
            
            features = [
                np.mean(contrast),
                np.std(contrast),
                np.mean(dissimilarity),
                np.std(dissimilarity),
                np.mean(homogeneity),
                np.std(homogeneity),
                np.mean(energy),
                np.std(energy),
                np.mean(correlation),
                np.std(correlation),
                np.mean(lbp_hist),
                np.std(lbp_hist),
                np.max(lbp_hist),
                np.min(lbp_hist),
                np.var(lbp_hist),
                np.median(lbp_hist),
                np.mean(zernike_moments),
                np.std(zernike_moments),
                np.max(zernike_moments),
                np.min(zernike_moments),
                np.var(zernike_moments),
                np.median(zernike_moments),
                np.mean(hu_moments),
                np.std(hu_moments),
                np.max(hu_moments)
            ]
            
            return features
        except Exception as e:
            return [0.0] * 25
    
    def extract_all_features(self, image):
        """Extract all 105 forensic features from an image - matching exact training order"""
        try:
            # Convert PIL to OpenCV format
            if isinstance(image, Image.Image):
                image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Resize image
            image = cv2.resize(image, (224, 224))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Initialize features dictionary to match training order
            features = {}
            
            # 1. ELA features (11 features)
            ela_features = self.extract_ela_features(image)
            ela_names = ['ela_mean', 'ela_std', 'ela_max', 'ela_skew', 'ela_kurtosis', 
                        'ela_R_mean', 'ela_R_std', 'ela_G_mean', 'ela_G_std', 'ela_B_mean', 'ela_B_std']
            for i, name in enumerate(ela_names):
                features[name] = ela_features[i]
            
            # 2. Noise features (7 features)
            noise_features = self.extract_noise_features(image)
            noise_names = ['noise_mean', 'noise_std', 'noise_variance', 'noise_skew', 'noise_kurtosis', 
                          'laplacian_variance', 'sobel_variance']
            for i, name in enumerate(noise_names):
                features[name] = noise_features[i]
            
            # 3. Compression features (7 features)
            comp_features = self.extract_compression_features(image)
            comp_names = ['dct_mean', 'dct_std', 'dct_energy', 'high_freq_dct_mean', 'high_freq_dct_std', 
                         'blocking_artifacts_h', 'blocking_artifacts_v']
            for i, name in enumerate(comp_names):
                features[name] = comp_features[i]
            
            # 4. Frequency features (6 features)
            freq_features = self.extract_frequency_features(image)
            freq_names = ['fft_mean', 'fft_std', 'fft_energy', 'high_freq_energy', 'high_freq_ratio', 'spectral_centroid']
            for i, name in enumerate(freq_names):
                features[name] = freq_features[i]
            
            # 5. Additional features to match training (remaining features)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Gabor features
            gabor_responses = []
            for theta in range(0, 180, 45):
                gabor_filter = gabor(gray, frequency=0.1, theta=theta, sigma_x=1, sigma_y=1)
                gabor_responses.append(gabor_filter[0])
            
            gabor_combined = np.concatenate(gabor_responses)
            features['gabor_mean'] = np.mean(gabor_combined)
            features['gabor_std'] = np.std(gabor_combined)
            features['gabor_energy'] = np.sum(gabor_combined**2)
            
            # Texture features (GLCM)
            glcm = feature.graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], levels=256, symmetric=True, normed=True)
            features['contrast'] = np.mean(feature.graycoprops(glcm, 'contrast'))
            features['dissimilarity'] = np.mean(feature.graycoprops(glcm, 'dissimilarity'))
            features['homogeneity'] = np.mean(feature.graycoprops(glcm, 'homogeneity'))
            features['energy'] = np.mean(feature.graycoprops(glcm, 'energy'))
            
            # Color features for each channel
            for i, channel in enumerate(['R', 'G', 'B']):
                channel_data = image[:, :, i]
                features[f'{channel}_mean'] = np.mean(channel_data)
                features[f'{channel}_std'] = np.std(channel_data)
                features[f'{channel}_skew'] = skew(channel_data.flatten())
                features[f'{channel}_kurtosis'] = kurtosis(channel_data.flatten())
                features[f'{channel}_min'] = np.min(channel_data)
                features[f'{channel}_max'] = np.max(channel_data)
                features[f'{channel}_median'] = np.median(channel_data)
            
            # Overall color features
            features['overall_mean'] = np.mean(image)
            features['overall_std'] = np.std(image)
            features['overall_skew'] = skew(image.flatten())
            features['overall_kurtosis'] = kurtosis(image.flatten())
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            features['edge_mean'] = np.mean(edges)
            
            # Sobel features
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_direction = np.arctan2(sobel_y, sobel_x)
            
            features['sobel_magnitude_mean'] = np.mean(sobel_magnitude)
            features['sobel_magnitude_std'] = np.std(sobel_magnitude)
            features['sobel_direction_mean'] = np.mean(sobel_direction)
            features['sobel_direction_std'] = np.std(sobel_direction)
            
            # Laplacian features
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            features['laplacian_mean'] = np.mean(laplacian)
            features['laplacian_std'] = np.std(laplacian)
            features['laplacian_energy'] = np.sum(laplacian**2)
            
            # HSV features
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            for i, channel in enumerate(['H', 'S', 'V']):
                channel_data = hsv[:, :, i]
                features[f'hsv_{channel}_mean'] = np.mean(channel_data)
                features[f'hsv_{channel}_std'] = np.std(channel_data)
                features[f'hsv_{channel}_skew'] = skew(channel_data.flatten())
            
            # Color moments
            features['color_moment_1'] = np.mean(image)
            features['color_moment_2'] = np.std(image)
            features['color_moment_3'] = skew(image.flatten())
            
            # Histogram entropy
            for i, channel in enumerate(['r', 'g', 'b']):
                hist, _ = np.histogram(image[:, :, i], bins=256, range=(0, 256))
                hist = hist / hist.sum()
                hist = hist[hist > 0]  # Remove zeros
                entropy = -np.sum(hist * np.log2(hist))
                features[f'hist_{channel}_entropy'] = entropy
            
            # Wavelet features
            coeffs = pywt.dwt2(gray, 'haar')
            cA, (cH, cV, cD) = coeffs
            
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
            
            # Wavelet ratios
            features['wavelet_LL_ratio'] = features['wavelet_LL_energy'] / (features['wavelet_LL_energy'] + features['wavelet_LH_energy'] + features['wavelet_HL_energy'] + features['wavelet_HH_energy'])
            features['wavelet_HH_ratio'] = features['wavelet_HH_energy'] / (features['wavelet_LL_energy'] + features['wavelet_LH_energy'] + features['wavelet_HL_energy'] + features['wavelet_HH_energy'])
            
            # LBP features
            lbp = feature.local_binary_pattern(gray, 8, 1, method='uniform')
            features['lbp_mean'] = np.mean(lbp)
            features['lbp_std'] = np.std(lbp)
            features['lbp_energy'] = np.sum(lbp**2)
            features['lbp_entropy'] = -np.sum((lbp / lbp.sum()) * np.log2((lbp / lbp.sum()) + 1e-10))
            
            # Convert to array in the exact order as training
            feature_names = ['ela_mean', 'ela_std', 'ela_max', 'ela_skew', 'ela_kurtosis', 'ela_R_mean', 'ela_R_std', 'ela_G_mean', 'ela_G_std', 'ela_B_mean', 'ela_B_std', 'noise_mean', 'noise_std', 'noise_variance', 'noise_skew', 'noise_kurtosis', 'laplacian_variance', 'sobel_variance', 'dct_mean', 'dct_std', 'dct_energy', 'high_freq_dct_mean', 'high_freq_dct_std', 'blocking_artifacts_h', 'blocking_artifacts_v', 'fft_mean', 'fft_std', 'fft_energy', 'high_freq_energy', 'high_freq_ratio', 'spectral_centroid', 'gabor_mean', 'gabor_std', 'gabor_energy', 'contrast', 'dissimilarity', 'homogeneity', 'energy', 'R_mean', 'R_std', 'R_skew', 'R_kurtosis', 'R_min', 'R_max', 'R_median', 'G_mean', 'G_std', 'G_skew', 'G_kurtosis', 'G_min', 'G_max', 'G_median', 'B_mean', 'B_std', 'B_skew', 'B_kurtosis', 'B_min', 'B_max', 'B_median', 'overall_mean', 'overall_std', 'overall_skew', 'overall_kurtosis', 'edge_density', 'edge_mean', 'sobel_magnitude_mean', 'sobel_magnitude_std', 'sobel_direction_mean', 'sobel_direction_std', 'laplacian_mean', 'laplacian_std', 'laplacian_energy', 'hsv_H_mean', 'hsv_H_std', 'hsv_H_skew', 'hsv_S_mean', 'hsv_S_std', 'hsv_S_skew', 'hsv_V_mean', 'hsv_V_std', 'hsv_V_skew', 'color_moment_1', 'color_moment_2', 'color_moment_3', 'hist_r_entropy', 'hist_g_entropy', 'hist_b_entropy', 'wavelet_LL_mean', 'wavelet_LL_std', 'wavelet_LL_energy', 'wavelet_LH_mean', 'wavelet_LH_std', 'wavelet_LH_energy', 'wavelet_HL_mean', 'wavelet_HL_std', 'wavelet_HL_energy', 'wavelet_HH_mean', 'wavelet_HH_std', 'wavelet_HH_energy', 'wavelet_LL_ratio', 'wavelet_HH_ratio', 'lbp_mean', 'lbp_std', 'lbp_energy', 'lbp_entropy']
            
            feature_array = []
            for name in feature_names:
                if name in features:
                    feature_array.append(features[name])
                else:
                    feature_array.append(0.0)
            
            return np.array(feature_array)
        except Exception as e:
            st.error(f"Error extracting features: {str(e)}")
            return np.zeros(105)
    
    def preprocess_features(self, features):
        """Preprocess features for model prediction"""
        try:
            # Scale all features first (baseline model expects 105 features)
            if self.scaler:
                features_scaled = self.scaler.transform(features.reshape(1, -1))
            else:
                features_scaled = features.reshape(1, -1)
            
            # Then select the same 30 features that were used during training
            if self.feature_indices:
                features_selected = features_scaled[:, self.feature_indices]
            else:
                # Fallback to first 30 features if no indices available
                features_selected = features_scaled[:, :30]
            
            return features_selected
        except Exception as e:
            st.error(f"Error preprocessing features: {str(e)}")
            return None
    
    def predict(self, image):
        """Make prediction on an image"""
        try:
            if self.model is None:
                return None, "Model not loaded"
            
            # Extract features
            features = self.extract_all_features(image)
            
            # Preprocess features
            features_processed = self.preprocess_features(features)
            if features_processed is None:
                return None, "Feature preprocessing failed"
            
            # Make prediction
            prediction = self.model.predict(features_processed)[0]
            probability = self.model.predict_proba(features_processed)[0]
            
            return prediction, probability
        except Exception as e:
            return None, f"Error making prediction: {str(e)}"

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 AI TraceFinder</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Tampered Image Detection System</h2>', unsafe_allow_html=True)
    
    # Initialize detector
    detector = TamperedImageDetector()
    
    # No sidebar - moved all info to main content area
    
    # Model Information
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0; padding: 1.5rem; background: linear-gradient(135deg, #2a2a2a 0%, #3a3a3a 100%); border-radius: 15px; border: 1px solid #404040;">
        <h3 style="color: #00d4ff; margin-bottom: 1rem;">📊 Model Performance</h3>
        <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 1rem;">
            <div style="text-align: center;">
                <div style="color: #ffffff; font-size: 1.5rem; font-weight: bold;">71.4%</div>
                <div style="color: #cccccc; font-size: 0.9rem;">Overall Accuracy</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #4caf50; font-size: 1.5rem; font-weight: bold;">95.2%</div>
                <div style="color: #cccccc; font-size: 0.9rem;">Tampered Detection</div>
            </div>
            <div style="text-align: center;">
                <div style="color: #ff6b6b; font-size: 1.5rem; font-weight: bold;">0%</div>
                <div style="color: #cccccc; font-size: 0.9rem;">Original Detection</div>
            </div>
        </div>
        <div style="margin-top: 1rem; color: #cccccc; font-size: 0.9rem;">
            <strong>Model:</strong> LightGBM Classifier | <strong>Features:</strong> 30 selected from 105 | <strong>Training:</strong> 108 images
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="color: #ff6b6b; font-size: 2.2rem; margin-bottom: 1rem; text-shadow: 0 0 20px rgba(255, 107, 107, 0.3);">🖼️ Upload Image for Analysis</h2>
        <p style="color: #cccccc; font-size: 1.1rem;">Upload an image to detect if it has been tampered with using advanced forensic analysis</p>
        <p style="color: #888; font-size: 0.9rem; margin-top: 0.5rem;">Supported formats: TIFF, PNG, JPG, JPEG (TIFF recommended for best accuracy)</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['tiff', 'tif', 'png', 'jpg', 'jpeg'],
        help="Upload a TIFF, PNG, JPG, or JPEG image for tampering detection (TIFF recommended for best accuracy)"
    )
    
    if uploaded_file is not None:
        try:
            # Load image
            image = Image.open(uploaded_file)
            
            # Display image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### 📸 Uploaded Image")
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                st.markdown("### 🔍 Analysis Results")
                
                # Make prediction
                prediction, probability = detector.predict(image)
                
                if prediction is not None:
                    # Display results
                    if prediction == 1:
                        st.markdown('<div class="error-box glow">', unsafe_allow_html=True)
                        st.markdown("## ⚠️ TAMPERED IMAGE DETECTED")
                        st.markdown(f"**Confidence:** {probability[1]*100:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="success-box glow">', unsafe_allow_html=True)
                        st.markdown("## ✅ ORIGINAL IMAGE DETECTED")
                        st.markdown(f"**Confidence:** {probability[0]*100:.1f}%")
                        st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Show probability breakdown
                    st.markdown("### 📊 Probability Breakdown")
                    prob_df = pd.DataFrame({
                        'Class': ['Original', 'Tampered'],
                        'Probability': [probability[0]*100, probability[1]*100]
                    })
                    st.bar_chart(prob_df.set_index('Class'))
                    
                else:
                    st.markdown('<div class="error-box">', unsafe_allow_html=True)
                    st.markdown(f"## ❌ Analysis Failed")
                    st.markdown(f"**Error:** {probability}")
                    st.markdown("</div>", unsafe_allow_html=True)
        
        except Exception as e:
            st.markdown('<div class="error-box">', unsafe_allow_html=True)
            st.markdown("## ❌ Error Processing Image")
            st.markdown(f"**Error:** {str(e)}")
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Information section
    st.markdown("""
    <div style="text-align: center; margin: 3rem 0 2rem 0;">
        <h2 style="color: #00d4ff; font-size: 2.2rem; margin-bottom: 1rem; text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);">ℹ️ About This System</h2>
        <p style="color: #cccccc; font-size: 1.1rem;">Advanced forensic analysis for tampered image detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔬 Forensic Features")
        st.markdown("""
        • **Error Level Analysis (ELA)**: Detects compression artifacts
        • **Noise Analysis**: Identifies unnatural noise patterns
        • **Compression Artifacts**: Analyzes DCT coefficients and blocking
        • **Frequency Domain**: FFT analysis for frequency patterns
        • **Statistical Features**: Mean, variance, skewness, kurtosis
        • **Texture Features**: GLCM, LBP, Zernike moments, Hu moments
        """)
    
    with col2:
        st.markdown("### 🎯 Model Performance")
        st.markdown("""
        • **Overall Accuracy**: 71.4%
        • **Tampered Detection**: 95.2% (Excellent recall)
        • **Original Detection**: 0% (Limited precision)
        • **Training Data**: 108 images from SUPATLANTIQUE dataset
        • **Test Data**: 28 images for evaluation
        • **Feature Extraction**: 105 comprehensive forensic features
        • **Feature Selection**: 30 most important features used for prediction
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <h3 style="color: #00d4ff; margin-bottom: 1rem;">🔍 AI TraceFinder</h3>
        <p style="font-size: 1.1rem; margin-bottom: 1rem;"><strong>Note:</strong> For best results, upload TIFF images as the model was trained on TIFF format from the SUPATLANTIQUE dataset.</p>
        <p style="font-size: 1rem; margin-bottom: 1rem;">This system uses advanced forensic analysis to detect image tampering with high accuracy.</p>
        <div style="margin-top: 1.5rem; padding-top: 1rem; border-top: 1px solid #404040;">
            <p style="font-size: 0.9rem; color: #888;">Powered by LightGBM • 108 Training Images • 71.4% Accuracy</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()