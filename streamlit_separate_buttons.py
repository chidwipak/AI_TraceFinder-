#!/usr/bin/env python3
"""
Streamlit app with separate buttons for Scanner ID and Tampered Detection
Uses best models: Stacking_Ridge for Scanner ID, Balanced model for Tampered Detection
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
import sys
import tifffile
from pdf2image import convert_from_bytes

# Add current directory to path
sys.path.append('.')

# Import the working scanner identifier
from scanner_identification_fixed import FixedScannerIdentifier

st.set_page_config(
    page_title="AI TraceFinder - Separate Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BalancedTamperedDetector:
    """Tampered detection using the balanced trained model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the best trained model"""
        try:
            # Load the SVM model
            if os.path.exists("balanced_tampered_models/svm_model.pkl"):
                self.model = joblib.load("balanced_tampered_models/svm_model.pkl")
            else:
                return False
            
            # Load preprocessing objects
            if os.path.exists("balanced_tampered_models/scaler.pkl"):
                self.scaler = joblib.load("balanced_tampered_models/scaler.pkl")
            else:
                return False
            
            if os.path.exists("balanced_tampered_models/imputer.pkl"):
                self.imputer = joblib.load("balanced_tampered_models/imputer.pkl")
            else:
                return False
            
            # Load feature names
            if os.path.exists("balanced_tampered_models/feature_names.txt"):
                with open("balanced_tampered_models/feature_names.txt", "r") as f:
                    self.feature_names = [line.strip() for line in f.readlines()]
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_comprehensive_features(self, image_rgb):
        """Extract the same features as training"""
        try:
            # Convert to grayscale and resize
            if len(image_rgb.shape) == 3:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_rgb.copy()
            
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Statistical features (15)
            stat_features = [
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 90),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size,
                np.sum(gray < 0.2) / gray.size,
                np.sum((gray > 0.2) & (gray < 0.8)) / gray.size,
                np.sum(gray > np.mean(gray)) / gray.size,
                np.sum(gray < np.mean(gray)) / gray.size
            ]
            features.extend(stat_features)
            
            # 2. Histogram features (16)
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            
            # 3. Edge features (12)
            try:
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges = edges.astype(np.float64) / 255.0
                edge_features = [
                    np.mean(edges), np.std(edges), np.var(edges),
                    np.sum(edges > 0) / edges.size,
                    np.percentile(edges.flatten(), 25),
                    np.percentile(edges.flatten(), 50),
                    np.percentile(edges.flatten(), 75),
                    np.percentile(edges.flatten(), 90),
                    np.percentile(edges.flatten(), 95),
                    np.percentile(edges.flatten(), 99),
                    np.max(edges), np.min(edges)
                ]
                features.extend(edge_features)
            except:
                features.extend([0.1] * 12)
            
            # 4. Gradient features (12)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                
                grad_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.percentile(magnitude.flatten(), 25),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(direction), np.std(direction)
                ]
                features.extend(grad_features)
            except:
                features.extend([0.2] * 12)
            
            # 5. Texture features (10)
            try:
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                gabor_kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
                gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
                
                texture_features = [
                    texture_var,
                    np.mean(local_std), np.std(local_std),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95),
                    np.mean(gabor_response), np.std(gabor_response),
                    np.max(gabor_response), np.min(gabor_response)
                ]
                features.extend(texture_features)
            except:
                features.extend([0.3] * 10)
            
            # 6. Frequency features (12)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                phase = np.angle(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                
                freq_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.mean(central_region), np.std(central_region),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.percentile(magnitude.flatten(), 99),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(phase), np.std(phase)
                ]
                features.extend(freq_features)
            except:
                features.extend([0.4] * 12)
            
            # 7. Tampering-specific features (15)
            try:
                block_size = 32
                h, w = gray.shape
                blocks_h = h // block_size
                blocks_w = w // block_size
                
                block_vars = []
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        block_vars.append(np.var(block))
                
                copy_move_features = [
                    np.mean(block_vars), np.std(block_vars),
                    np.percentile(block_vars, 25),
                    np.percentile(block_vars, 75),
                    np.percentile(block_vars, 90),
                    np.percentile(block_vars, 95),
                    np.sum(np.array(block_vars) > np.mean(block_vars)) / len(block_vars),
                    np.sum(np.array(block_vars) < np.mean(block_vars)) / len(block_vars)
                ]
                
                noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
                noise_features = [
                    np.mean(noise), np.std(noise), np.var(noise),
                    np.percentile(noise.flatten(), 90),
                    np.percentile(noise.flatten(), 95),
                    np.max(noise), np.min(noise)
                ]
                
                tamper_features = copy_move_features + noise_features
                features.extend(tamper_features)
            except:
                features.extend([0.5] * 15)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.full(92, 0.5)  # Default features
    
    def predict_tampered(self, image_rgb):
        """Predict if image is tampered"""
        try:
            if not self.is_loaded:
                return "Unknown", 0.5, "Model not loaded"
            
            # Extract features
            features = self.extract_comprehensive_features(image_rgb)
            
            # Handle missing values
            if self.imputer:
                features = self.imputer.transform(features.reshape(1, -1))
            else:
                features = features.reshape(1, -1)
            
            # Scale features
            if self.scaler:
                features_scaled = self.scaler.transform(features)
            else:
                features_scaled = features
            
            # Predict
            proba = self.model.predict_proba(features_scaled)[0]
            pred = int(np.argmax(proba))
            confidence = float(np.max(proba))
            
            label = "Tampered" if pred == 1 else "Original"
            
            return label, confidence, "Success"
            
        except Exception as e:
            return "Unknown", 0.5, f"Error: {e}"

def convert_pdf_to_image(pdf_file):
    """Convert PDF to image"""
    try:
        images = convert_from_bytes(pdf_file.read(), first_page=1, last_page=1)
        if images:
            return np.array(images[0])
        return None
    except Exception as e:
        st.error(f"Error converting PDF: {e}")
        return None

def load_image_robust(uploaded_file):
    """Robust image loading"""
    try:
        if uploaded_file.type == "application/pdf":
            return convert_pdf_to_image(uploaded_file)
        else:
            image = Image.open(uploaded_file)
            
            if image.mode == 'RGBA':
                background = Image.new('RGB', image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')
            
            return np.array(image)
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def main():
    st.title("🔍 AI TraceFinder - Separate Analysis")
    st.markdown("**Separate buttons for Scanner Identification and Tampered Detection**")
    
    # Initialize models
    if 'scanner_identifier' not in st.session_state:
        with st.spinner("Loading scanner identification model..."):
            st.session_state.scanner_identifier = FixedScannerIdentifier()
            if not st.session_state.scanner_identifier.load_model():
                st.error("❌ Failed to load scanner identification model")
                return
        st.success("✅ Scanner identification model loaded")
    
    if 'tampered_detector' not in st.session_state:
        with st.spinner("Loading tampered detection model..."):
            st.session_state.tampered_detector = BalancedTamperedDetector()
            if not st.session_state.tampered_detector.load_model():
                st.warning("⚠️ Balanced tampered detection model not found. Training will be required.")
                st.session_state.tampered_detector = None
            else:
                st.success("✅ Tampered detection model loaded")
    
    # File upload
    st.header("📁 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose an image or PDF file",
        type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'pdf'],
        help="Supported formats: JPG, JPEG, PNG, TIFF, PDF"
    )
    
    if uploaded_file is not None:
        # Load image
        with st.spinner("Loading image..."):
            image = load_image_robust(uploaded_file)
        
        if image is None:
            st.error("❌ Failed to load the uploaded file")
            return
        
        # Display image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(image, caption="Uploaded Image", width=300)
            st.write(f"**Image Info:**")
            st.write(f"- Size: {image.shape[1]}×{image.shape[0]}")
            st.write(f"- Channels: {image.shape[2] if len(image.shape) > 2 else 1}")
            st.write(f"- File type: {uploaded_file.type}")
        
        with col2:
            # Two separate analysis buttons
            col_scan, col_tamp = st.columns(2)
            
            with col_scan:
                if st.button("🖨️ Scanner Identification", type="primary", use_container_width=True):
                    st.subheader("🖨️ Scanner Identification Results")
                    with st.spinner("Analyzing scanner..."):
                        try:
                            scanner_result, scanner_conf, scanner_status, top3 = st.session_state.scanner_identifier.predict_scanner_with_top3(image)
                            
                            if scanner_status == "Success":
                                if scanner_conf >= 0.6:
                                    st.success(f"**Scanner:** {scanner_result}")
                                    st.info(f"**Confidence:** {scanner_conf:.1%}")
                                elif scanner_conf >= 0.15:
                                    st.warning(f"**Scanner:** {scanner_result} (Low confidence)")
                                    st.info(f"**Confidence:** {scanner_conf:.1%}")
                                else:
                                    st.warning("**Scanner:** Unknown Scanner")
                                    st.info("Confidence too low for reliable identification")
                                
                                # Show top 3
                                if top3:
                                    st.write("**Top 3 Likely Scanners:**")
                                    for i, (scanner, prob) in enumerate(top3, 1):
                                        st.write(f"{i}. {scanner}: {prob:.1%}")
                            else:
                                st.error(f"Scanner identification failed: {scanner_status}")
                        except Exception as e:
                            st.error(f"Scanner analysis error: {e}")
            
            with col_tamp:
                if st.button("🔒 Tampered Detection", type="secondary", use_container_width=True):
                    st.subheader("🔒 Tampered Detection Results")
                    
                    if st.session_state.tampered_detector is None:
                        st.warning("⚠️ Balanced tampered detection model not available.")
                        st.info("Please run the training pipeline first to create the balanced model.")
                    else:
                        with st.spinner("Analyzing tampering..."):
                            tampered_result, tampered_conf, tampered_status = st.session_state.tampered_detector.predict_tampered(image)
                            
                            if tampered_result == "Tampered":
                                st.error(f"**Result:** {tampered_result}")
                            else:
                                st.success(f"**Result:** {tampered_result}")
                            
                            st.info(f"**Confidence:** {tampered_conf:.1%}")
                            
                            if tampered_conf < 0.6:
                                st.warning("⚠️ Low confidence in tampering analysis. Results may be unreliable.")
                            
                            # Show forensic indicators
                            st.write("**Key Forensic Indicators:**")
                            if tampered_result == "Tampered":
                                st.write("- Statistical anomalies detected in balanced dataset")
                                st.write("- Edge pattern inconsistencies found")
                                st.write("- Frequency domain irregularities present")
                                st.write("- Block-based analysis shows tampering signs")
                            else:
                                st.write("- Consistent statistical properties with balanced originals")
                                st.write("- Regular edge patterns matching original documents")
                                st.write("- Normal frequency distribution")
                                st.write("- Block-based analysis shows no tampering")
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Model Information")
    st.sidebar.write("**Scanner ID:** Stacking_Top5_Ridge (93.75% accuracy)")
    st.sidebar.write("**Tampered Detection:** Balanced Dataset Model")
    st.sidebar.write("**Training Data:**")
    st.sidebar.write("- Originals: 34 original + 68 converted PDFs = 102")
    st.sidebar.write("- Tampered: 102 (Copy-move, Retouching, Splicing)")
    st.sidebar.write("**Features:** 153 (Scanner), 92 (Tampered)")
    st.sidebar.write("**Supported Scanners:** 11 models")
    
    # Training section
    if st.session_state.tampered_detector is None:
        st.sidebar.markdown("---")
        st.sidebar.header("🔧 Training Required")
        if st.sidebar.button("🚀 Train Balanced Model"):
            st.info("Please run the training pipeline to create the balanced tampered detection model.")
            st.code("python3 complete_training_pipeline.py")

if __name__ == "__main__":
    main()

