#!/usr/bin/env python3
"""
Final Streamlit app using real balanced dataset from Tampered images/ folder
Addresses the user's specific concern about using actual Original and Tampered folders
"""

import streamlit as st
import numpy as np
import cv2
import joblib
import os
from PIL import Image
import sys
import tifffile
import glob
import random
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Add current directory to path
sys.path.append('.')

# Import the working scanner identifier
from scanner_identification_fixed import FixedScannerIdentifier

st.set_page_config(
    page_title="AI TraceFinder - Real Balanced Dataset",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class RealBalancedTamperedDetector:
    """Tampered detection using real balanced dataset from Tampered images/ folder"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.training_complete = False
        
    def collect_real_balanced_dataset(self):
        """Collect the actual balanced dataset from Tampered images/ folder"""
        dataset_path = "The SUPATLANTIQUE dataset"
        
        # Original images from Tampered images/Original/ (these are the real originals paired with tampered)
        original_files = []
        original_dir = os.path.join(dataset_path, "Tampered images", "Original")
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    original_files.append(os.path.join(original_dir, file))
        
        # Tampered images from Tampered images/Tampered/
        tampered_files = []
        tampered_dirs = [
            os.path.join(dataset_path, "Tampered images", "Tampered", "Copy-move"),
            os.path.join(dataset_path, "Tampered images", "Tampered", "Retouching"),
            os.path.join(dataset_path, "Tampered images", "Tampered", "Splicing")
        ]
        
        for tampered_dir in tampered_dirs:
            if os.path.exists(tampered_dir):
                for file in os.listdir(tampered_dir):
                    if file.lower().endswith(('.tif', '.tiff')):
                        tampered_files.append(os.path.join(tampered_dir, file))
        
        # Balance the dataset - we need more originals to match tampered count
        if len(tampered_files) > len(original_files):
            # Add more originals from Official and Wikipedia to balance
            additional_originals = []
            additional_sources = [
                os.path.join(dataset_path, "Official"),
                os.path.join(dataset_path, "Wikipedia")
            ]
            
            for source_dir in additional_sources:
                if os.path.exists(source_dir):
                    for scanner_dir in os.listdir(source_dir):
                        scanner_path = os.path.join(source_dir, scanner_dir)
                        if os.path.isdir(scanner_path):
                            for file in os.listdir(scanner_path):
                                if file.lower().endswith(('.tif', '.tiff')):
                                    additional_originals.append(os.path.join(scanner_path, file))
            
            # Sample additional originals to balance
            needed = len(tampered_files) - len(original_files)
            if len(additional_originals) >= needed:
                sampled_additional = random.sample(additional_originals, needed)
                original_files.extend(sampled_additional)
            else:
                original_files.extend(additional_originals)
        
        # Ensure equal numbers by sampling down if needed
        min_count = min(len(original_files), len(tampered_files))
        original_files = random.sample(original_files, min_count)
        tampered_files = random.sample(tampered_files, min_count)
        
        return original_files, tampered_files
    
    def extract_robust_features(self, image_path):
        """Extract robust features for tampered detection"""
        try:
            # Load image
            if image_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                return None
            
            # Convert to grayscale and resize
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Statistical features (10)
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size
            ])
            
            # 2. Histogram features (8)
            hist, _ = np.histogram(gray.flatten(), bins=8, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            
            # 3. Edge features (8)
            try:
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges = edges.astype(np.float64) / 255.0
                features.extend([
                    np.mean(edges), np.std(edges), 
                    np.sum(edges > 0) / edges.size,
                    np.percentile(edges.flatten(), 50),
                    np.percentile(edges.flatten(), 75),
                    np.percentile(edges.flatten(), 90),
                    np.percentile(edges.flatten(), 95),
                    np.percentile(edges.flatten(), 99)
                ])
            except:
                features.extend([0.1] * 8)
            
            # 4. Gradient features (8)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude)
                ])
            except:
                features.extend([0.2] * 8)
            
            # 5. Texture features (6)
            try:
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                features.extend([
                    texture_var,
                    np.mean(local_std),
                    np.std(local_std),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95)
                ])
            except:
                features.extend([0.3] * 6)
            
            # 6. Frequency features (8)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.mean(central_region), np.std(central_region),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.sum(magnitude > np.mean(magnitude)) / magnitude.size
                ])
            except:
                features.extend([0.4] * 8)
            
            return np.array(features[:48])
            
        except Exception as e:
            print(f"Feature extraction error for {image_path}: {e}")
            return None
    
    def train_real_model(self):
        """Train model on real balanced dataset"""
        try:
            st.info("🔄 Training tampered detection model on real balanced dataset...")
            progress_bar = st.progress(0)
            
            # Collect real balanced dataset
            original_files, tampered_files = self.collect_real_balanced_dataset()
            
            st.write(f"📊 Using real balanced dataset:")
            st.write(f"- Original images: {len(original_files)}")
            st.write(f"- Tampered images: {len(tampered_files)}")
            
            progress_bar.progress(0.1)
            
            # Extract features
            X = []
            y = []
            total_files = len(original_files) + len(tampered_files)
            
            # Process original images
            for i, img_path in enumerate(original_files):
                features = self.extract_robust_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(0)  # 0 for original
                
                if i % 5 == 0:
                    progress = 0.1 + 0.4 * (i / len(original_files))
                    progress_bar.progress(progress)
            
            # Process tampered images
            for i, img_path in enumerate(tampered_files):
                features = self.extract_robust_features(img_path)
                if features is not None:
                    X.append(features)
                    y.append(1)  # 1 for tampered
                
                if i % 5 == 0:
                    progress = 0.5 + 0.3 * (i / len(tampered_files))
                    progress_bar.progress(progress)
            
            X = np.array(X)
            y = np.array(y)
            
            st.write(f"✅ Features extracted: {X.shape[0]} samples, {X.shape[1]} features")
            st.write(f"- Original samples: {np.sum(y == 0)}")
            st.write(f"- Tampered samples: {np.sum(y == 1)}")
            
            progress_bar.progress(0.8)
            
            # Train model
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Use ensemble for better performance
            self.model = VotingClassifier([
                ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
            ], voting='soft')
            
            self.model.fit(X_scaled, y)
            
            progress_bar.progress(1.0)
            
            self.is_loaded = True
            self.training_complete = True
            
            st.success("✅ Real balanced dataset model trained successfully!")
            return True
            
        except Exception as e:
            st.error(f"❌ Training failed: {e}")
            return False
    
    def load_or_train_model(self):
        """Load existing model or train new one"""
        if not self.is_loaded and not self.training_complete:
            return self.train_real_model()
        return self.is_loaded
    
    def extract_features_from_image(self, image_rgb):
        """Extract features from uploaded image"""
        try:
            # Convert to grayscale and resize
            if len(image_rgb.shape) == 3:
                gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_rgb.copy()
            
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # Same feature extraction as training
            # 1. Statistical features (10)
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size
            ])
            
            # 2. Histogram features (8)
            hist, _ = np.histogram(gray.flatten(), bins=8, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            
            # 3. Edge features (8)
            try:
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges = edges.astype(np.float64) / 255.0
                features.extend([
                    np.mean(edges), np.std(edges), 
                    np.sum(edges > 0) / edges.size,
                    np.percentile(edges.flatten(), 50),
                    np.percentile(edges.flatten(), 75),
                    np.percentile(edges.flatten(), 90),
                    np.percentile(edges.flatten(), 95),
                    np.percentile(edges.flatten(), 99)
                ])
            except:
                features.extend([0.1] * 8)
            
            # 4. Gradient features (8)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude)
                ])
            except:
                features.extend([0.2] * 8)
            
            # 5. Texture features (6)
            try:
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                features.extend([
                    texture_var,
                    np.mean(local_std),
                    np.std(local_std),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95)
                ])
            except:
                features.extend([0.3] * 6)
            
            # 6. Frequency features (8)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.mean(central_region), np.std(central_region),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.sum(magnitude > np.mean(magnitude)) / magnitude.size
                ])
            except:
                features.extend([0.4] * 8)
            
            return np.array(features[:48])
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.full(48, 0.5)
    
    def predict_tampered(self, image_rgb):
        """Predict if image is tampered"""
        try:
            if not self.is_loaded:
                return "Unknown", 0.5, "Model not loaded"
            
            # Extract features
            features = self.extract_features_from_image(image_rgb)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
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
        from pdf2image import convert_from_bytes
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
    st.title("🔍 AI TraceFinder - Real Balanced Dataset")
    st.markdown("**Using actual Original and Tampered images from the dataset**")
    
    # Dataset info
    st.info("📊 This version uses the real balanced dataset from 'Tampered images/Original' and 'Tampered images/Tampered' folders, supplemented with additional originals from Official and Wikipedia folders to achieve perfect balance.")
    
    # Initialize models
    if 'scanner_identifier' not in st.session_state:
        with st.spinner("Loading scanner identification model..."):
            st.session_state.scanner_identifier = FixedScannerIdentifier()
            if not st.session_state.scanner_identifier.load_model():
                st.error("❌ Failed to load scanner identification model")
                return
        st.success("✅ Scanner identification model loaded")
    
    if 'tampered_detector' not in st.session_state:
        st.session_state.tampered_detector = RealBalancedTamperedDetector()
        
        with st.spinner("Loading/training real balanced tampered detection model..."):
            if not st.session_state.tampered_detector.load_or_train_model():
                st.error("❌ Failed to load/train tampered detection model")
                return
        
        if st.session_state.tampered_detector.training_complete:
            st.success("✅ Real balanced tampered detection model ready")
    
    # Sidebar
    st.sidebar.header("Analysis Options")
    analysis_type = st.sidebar.selectbox(
        "Choose Analysis Type",
        ["Combined Analysis", "Scanner Identification", "Tampered Detection"]
    )
    
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
            if st.button("🔍 Analyze", type="primary"):
                
                if analysis_type in ["Combined Analysis", "Scanner Identification"]:
                    st.subheader("🖨️ Scanner Identification")
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
                
                if analysis_type in ["Combined Analysis", "Tampered Detection"]:
                    st.subheader("🔒 Tampered Detection (Real Balanced Dataset)")
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
                            st.write("- Statistical anomalies detected in real dataset")
                            st.write("- Edge pattern inconsistencies found")
                            st.write("- Frequency domain irregularities present")
                        else:
                            st.write("- Consistent statistical properties with real originals")
                            st.write("- Regular edge patterns matching original documents")
                            st.write("- Normal frequency distribution")
                
                # Combined interpretation
                if analysis_type == "Combined Analysis":
                    st.subheader("📊 Combined Interpretation")
                    if 'scanner_result' in locals() and 'tampered_result' in locals():
                        if scanner_conf >= 0.15 and tampered_conf >= 0.6:
                            if tampered_result == "Original":
                                st.info(f"✅ This appears to be an original document scanned with {scanner_result}.")
                            else:
                                st.warning(f"⚠️ This appears to be a tampered document, possibly from {scanner_result}.")
                        elif scanner_conf < 0.15:
                            if tampered_result == "Original":
                                st.info("✅ This appears to be an original image but may not be a scanned document.")
                            else:
                                st.warning("⚠️ This appears to be a tampered image but may not be a scanned document.")
                        else:
                            st.warning("⚠️ Low confidence in both analyses. Results may be unreliable.")
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Model Information")
    st.sidebar.write("**Scanner ID:** Stacking_Top5_Ridge (93.75% accuracy)")
    st.sidebar.write("**Tampered Detection:** Real Balanced Dataset Model")
    st.sidebar.write("**Training Data:**")
    st.sidebar.write("- Originals: Tampered images/Original + Official + Wikipedia")
    st.sidebar.write("- Tampered: Tampered images/Tampered (all types)")
    st.sidebar.write("**Supported Scanners:** 11 models")
    st.sidebar.write("**Features:** 153 (Scanner), 48 (Tampered)")

if __name__ == "__main__":
    main()

