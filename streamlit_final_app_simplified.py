#!/usr/bin/env python3
"""
AI TraceFinder - Final Simplified Streamlit Application
Scanner Identification (93.75% accuracy) + Tampered Detection Placeholder
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# Image processing imports
import tifffile
import imageio.v2 as imageio
from pdf2image import convert_from_path
import cv2

# Import our fixed scanner identifier
from scanner_identification_fixed import FixedScannerIdentifier

# Configure page
st.set_page_config(
    page_title="AI TraceFinder - Forensic Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
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
        padding: 1rem 1.5rem;
        border-radius: 1rem;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .accuracy-badge:hover {
        transform: translateY(-2px);
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
    .tampered-result {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left-color: #dc3545;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: bold;
        color: #721c24;
    }
    .original-result {
        background: linear-gradient(135deg, #d4edda, #a8e6cf);
        border-left-color: #28a745;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-size: 1.1rem;
        font-weight: bold;
        color: #155724;
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
    .processing-status {
        background: linear-gradient(135deg, #e3f2fd, #bbdefb);
        padding: 1rem;
        border-radius: 0.8rem;
        border-left: 5px solid #2196f3;
        margin: 1rem 0;
        font-size: 1rem;
        color: #0d47a1;
    }
    .coming-soon {
        background: linear-gradient(135deg, #e2e3e5, #d1ecf1);
        border-left-color: #6c757d;
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 1rem 0;
        font-size: 1rem;
        color: #495057;
        text-align: center;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class SimpleTamperedDetector:
    """Simple tampered detection using basic image analysis"""
    
    def __init__(self):
        self.is_loaded = True  # Always available
    
    def predict_tampered(self, image):
        """Simple heuristic-based tampered detection"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Simple heuristics
            variance = np.var(gray)
            edge_density = self.calculate_edge_density(gray)
            noise_level = self.calculate_noise_level(gray)
            
            # Simple decision logic
            tamper_score = 0
            
            # High variance might indicate tampering
            if variance > 8000:
                tamper_score += 1
            
            # High edge density might indicate tampering
            if edge_density > 0.3:
                tamper_score += 1
            
            # High noise level might indicate tampering
            if noise_level > 0.1:
                tamper_score += 1
            
            # Decision
            if tamper_score >= 2:
                result = "Tampered"
                confidence = 0.65
            else:
                result = "Original"
                confidence = 0.70
            
            return result, confidence
            
        except Exception as e:
            return "Analysis Error", 0.0
    
    def calculate_edge_density(self, image):
        """Calculate edge density using Sobel operator"""
        try:
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            edge_density = np.sum(edge_magnitude > np.mean(edge_magnitude)) / edge_magnitude.size
            return edge_density
        except:
            return 0.0
    
    def calculate_noise_level(self, image):
        """Calculate noise level using Laplacian"""
        try:
            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            noise_level = np.var(laplacian) / 10000  # Normalize
            return noise_level
        except:
            return 0.0

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
                return None, None
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
    st.markdown('<div class="sub-header">Forensic Image Analysis & Scanner Identification</div>', unsafe_allow_html=True)
    
    # Accuracy badges
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="accuracy-badge">🎯 Scanner ID<br>93.75% Accuracy<br>11 Scanner Models</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="accuracy-badge">🛡️ Tampered Detection<br>Basic Analysis<br>Heuristic-based</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="accuracy-badge">🚀 Combined Analysis<br>Real-time Processing<br>PDF Support</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'scanner_identifier' not in st.session_state:
        st.session_state.scanner_identifier = FixedScannerIdentifier()
    
    if 'tampered_detector' not in st.session_state:
        st.session_state.tampered_detector = SimpleTamperedDetector()
    
    # Load models
    if not st.session_state.scanner_identifier.is_loaded:
        with st.spinner("Loading Scanner Identification Model..."):
            success = st.session_state.scanner_identifier.load_model()
            if success:
                st.success("✅ Scanner identification model loaded successfully!")
            else:
                st.error("❌ Failed to load scanner identification model.")
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
            st.image(color_image, caption="Uploaded Image", use_container_width=True)
            
            # Analysis section
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Scanner prediction
                        scanner_result, scanner_confidence, scanner_status = st.session_state.scanner_identifier.predict_scanner(gray_image)
                        
                        # Tampered detection
                        tampered_result, tampered_confidence = st.session_state.tampered_detector.predict_tampered(color_image)
                        
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
                            
                            if tampered_result == "Tampered":
                                st.markdown(f"""
                                <div class="tampered-result">
                                    <h4>🚨 Image is TAMPERED</h4>
                                    <p>Evidence of digital manipulation detected!</p>
                                    <p class="confidence-high">Confidence: {tampered_confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            elif tampered_result == "Original":
                                st.markdown(f"""
                                <div class="original-result">
                                    <h4>✅ Image is ORIGINAL</h4>
                                    <p>No signs of digital manipulation detected.</p>
                                    <p class="confidence-high">Confidence: {tampered_confidence:.1%}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.error(f"❌ Tampered detection error: {tampered_result}")
                            
                            # Note about basic analysis
                            st.markdown("""
                            <div class="coming-soon">
                                <p><strong>Note:</strong> Basic heuristic-based analysis</p>
                                <p>Advanced ML model coming soon!</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Combined analysis summary
                        st.markdown("### 📋 Combined Analysis Summary")
                        
                        if scanner_result != "Unknown Scanner":
                            st.success(f"✅ **Scanner Identification Successful**")
                            st.info(f"**Scanner:** {scanner_result} (Confidence: {scanner_confidence:.1%})")
                            st.info(f"**Authenticity:** {tampered_result} (Confidence: {tampered_confidence:.1%})")
                        else:
                            st.warning("⚠️ **Partial Analysis** - Scanner not recognized")
                            st.info(f"**Scanner:** {scanner_result} (Confidence: {scanner_confidence:.1%})")
                            st.info(f"**Authenticity:** {tampered_result} (Confidence: {tampered_confidence:.1%})")
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
    
    # Sidebar information
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("**Scanner Identification:**")
        st.markdown("- Model: Stacking_Top5_Ridge")
        st.markdown("- Accuracy: 93.75%")
        st.markdown("- Features: 153 (PRNU, LBP, GLCM, FFT, DCT)")
        st.markdown("- Preprocessing: SelectKBest (90 features)")
        
        st.markdown("**Tampered Detection:**")
        st.markdown("- Method: Heuristic-based")
        st.markdown("- Analysis: Edge density, noise level, variance")
        st.markdown("- Status: Basic implementation")
        st.markdown("- Note: Advanced ML model in development")
        
        st.markdown("---")
        st.header("📁 Supported Formats")
        st.markdown("- **Images:** PNG, JPG, TIFF")
        st.markdown("- **Documents:** PDF (first page)")
        st.markdown("- **Processing:** Automatic conversion")
        
        st.markdown("---")
        st.header("⚠️ Important Notes")
        st.markdown("- Scanner ID works with 11 specific models")
        st.markdown("- Unknown scanners show low confidence")
        st.markdown("- Tampered detection is basic heuristic")
        st.markdown("- Results may vary with image quality")
        
        st.markdown("---")
        st.header("🔬 Technical Details")
        st.markdown("- **Scanner Features:** PRNU fingerprints, texture analysis")
        st.markdown("- **Tampered Features:** Edge detection, noise analysis")
        st.markdown("- **Preprocessing:** Image normalization, feature selection")
        st.markdown("- **Models:** Ensemble learning, heuristic analysis")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-top: 2rem;">
        <p>🔍 <strong>AI TraceFinder</strong> - Forensic Image Analysis System</p>
        <p>Scanner Identification: 93.75% accuracy | Tampered Detection: Basic heuristic analysis</p>
        <p>Built with Streamlit, scikit-learn, and computer vision techniques</p>
        <p><em>Advanced forensic investigation for digital evidence analysis</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
