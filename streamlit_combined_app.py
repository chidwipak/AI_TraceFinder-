#!/usr/bin/env python3
"""
AI TraceFinder - Combined Streamlit Application
Combines Objective 1 (Scanner Identification) and Objective 2 (Tampered Detection)
Uses best models: Stacking_Top5_Ridge (93.75%) and Hybrid_ML_DL (89.86%)
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

# Import our custom modules
from combined_feature_extractors import CombinedFeatureExtractor
from combined_model_loaders import CombinedModelManager

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
    .tampered-result {
        background: linear-gradient(135deg, #f8d7da, #f5c6cb);
        border-left-color: #dc3545;
        color: #721c24;
    }
    .original-result {
        background: linear-gradient(135deg, #d4edda, #a8e6cf);
        border-left-color: #28a745;
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
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border: 1px solid #e9ecef;
    }
    .upload-area {
        border: 3px dashed #1f77b4;
        border-radius: 1rem;
        padding: 3rem;
        text-align: center;
        background: #f8f9ff;
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    .upload-area:hover {
        border-color: #28a745;
        background: #f0fff4;
    }
    .processing-indicator {
        background: linear-gradient(90deg, #17a2b8, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
        margin: 1rem 0;
        font-weight: bold;
    }
    .results-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 1rem;
        margin: 2rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_manager' not in st.session_state:
    st.session_state.model_manager = CombinedModelManager()
if 'feature_extractor' not in st.session_state:
    st.session_state.feature_extractor = CombinedFeatureExtractor()
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

def load_models():
    """Load all models"""
    with st.spinner("Loading AI models..."):
        success = st.session_state.model_manager.load_all_models()
        if success:
            st.session_state.models_loaded = True
            st.success("✅ All models loaded successfully!")
            return True
        else:
            st.error("❌ Failed to load models. Please check model files.")
            return False

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
                image = np.array(images[0])  # Use first page
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
            image = image[:, :, :3]  # Remove alpha channel
        
        return image
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        return None

def display_results(results):
    """Display analysis results"""
    if "error" in results:
        st.error(f"❌ {results['error']}")
        return
    
    # Create two columns for results
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔍 Scanner Identification Results")
        scanner_result = results["scanner"]["result"]
        scanner_confidence = results["scanner"]["confidence"]
        
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
        for scanner in results["scanner"]["supported_scanners"]:
            st.markdown(f'<div class="scanner-card">• {scanner}</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 🛡️ Tampered Detection Results")
        tampered_result = results["tampered"]["result"]
        tampered_confidence = results["tampered"]["confidence"]
        tamper_type = results["tampered"]["tamper_type"]
        
        # Determine confidence class
        if tampered_confidence >= 0.7:
            conf_class = "confidence-high"
        elif tampered_confidence >= 0.4:
            conf_class = "confidence-medium"
        else:
            conf_class = "confidence-low"
        
        # Display result
        if tampered_result == "Tampered":
            st.markdown(f"""
            <div class="scanner-card tampered-result">
                <h4>⚠️ Image is Tampered</h4>
                <p>Evidence of digital manipulation detected.</p>
                <p><strong>Tamper Type:</strong> {tamper_type}</p>
                <p class="{conf_class}">Confidence: {tampered_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="scanner-card original-result">
                <h4>✅ Image is Original</h4>
                <p>No evidence of digital tampering detected.</p>
                <p class="{conf_class}">Confidence: {tampered_confidence:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

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
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Controls")
        
        # Model loading
        if not st.session_state.models_loaded:
            if st.button("🔄 Load Models", type="primary"):
                load_models()
        else:
            st.success("✅ Models Loaded")
        
        # Analysis options
        st.markdown("### 📊 Analysis Options")
        run_scanner_analysis = st.checkbox("Scanner Identification", value=True)
        run_tampered_analysis = st.checkbox("Tampered Detection", value=True)
        
        # Confidence threshold
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.3, 0.1)
        
        # Model info
        st.markdown("### ℹ️ Model Information")
        st.markdown("""
        **Scanner ID Model:**
        - Stacking_Top5_Ridge
        - 93.75% accuracy
        - 11 supported scanners
        
        **Tampered Detection Model:**
        - Hybrid_ML_DL Ensemble
        - 89.86% accuracy
        - Original vs Tampered
        """)
    
    # Main content
    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load models first using the sidebar controls.")
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
        image = preprocess_image(uploaded_file)
        if image is not None:
            # Display image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Analysis section
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Extract features for both objectives
                        scanner_features = st.session_state.feature_extractor.extract_scanner_features(image)
                        tampered_features = st.session_state.feature_extractor.extract_tampered_features(image)
                        
                        # Run predictions
                        results = {}
                        
                        if run_scanner_analysis:
                            scanner_result, scanner_confidence = st.session_state.model_manager.scanner_loader.predict_scanner(scanner_features)
                            results["scanner"] = {
                                "result": scanner_result,
                                "confidence": scanner_confidence,
                                "supported_scanners": st.session_state.model_manager.scanner_loader.supported_scanners
                            }
                        
                        if run_tampered_analysis:
                            tampered_result, tampered_confidence, tamper_type = st.session_state.model_manager.tampered_loader.predict_tampered(tampered_features)
                            results["tampered"] = {
                                "result": tampered_result,
                                "confidence": tampered_confidence,
                                "tamper_type": tamper_type
                            }
                        
                        # Display results
                        if results:
                            st.markdown("## 📊 Analysis Results")
                            display_results(results)
                        
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
    
    # Batch processing section
    st.markdown("## 📦 Batch Processing")
    st.markdown("Upload multiple images for batch analysis")
    
    uploaded_files = st.file_uploader(
        "Choose multiple image files",
        type=['png', 'jpg', 'jpeg', 'tiff', 'tif', 'pdf'],
        accept_multiple_files=True,
        help="Select multiple files for batch processing"
    )
    
    if uploaded_files and st.button("🔄 Process Batch", type="secondary"):
        if len(uploaded_files) > 10:
            st.warning("⚠️ Processing more than 10 files may take some time...")
        
        batch_results = []
        progress_bar = st.progress(0)
        
        for i, uploaded_file in enumerate(uploaded_files):
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Process each file
            image = preprocess_image(uploaded_file)
            if image is not None:
                try:
                    # Extract features
                    scanner_features = st.session_state.feature_extractor.extract_scanner_features(image)
                    tampered_features = st.session_state.feature_extractor.extract_tampered_features(image)
                    
                    # Predict
                    scanner_result, scanner_confidence = st.session_state.model_manager.scanner_loader.predict_scanner(scanner_features)
                    tampered_result, tampered_confidence, tamper_type = st.session_state.model_manager.tampered_loader.predict_tampered(tampered_features)
                    
                    batch_results.append({
                        "File": uploaded_file.name,
                        "Scanner": scanner_result,
                        "Scanner Confidence": f"{scanner_confidence:.1%}",
                        "Tampered": tampered_result,
                        "Tampered Confidence": f"{tampered_confidence:.1%}",
                        "Tamper Type": tamper_type
                    })
                    
                except Exception as e:
                    batch_results.append({
                        "File": uploaded_file.name,
                        "Scanner": "Error",
                        "Scanner Confidence": "0%",
                        "Tampered": "Error",
                        "Tampered Confidence": "0%",
                        "Tamper Type": "Error"
                    })
        
        # Display batch results
        if batch_results:
            st.markdown("### 📋 Batch Processing Results")
            df = pd.DataFrame(batch_results)
            st.dataframe(df, use_container_width=True)
            
            # Download results
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name=f"tracefinder_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6c757d; margin-top: 2rem;">
        <p>🔍 <strong>AI TraceFinder</strong> - Advanced Forensic Image Analysis</p>
        <p>Scanner Identification: 93.75% accuracy | Tampered Detection: 89.86% accuracy</p>
        <p>Built with Streamlit, scikit-learn, TensorFlow, and advanced computer vision techniques</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
