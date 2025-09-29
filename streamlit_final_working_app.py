#!/usr/bin/env python3
"""
Final Working Streamlit Application for AI TraceFinder
Combines Scanner Identification and Tampered Detection
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import pickle
from PIL import Image
import tempfile
import traceback
from scanner_identification_fixed import FixedScannerIdentifier
from combined_feature_extractors import CombinedFeatureExtractor

# Page configuration
st.set_page_config(
    page_title="AI TraceFinder - Scanner & Tampered Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TamperedDetector:
    """Production tampered detection using trained ensemble and exact features"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.model_path = "new_objective2_ensemble_results/models/best_ensemble_model.pkl"
        self.scaler_path = "new_objective2_ensemble_results/models/feature_scaler.pkl"
        self.feature_extractor = CombinedFeatureExtractor()
    
    def load_model(self):
        try:
            import joblib
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                print(f"❌ Objective2 model/scaler missing: {self.model_path}, {self.scaler_path}")
                return False
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            print("✅ Loaded Objective2 ensemble model and scaler")
            return True
        except Exception as e:
            print(f"❌ Error loading Objective2 model: {e}")
            traceback.print_exc()
            return False
    
    def predict_tampered(self, image_rgb):
        try:
            if not self.is_loaded:
                return "Unknown", 0.0, "Model not loaded"
            # Extract exact 84-dim features
            features = self.feature_extractor.extract_tampered_features(image_rgb)
            features = features.reshape(1, -1)
            features_scaled = self.scaler.transform(features)
            # Predict
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)[0]
                pred = int(np.argmax(proba))
                confidence = float(np.max(proba))
            else:
                pred = int(self.model.predict(features_scaled)[0])
                confidence = 0.5
            label = "Tampered" if pred == 1 else "Original"
            return label, confidence, "Success"
        except Exception as e:
            print(f"❌ Error in tampered prediction: {e}")
            return "Unknown", 0.0, f"Error: {e}"

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔍 AI TraceFinder - Complete Analysis")
    st.markdown("""
    **Advanced Scanner Identification & Tampered Detection System**
    
    Upload an image to analyze:
    - **Scanner Identification**: Identify which scanner model created the image
    - **Tampered Detection**: Detect if the image has been tampered with
    """)
    
    # Initialize session state
    if 'scanner_identifier' not in st.session_state:
        st.session_state.scanner_identifier = FixedScannerIdentifier()
    
    if 'tampered_detector' not in st.session_state:
        st.session_state.tampered_detector = TamperedDetector()
    
    # Load models
    if st.session_state.scanner_identifier.model is None:
        with st.spinner("Loading Scanner Identification Model..."):
            success = st.session_state.scanner_identifier.load_model()
            if success:
                st.success("✅ Scanner identification model loaded successfully!")
            else:
                st.error("❌ Failed to load scanner identification model.")
                return
    
    if not st.session_state.tampered_detector.is_loaded:
        with st.spinner("Loading Tampered Detection Model..."):
            success = st.session_state.tampered_detector.load_model()
            if success:
                st.success("✅ Tampered detection model loaded successfully!")
            else:
                st.error("❌ Failed to load tampered detection model.")
                st.info("⚠️ Tampered detection will be disabled.")
    
    # File upload section
    st.markdown("## 📁 Upload Image for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'pdf'],
        help="Upload an image file to analyze for scanner identification and tampered detection"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded Image")
            try:
                # Handle different file types
                if uploaded_file.type == "application/pdf":
                    st.info("📄 PDF detected - will be converted to image for analysis")
                    # For now, show a placeholder for PDF
                    st.image("https://via.placeholder.com/400x300/cccccc/666666?text=PDF+File", 
                            caption="PDF file uploaded", use_container_width=True)
                else:
                    # Display image
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Error displaying image: {e}")
        
        with col2:
            st.markdown("### 🎯 Analysis Results")
            
            # Process image
            if st.button("🔍 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Read image
                        if uploaded_file.type == "application/pdf":
                            st.warning("⚠️ PDF conversion not implemented yet. Please upload an image file.")
                            return
                        else:
                            image = Image.open(uploaded_file)
                            image_array = np.array(image)
                        
                        # Scanner identification
                        st.markdown("#### 🔍 Scanner Identification")
                        scanner_result, scanner_confidence, scanner_status = st.session_state.scanner_identifier.predict_scanner(image_array)
                        
                        if scanner_status == "Success":
                            if scanner_confidence >= 0.6:
                                st.success(f"✅ **Scanner**: {scanner_result}")
                                st.info(f"📊 **Confidence**: {scanner_confidence:.1%}")
                            else:
                                st.warning(f"⚠️ **Scanner**: {scanner_result}")
                                st.info(f"📊 **Confidence**: {scanner_confidence:.1%}")
                        else:
                            st.error(f"❌ **Scanner**: {scanner_result}")
                            st.info(f"📊 **Status**: {scanner_status}")
                        
                        # Tampered detection
                        st.markdown("#### 🔒 Tampered Detection")
                        if st.session_state.tampered_detector.is_loaded:
                            tampered_result, tampered_confidence, tampered_status = st.session_state.tampered_detector.predict_tampered(image_array)
                            
                            if tampered_status == "Success":
                                if tampered_result == "Tampered":
                                    st.error(f"🚨 **Result**: {tampered_result}")
                                else:
                                    st.success(f"✅ **Result**: {tampered_result}")
                                st.info(f"📊 **Confidence**: {tampered_confidence:.1%}")
                            else:
                                st.error(f"❌ **Result**: {tampered_result}")
                                st.info(f"📊 **Status**: {tampered_status}")
                        else:
                            st.warning("⚠️ Tampered detection model not available")
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        st.error(traceback.format_exc())
    
    # Sidebar information
    with st.sidebar:
        st.markdown("## ℹ️ Supported Scanner Models")
        
        supported_scanners = {
            "Canon120-1": "Canon CanoScan LiDE 120 (Model 1)",
            "Canon120-2": "Canon CanoScan LiDE 120 (Model 2)", 
            "Canon220": "Canon CanoScan LiDE 220",
            "Canon9000-1": "Canon CanoScan 9000F (Model 1)",
            "Canon9000-2": "Canon CanoScan 9000F (Model 2)",
            "EpsonV370-1": "Epson Perfection V370 (Model 1)",
            "EpsonV370-2": "Epson Perfection V370 (Model 2)",
            "EpsonV39-1": "Epson Perfection V39 (Model 1)",
            "EpsonV39-2": "Epson Perfection V39 (Model 2)",
            "EpsonV550": "Epson Perfection V550",
            "HP": "HP ScanJet"
        }
        
        for scanner, description in supported_scanners.items():
            st.markdown(f"• **{scanner}**")
            st.caption(f"  {description}")
        
        st.markdown("---")
        st.markdown("## 🎯 Analysis Types")
        st.markdown("""
        **Scanner Identification:**
        - Identifies which scanner model created the image
        - Confidence threshold: 60%
        - Unknown scanners classified as "Unknown Scanner"
        
        **Tampered Detection:**
        - Detects if image has been digitally tampered with
        - Binary classification: Original vs Tampered
        - Uses forensic analysis techniques
        """)
        
        st.markdown("---")
        st.markdown("## 📊 Model Information")
        st.markdown("""
        **Scanner Model:**
        - Stacking_Top5_Ridge Ensemble
        - Accuracy: 93.75%
        - Features: 90 (from 153 extracted)
        
        **Tampered Model:**
        - Objective2 Ensemble (best_ensemble_model)
        - Accuracy: ~88.4% (test split)
        - Features: 84 forensic features (exact training order)
        """)

if __name__ == "__main__":
    main()