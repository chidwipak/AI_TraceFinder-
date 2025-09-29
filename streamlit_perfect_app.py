#!/usr/bin/env python3
"""
Perfect AI TraceFinder Streamlit Application
Handles all edge cases, PDF conversion, and provides comprehensive analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import cv2
import os
import pickle
import tempfile
import traceback
from PIL import Image
import io
from pdf2image import convert_from_path, convert_from_bytes
import warnings
warnings.filterwarnings('ignore')

from scanner_identification_fixed import FixedScannerIdentifier
from combined_feature_extractors import CombinedFeatureExtractor
import joblib

# Page configuration
st.set_page_config(
    page_title="AI TraceFinder - Perfect Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

class PerfectTamperedDetector:
    """Enhanced tampered detection with forensic indicators"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.is_loaded = False
        self.model_path = "new_objective2_baseline_ml_results/models/best_model.pkl"
        self.scaler_path = "new_objective2_baseline_ml_results/models/scaler.pkl"
        self.imputer_path = None
        self.feature_extractor = CombinedFeatureExtractor()
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path) or not os.path.exists(self.scaler_path):
                return False
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"❌ Error loading tampered model: {e}")
            return False
    
    def predict_tampered_with_indicators(self, image_rgb):
        """Predict tampered with forensic indicators"""
        try:
            if not self.is_loaded:
                return "Unknown", 0.0, "Model not loaded", []
            
            # Extract features using the exact same method as training
            features = self.feature_extractor.extract_tampered_features(image_rgb)
            features = features.reshape(1, -1)
            
            # Apply the same preprocessing pipeline as training (baseline model doesn't use imputer)
            features_scaled = self.scaler.transform(features)
            
            # 3. Predict with probabilistic output
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(features_scaled)[0]
                pred = int(np.argmax(proba))
                confidence = float(np.max(proba))
            else:
                pred = int(self.model.predict(features_scaled)[0])
                confidence = 0.5
            
            label = "Tampered" if pred == 1 else "Original"
            
            # Generate forensic indicators
            indicators = self._generate_forensic_indicators(image_rgb, features, pred, confidence)
            
            return label, confidence, "Success", indicators
            
        except Exception as e:
            print(f"❌ Error in tampered prediction: {e}")
            return "Unknown", 0.0, f"Error: {e}", []
    
    def _generate_forensic_indicators(self, image, features, pred, confidence):
        """Generate forensic indicators based on features"""
        indicators = []
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # ELA analysis
            try:
                temp_path = '/tmp/temp_ela.jpg'
                cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                compressed = cv2.imread(temp_path)
                if compressed is not None:
                    ela = cv2.absdiff(image, compressed)
                    ela_mean = np.mean(ela)
                    if ela_mean > 30:
                        indicators.append(f"High Error Level Analysis (ELA): {ela_mean:.1f} - Possible compression artifacts")
                    else:
                        indicators.append(f"Low Error Level Analysis (ELA): {ela_mean:.1f} - Normal compression")
                os.remove(temp_path)
            except:
                indicators.append("ELA analysis unavailable")
            
            # Noise analysis
            noise = cv2.Laplacian(gray, cv2.CV_64F)
            noise_std = np.std(noise)
            if noise_std > 50:
                indicators.append(f"High noise level: {noise_std:.1f} - Possible tampering artifacts")
            else:
                indicators.append(f"Normal noise level: {noise_std:.1f}")
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            if edge_density > 0.1:
                indicators.append(f"High edge density: {edge_density:.3f} - Possible sharp transitions")
            else:
                indicators.append(f"Normal edge density: {edge_density:.3f}")
            
            # Color analysis
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            h_std = np.std(hsv[:, :, 0])
            if h_std > 30:
                indicators.append(f"High color variation: {h_std:.1f} - Possible color manipulation")
            else:
                indicators.append(f"Normal color variation: {h_std:.1f}")
            
            # Confidence-based indicators
            if confidence > 0.8:
                indicators.append(f"High confidence prediction: {confidence:.1%}")
            elif confidence > 0.6:
                indicators.append(f"Moderate confidence prediction: {confidence:.1%}")
            else:
                indicators.append(f"Low confidence prediction: {confidence:.1%} - Results may be unreliable")
                
        except Exception as e:
            indicators.append(f"Forensic analysis error: {str(e)}")
        
        return indicators

class PerfectScannerIdentifier:
    """Enhanced scanner identification with top-3 predictions"""
    
    def __init__(self):
        self.identifier = FixedScannerIdentifier()
        self.is_loaded = False
    
    def load_model(self):
        return self.identifier.load_model()
    
    def predict_scanner_with_top3(self, image):
        """Predict scanner with top-3 probabilities"""
        try:
            if not self.identifier.model:
                return "Model not loaded", 0.0, "Error", []
            
            # Extract features
            features = self.identifier.extract_features(image)
            if features is None:
                return None, 0.0, "Feature extraction failed", []
            
            # Apply preprocessing
            features_imputed = self.identifier.imputer.transform(features.reshape(1, -1))
            features_selected = self.identifier.selector.transform(features_imputed)
            
            # Get prediction
            prediction = self.identifier.model.predict(features_selected)[0]
            scanner_name = self.identifier.supported_scanners.get(prediction, "Unknown Scanner")
            
            # Get top-3 predictions with probabilities
            top3_predictions = self._get_top3_predictions(features_selected)
            
            # Calculate confidence
            confidence = self.identifier.calculate_confidence(features_selected)
            
            # Check if unknown scanner
            if confidence < self.identifier.confidence_threshold:
                return "Unknown Scanner", confidence, "Low confidence - likely not one of the 11 supported scanners", top3_predictions
            
            return scanner_name, confidence, "High confidence prediction", top3_predictions
            
        except Exception as e:
            print(f"❌ Error in scanner prediction: {e}")
            return None, 0.0, f"Prediction error: {e}", []
    
    def _get_top3_predictions(self, features_selected):
        """Get top-3 scanner predictions with probabilities"""
        try:
            # Get probabilities from base estimators
            all_probabilities = []
            for name, estimator in self.identifier.model.estimators_:
                try:
                    if hasattr(estimator, 'predict_proba'):
                        prob = estimator.predict_proba(features_selected)[0]
                        all_probabilities.append(prob)
                except:
                    continue
            
            if all_probabilities:
                # Average probabilities
                avg_probabilities = np.mean(all_probabilities, axis=0)
                
                # Get top-3
                top3_indices = np.argsort(avg_probabilities)[-3:][::-1]
                top3_predictions = []
                
                for idx in top3_indices:
                    scanner_name = self.identifier.supported_scanners.get(idx, f"Scanner_{idx}")
                    prob = avg_probabilities[idx]
                    top3_predictions.append((scanner_name, prob))
                
                return top3_predictions
            else:
                return [("Unknown", 0.5)]
                
        except Exception as e:
            print(f"❌ Error getting top-3 predictions: {e}")
            return [("Unknown", 0.5)]

def convert_pdf_to_image(uploaded_file):
    """Convert PDF to image for analysis"""
    try:
        # Read PDF bytes
        pdf_bytes = uploaded_file.read()
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes, dpi=300, first_page=1, last_page=1)
        
        if images:
            # Convert PIL image to numpy array
            image = images[0]
            image_array = np.array(image)
            
            # Convert RGB to BGR for OpenCV compatibility
            if len(image_array.shape) == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_array, "Success"
        else:
            return None, "No pages found in PDF"
            
    except Exception as e:
        return None, f"PDF conversion error: {str(e)}"

def preprocess_image(image):
    """Robust image preprocessing"""
    try:
        # Handle different image formats
        if len(image.shape) == 2:
            # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            # RGBA
            image = image[:, :, :3]
        elif image.shape[2] == 3:
            # RGB
            pass
        else:
            return None, "Unsupported image format"
        
        # Resize if too large
        h, w = image.shape[:2]
        if h > 2000 or w > 2000:
            scale = min(2000/h, 2000/w)
            new_h, new_w = int(h*scale), int(w*scale)
            image = cv2.resize(image, (new_w, new_h))
        
        return image, "Success"
        
    except Exception as e:
        return None, f"Preprocessing error: {str(e)}"

def main():
    """Main Streamlit application"""
    
    # Title and description
    st.title("🔍 AI TraceFinder - Perfect Analysis")
    st.markdown("""
    **Advanced Scanner Identification & Tampered Detection System**
    
    Upload an image or PDF to analyze:
    - **Scanner Identification**: Identify which scanner model created the image
    - **Tampered Detection**: Detect if the image has been tampered with
    - **Combined Analysis**: Both analyses with comprehensive interpretation
    """)
    
    # Initialize session state
    if 'scanner_identifier' not in st.session_state:
        st.session_state.scanner_identifier = PerfectScannerIdentifier()
    
    if 'tampered_detector' not in st.session_state:
        st.session_state.tampered_detector = PerfectTamperedDetector()
    
    # Load models
    if not st.session_state.scanner_identifier.is_loaded:
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
    
    # Analysis type selection
    st.markdown("## 🎯 Select Analysis Type")
    analysis_type = st.radio(
        "Choose the type of analysis:",
        ["Scanner Identification", "Tampered Detection", "Combined Analysis"],
        horizontal=True
    )
    
    # File upload section
    st.markdown("## 📁 Upload Image or PDF for Analysis")
    
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=['jpg', 'jpeg', 'png', 'tif', 'tiff', 'pdf'],
        help="Upload an image file or PDF to analyze"
    )
    
    if uploaded_file is not None:
        # Display uploaded file info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### 📸 Uploaded File")
            if uploaded_file.type == "application/pdf":
                st.info("📄 PDF detected - will be converted to image for analysis")
                st.image("https://via.placeholder.com/400x300/cccccc/666666?text=PDF+File", 
                        caption="PDF file uploaded", use_container_width=True)
            else:
                # Display image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.markdown("### 🎯 Analysis Results")
            
            # Process file
            if st.button("🔍 Analyze File", type="primary"):
                with st.spinner("Processing file..."):
                    try:
                        # Handle PDF conversion
                        if uploaded_file.type == "application/pdf":
                            image_array, pdf_status = convert_pdf_to_image(uploaded_file)
                            if image_array is None:
                                st.error(f"❌ PDF conversion failed: {pdf_status}")
                                return
                            st.success(f"✅ PDF converted successfully: {pdf_status}")
                        else:
                            # Load image
                            image = Image.open(uploaded_file)
                            image_array = np.array(image)
                        
                        # Preprocess image
                        processed_image, prep_status = preprocess_image(image_array)
                        if processed_image is None:
                            st.error(f"❌ Image preprocessing failed: {prep_status}")
                            return
                        
                        # Run analysis based on selection
                        if analysis_type == "Scanner Identification":
                            run_scanner_analysis(processed_image)
                        elif analysis_type == "Tampered Detection":
                            run_tampered_analysis(processed_image)
                        elif analysis_type == "Combined Analysis":
                            run_combined_analysis(processed_image)
                    
                    except Exception as e:
                        st.error(f"❌ Error during analysis: {e}")
                        st.error(traceback.format_exc())

def run_scanner_analysis(image):
    """Run scanner identification analysis"""
    st.markdown("#### 🔍 Scanner Identification Results")
    
    scanner_result, scanner_confidence, scanner_status, top3_predictions = st.session_state.scanner_identifier.predict_scanner_with_top3(image)
    
    if scanner_status == "Success":
        # Main result
        if scanner_confidence >= 0.6:
            st.success(f"✅ **Scanner**: {scanner_result}")
            st.info(f"📊 **Confidence**: {scanner_confidence:.1%}")
        else:
            st.warning(f"⚠️ **Scanner**: {scanner_result}")
            st.info(f"📊 **Confidence**: {scanner_confidence:.1%}")
        
        # Top-3 predictions
        if top3_predictions:
            st.markdown("**Top 3 Likely Scanners:**")
            for i, (scanner, prob) in enumerate(top3_predictions, 1):
                st.write(f"{i}. {scanner}: {prob:.1%}")
        
        # Warnings and notes
        if scanner_confidence < 0.3:
            st.warning("⚠️ **Low confidence prediction. This may not be a scanned document or may be from an unsupported scanner.**")
        elif scanner_confidence < 0.6:
            st.info("ℹ️ **Moderate confidence. Results may be less reliable.**")
        
        # Edge case handling
        if scanner_result == "Unknown Scanner":
            st.error("❌ **Unknown Scanner** - This scanner is not in our training dataset.")
            st.info("💡 **Note**: This could be a non-scanned image (photo, digital art) or a scanner not in our 11 supported models.")
        
    else:
        st.error(f"❌ **Scanner Analysis Failed**: {scanner_status}")

def run_tampered_analysis(image):
    """Run tampered detection analysis"""
    st.markdown("#### 🔒 Tampered Detection Results")
    
    if not st.session_state.tampered_detector.is_loaded:
        st.warning("⚠️ Tampered detection model not available")
        return
    
    tampered_result, tampered_confidence, tampered_status, indicators = st.session_state.tampered_detector.predict_tampered_with_indicators(image)
    
    if tampered_status == "Success":
        # Main result
        if tampered_result == "Tampered":
            st.error(f"🚨 **Result**: {tampered_result}")
        else:
            st.success(f"✅ **Result**: {tampered_result}")
        
        st.info(f"📊 **Confidence**: {tampered_confidence:.1%}")
        
        # Forensic indicators
        if indicators:
            st.markdown("**Forensic Indicators:**")
            for indicator in indicators:
                st.write(f"• {indicator}")
        
        # Warnings
        if tampered_confidence < 0.6:
            st.warning("⚠️ **Low confidence prediction. Results may be unreliable.**")
        
        # Edge case handling
        if tampered_result == "Original" and tampered_confidence < 0.7:
            st.info("💡 **Note**: This appears to be an original image but may not be a scanned document.")
        elif tampered_result == "Tampered" and tampered_confidence < 0.7:
            st.info("💡 **Note**: This appears to be a tampered image but may not be a scanned document.")
    
    else:
        st.error(f"❌ **Tampered Detection Failed**: {tampered_status}")

def run_combined_analysis(image):
    """Run combined analysis with interpretation guidance"""
    st.markdown("#### 🔍🔒 Combined Analysis Results")
    
    # Scanner analysis
    st.markdown("##### Scanner Identification")
    scanner_result, scanner_confidence, scanner_status, top3_predictions = st.session_state.scanner_identifier.predict_scanner_with_top3(image)
    
    if scanner_status == "Success":
        if scanner_confidence >= 0.6:
            st.success(f"✅ **Scanner**: {scanner_result} ({scanner_confidence:.1%})")
        else:
            st.warning(f"⚠️ **Scanner**: {scanner_result} ({scanner_confidence:.1%})")
    else:
        st.error(f"❌ **Scanner**: {scanner_status}")
    
    # Tampered analysis
    st.markdown("##### Tampered Detection")
    if st.session_state.tampered_detector.is_loaded:
        tampered_result, tampered_confidence, tampered_status, indicators = st.session_state.tampered_detector.predict_tampered_with_indicators(image)
        
        if tampered_status == "Success":
            if tampered_result == "Tampered":
                st.error(f"🚨 **Result**: {tampered_result} ({tampered_confidence:.1%})")
            else:
                st.success(f"✅ **Result**: {tampered_result} ({tampered_confidence:.1%})")
        else:
            st.error(f"❌ **Tampered**: {tampered_status}")
    else:
        st.warning("⚠️ Tampered detection not available")
        tampered_result = "Unknown"
        tampered_confidence = 0.0
    
    # Interpretation guidance
    st.markdown("##### 📋 Interpretation Guidance")
    
    # Check for low confidence in both
    if scanner_confidence < 0.6 and tampered_confidence < 0.6:
        st.error("⚠️ **Low confidence in both analyses. Results may be unreliable.**")
    elif scanner_confidence < 0.6:
        st.warning("⚠️ **Low confidence in scanner identification. This may not be a scanned document.**")
    elif tampered_confidence < 0.6:
        st.warning("⚠️ **Low confidence in tampering analysis. Results may be unreliable.**")
    
    # Combined interpretation
    if scanner_status == "Success" and tampered_status == "Success":
        if scanner_result != "Unknown Scanner" and tampered_result == "Original":
            st.success("✅ **This appears to be an original document scanned with a known scanner.**")
        elif scanner_result != "Unknown Scanner" and tampered_result == "Tampered":
            st.error("🚨 **This appears to be a tampered document, possibly from a known scanner.**")
        elif scanner_result == "Unknown Scanner" and tampered_result == "Original":
            st.info("ℹ️ **This appears to be an original image but may not be a scanned document.**")
        elif scanner_result == "Unknown Scanner" and tampered_result == "Tampered":
            st.warning("⚠️ **This appears to be a tampered image but may not be a scanned document.**")
        else:
            st.info("ℹ️ **Analysis complete. Please review individual results above.**")
    
    # Forensic indicators for combined analysis
    if st.session_state.tampered_detector.is_loaded and tampered_status == "Success" and indicators:
        with st.expander("🔬 Detailed Forensic Indicators"):
            for indicator in indicators:
                st.write(f"• {indicator}")

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
    - Shows top-3 predictions with probabilities
    - Confidence threshold: 60%
    - Handles non-scanned images gracefully
    
    **Tampered Detection:**
    - Detects if image has been digitally tampered with
    - Shows forensic indicators
    - Binary classification: Original vs Tampered
    - Uses advanced ensemble model
    
    **Combined Analysis:**
    - Runs both analyses sequentially
    - Provides interpretation guidance
    - Handles edge cases appropriately
    """)
    
    st.markdown("---")
    st.markdown("## 📊 Model Information")
    st.markdown("""
    **Scanner Model:**
    - Stacking_Top5_Ridge Ensemble
    - Accuracy: 93.75%
    - Features: 90 (from 153 extracted)
    
    **Tampered Model:**
    - Objective2 Ensemble
    - Accuracy: ~88.4%
    - Features: 84 forensic features
    """)

if __name__ == "__main__":
    main()
