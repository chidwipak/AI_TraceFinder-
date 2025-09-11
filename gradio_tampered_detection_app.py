#!/usr/bin/env python3
"""
Gradio Application for Tampered Image Detection
Alternative to Streamlit with better deployment compatibility
"""

import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import json
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Try to import OpenCV with fallback handling for deployment
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    # Try headless version for deployment environments
    try:
        import cv2
        CV2_AVAILABLE = True
    except ImportError:
        CV2_AVAILABLE = False
        print("⚠️ OpenCV not available. Some features may be limited.")

# Import our custom feature extraction classes
import sys
# Add current directory to path for deployment compatibility
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

class TamperedImageDetector:
    """Main class for tampered image detection using the best model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_extractor = None
        self.feature_indices = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the best performing model and preprocessing components"""
        try:
            # Load the robust ensemble model (much better performance on original images)
            model_path = 'objective2_robust_models/ensemble_robust.joblib'
            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                print("✅ Loaded robust ensemble model (83.3% overall accuracy)")
            else:
                # Fallback to baseline model
                model_path = 'objective2_baseline_models/lightgbm.joblib'
                if os.path.exists(model_path):
                    self.model = joblib.load(model_path)
                    print("⚠️ Using baseline model (biased towards tampered)")
                else:
                    print("❌ No model found")
                    return False
            
            # Load scaler
            scaler_path = 'objective2_robust_models/scaler_robust.joblib'
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
            else:
                # Fallback to baseline scaler
                scaler_path = 'objective2_baseline_models/scaler.joblib'
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
            
            # Load feature indices
            feature_info_path = 'objective2_robust_models/feature_info_robust.json'
            if os.path.exists(feature_info_path):
                with open(feature_info_path, 'r') as f:
                    feature_info = json.load(f)
                    self.feature_indices = feature_info.get('selected_features', list(range(105)))
            else:
                # Fallback to baseline feature info
                feature_info_path = 'objective2_baseline_models/feature_info.json'
                if os.path.exists(feature_info_path):
                    with open(feature_info_path, 'r') as f:
                        feature_info = json.load(f)
                        self.feature_indices = feature_info.get('selected_features', list(range(105)))
            
            # Initialize feature extractor
            if CV2_AVAILABLE:
                from objective2_feature_extraction import ForensicFeatureExtractor
                self.feature_extractor = ForensicFeatureExtractor(target_size=(224, 224))
            else:
                print("Cannot initialize feature extractor without OpenCV")
                return False
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def extract_features(self, image_path):
        """Extract features from an image"""
        try:
            features = self.feature_extractor.extract_all_features(image_path)
            
            # Convert to array and select relevant features
            feature_cols = [col for col in features.keys() if col not in ['image_path', 'label', 'tamper_type', 'category']]
            feature_values = [features[col] for col in feature_cols]
            feature_array = np.array(feature_values).reshape(1, -1)
            
            # Handle infinite and NaN values
            feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=1e6, neginf=-1e6)
            
            return feature_array
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def preprocess_features(self, features):
        """Preprocess features using the same pipeline as training"""
        try:
            # Select the exact same features used during training first
            if self.feature_indices:
                # Use the exact feature indices from training
                features_selected = features[:, self.feature_indices]
            else:
                # Fallback: use first 30 features if indices not available
                features_selected = features[:, :30]
            
            # Apply robust scaling
            if self.scaler:
                features_scaled = self.scaler.transform(features_selected)
            else:
                features_scaled = features_selected
            
            return features_scaled
            
        except Exception as e:
            print(f"Error preprocessing features: {e}")
            return None
    
    def predict(self, image_path):
        """Make prediction on an image"""
        if not self.is_loaded:
            return None, None, None
        
        try:
            # Extract features
            features = self.extract_features(image_path)
            if features is None:
                return None, None, None
            
            # Preprocess features
            features_processed = self.preprocess_features(features)
            if features_processed is None:
                return None, None, None
            
            # Make prediction
            prediction = self.model.predict(features_processed)[0]
            probability = self.model.predict_proba(features_processed)[0]
            
            return prediction, probability, features_processed
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None, None, None

# Initialize detector
detector = TamperedImageDetector()
detector.load_model()

def analyze_image(image):
    """Analyze uploaded image and return results"""
    if image is None:
        return "Please upload an image", None, None
    
    try:
        # Save image temporarily
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Make prediction
        prediction, probability, features = detector.predict(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        if prediction is not None:
            # Format results
            if prediction == 1:
                result_text = "🚨 TAMPERED IMAGE DETECTED"
                result_color = "red"
            else:
                result_text = "✅ ORIGINAL IMAGE"
                result_color = "green"
            
            # Create confidence visualization
            fig, ax = plt.subplots(figsize=(8, 4))
            categories = ['Original', 'Tampered']
            values = [probability[0], probability[1]]
            colors = ['green', 'red']
            
            bars = ax.bar(categories, values, color=colors, alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_ylabel('Confidence')
            ax.set_title('Prediction Confidence')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Create detailed results
            details = f"""
            **Analysis Results:**
            
            **Prediction:** {result_text}
            
            **Confidence Scores:**
            - Original: {probability[0]*100:.1f}%
            - Tampered: {probability[1]*100:.1f}%
            
            **Model Information:**
            - Model: Robust Ensemble (5 algorithms)
            - Overall Accuracy: 83.3%
            - Features Used: 30 forensic features
            - Training Data: SUPATLANTIQUE dataset (136 images)
            """
            
            return result_text, details, fig
            
        else:
            return "❌ Failed to analyze image. Please try again.", None, None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None, None

# Create Gradio interface
def create_interface():
    """Create the Gradio interface"""
    
    with gr.Blocks(title="Tampered Image Detection System", theme=gr.themes.Soft()) as interface:
        gr.Markdown("""
        # 🔍 Tampered Image Detection System
        **Advanced Forensic Image Analysis using Machine Learning**
        
        This system uses a robust ensemble model to detect tampered images with 83.3% overall accuracy.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📁 Upload Image")
                image_input = gr.Image(
                    type="pil",
                    label="Upload an image to analyze",
                    height=300
                )
                
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")
                
                gr.Markdown("""
                ### 📊 Model Performance
                - **Overall Accuracy**: 83.3%
                - **Original Images**: 66.7% accuracy
                - **Tampered Images**: 100% accuracy
                - **Features**: 30 forensic features
                - **Training Data**: 136 images
                """)
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Detection Results")
                result_text = gr.Textbox(
                    label="Prediction Result",
                    interactive=False,
                    height=100
                )
                
                result_details = gr.Markdown(label="Detailed Results")
                
                confidence_plot = gr.Plot(label="Confidence Visualization")
        
        # Event handlers
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input],
            outputs=[result_text, result_details, confidence_plot]
        )
        
        # Sample images section
        gr.Markdown("### 🧪 Test with Sample Images")
        
        sample_images = {
            "Original Image": "The SUPATLANTIQUE dataset/Tampered images/Original/s1_70.tif",
            "Copy-Move Tampered": "The SUPATLANTIQUE dataset/Tampered images/Tampered/Copy-move/s1_70_b.tif",
            "Retouching Tampered": "The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching/s1_70_c.tif",
            "Splicing Tampered": "The SUPATLANTIQUE dataset/Tampered images/Tampered/Splicing/s1_70_a.tif"
        }
        
        with gr.Row():
            for name, path in sample_images.items():
                if os.path.exists(path):
                    gr.Examples(
                        examples=[[path]],
                        inputs=[image_input],
                        label=name
                    )
        
        # About section
        gr.Markdown("""
        ### ℹ️ About This Application
        
        **How It Works:**
        1. **Feature Extraction**: Extracts 105 forensic features including Error Level Analysis (ELA), noise analysis, compression artifacts, frequency domain features, texture analysis, and statistical features
        2. **Preprocessing**: Applies robust scaling and feature selection (30 most relevant features)
        3. **Prediction**: Uses Robust Ensemble model combining 5 algorithms (RandomForest, LightGBM, XGBoost, SVM, LogisticRegression)
        
        **Model Performance:**
        - **Overall Accuracy**: 83.3% on test set
        - **Original Images**: 66.7% accuracy (correctly identifies original images)
        - **Tampered Images**: 100% accuracy (perfect detection of tampered images)
        - **Training Data**: 136 images (102 tampered, 34 original)
        - **Tampering Types**: Copy-move, Retouching, Splicing
        
        **Limitations:**
        - Model accuracy is limited by small training dataset
        - May not detect very sophisticated tampering
        - Results should be used as a preliminary analysis tool
        """)
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
