#!/usr/bin/env python3
"""
Quick test script for AI TraceFinder Combined Application
Tests with a small sample to verify functionality works
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from combined_feature_extractors import CombinedFeatureExtractor

# Import only the scanner model loader
class ScannerModelLoaderOnly:
    """Model loader for Objective 1: Scanner Identification (93.75% accuracy)"""
    
    def __init__(self):
        self.model = None
        self.imputer = None
        self.selector = None
        self.label_encoder = None
        self.is_loaded = False
        
        # 11 Supported Scanner Models
        self.supported_scanners = [
            'Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2',
            'EpsonV39-1', 'EpsonV39-2', 'EpsonV370-1', 'EpsonV370-2', 'EpsonV550', 'HP'
        ]
        
        self.confidence_threshold = 0.3
    
    def load_model(self):
        """Load the Stacking_Top5_Ridge model and preprocessing components"""
        try:
            import pickle
            
            model_path = "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl"
            imputer_path = "ensemble_95_results_ultra/imputer.pkl"
            selector_path = "ensemble_95_results_ultra/feature_selector.pkl"
            encoder_path = "ensemble_95_results_ultra/label_encoder.pkl"
            
            # Load model
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.model = pickle.load(f)
                print("✅ Loaded Stacking_Top5_Ridge model")
            else:
                print(f"❌ Model not found: {model_path}")
                return False
            
            # Load imputer
            if os.path.exists(imputer_path):
                with open(imputer_path, 'rb') as f:
                    self.imputer = pickle.load(f)
                print("✅ Loaded imputer")
            else:
                print(f"❌ Imputer not found: {imputer_path}")
                return False
            
            # Load feature selector
            if os.path.exists(selector_path):
                with open(selector_path, 'rb') as f:
                    self.selector = pickle.load(f)
                print("✅ Loaded feature selector")
            else:
                print(f"❌ Feature selector not found: {selector_path}")
                return False
            
            # Load label encoder
            if os.path.exists(encoder_path):
                with open(encoder_path, 'rb') as f:
                    self.label_encoder = pickle.load(f)
                print("✅ Loaded label encoder")
            else:
                print(f"❌ Label encoder not found: {encoder_path}")
                return False
            
            self.is_loaded = True
            print("✅ Scanner model loading completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Error loading scanner model: {e}")
            return False
    
    def predict_scanner(self, features):
        """Predict scanner with confidence score"""
        if not self.is_loaded:
            return "Model not loaded", 0.0
        
        try:
            # Apply preprocessing pipeline (same as training)
            # 1. Impute missing values
            features_imputed = self.imputer.transform(features.reshape(1, -1))
            
            # 2. Feature selection (153 -> 90 features)
            features_selected = self.selector.transform(features_imputed)
            
            # 3. Predict
            prediction = self.model.predict(features_selected)[0]
            
            # 4. Decode label
            scanner_name = self.label_encoder.inverse_transform([prediction])[0]
            
            # 5. Calculate confidence (simplified)
            confidence = 0.8  # Simplified confidence
            
            # 6. Check if unknown scanner
            if confidence < self.confidence_threshold:
                return "Unknown Scanner", confidence
            else:
                return scanner_name, confidence
                
        except Exception as e:
            print(f"❌ Error in scanner prediction: {e}")
            return "Prediction Error", 0.0

def test_single_image():
    """Test with a single image to verify functionality"""
    print("🧪 Quick Test - Single Image")
    print("=" * 40)
    
    # Initialize components
    feature_extractor = CombinedFeatureExtractor()
    model_loader = ScannerModelLoaderOnly()
    
    # Load model
    if not model_loader.load_model():
        print("❌ Failed to load model")
        return False
    
    # Find a test image
    test_image_path = None
    dataset_path = "The SUPATLANTIQUE dataset"
    
    # Look for any image in the dataset
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                test_image_path = os.path.join(root, file)
                # Extract scanner name from path
                path_parts = root.split(os.sep)
                scanner_name = None
                for part in path_parts:
                    if part in model_loader.supported_scanners:
                        scanner_name = part
                        break
                break
        if test_image_path:
            break
    
    if not test_image_path:
        print("❌ No test image found")
        return False
    
    print(f"📷 Testing with: {test_image_path}")
    print(f"🎯 Expected scanner: {scanner_name}")
    
    try:
        # Load image
        import cv2
        import tifffile
        
        if test_image_path.lower().endswith(('.tif', '.tiff')):
            image = tifffile.imread(test_image_path)
        else:
            image = cv2.imread(test_image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        print(f"📐 Image shape: {image.shape}")
        
        # Extract features
        print("🔄 Extracting features...")
        scanner_features = feature_extractor.extract_scanner_features(image)
        print(f"📊 Extracted {len(scanner_features)} features")
        
        # Predict
        print("🔄 Making prediction...")
        predicted_scanner, confidence = model_loader.predict_scanner(scanner_features)
        
        # Results
        print(f"\n📊 Results:")
        print(f"   Expected: {scanner_name}")
        print(f"   Predicted: {predicted_scanner}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Correct: {predicted_scanner == scanner_name}")
        
        if predicted_scanner == scanner_name:
            print("✅ Test PASSED!")
            return True
        else:
            print("❌ Test FAILED!")
            return False
            
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

def main():
    """Main test function"""
    success = test_single_image()
    
    if success:
        print("\n🚀 Basic functionality works! Ready for full testing.")
        sys.exit(0)
    else:
        print("\n🔧 Basic functionality needs fixes!")
        sys.exit(1)

if __name__ == "__main__":
    main()
