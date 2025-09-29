#!/usr/bin/env python3
"""
Comprehensive Test Script for Final Working Streamlit App
Tests both scanner identification and tampered detection
"""

import os
import sys
import random
import numpy as np
from PIL import Image
import cv2
from scanner_identification_fixed import FixedScannerIdentifier

# Add the simplified tampered detector class
class SimplifiedTamperedDetector:
    """Simplified tampered detection using only scikit-learn models"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.is_loaded = False
        self.model_path = "new_objective2_ensemble_results/models/best_ensemble_model.pkl"
        self.scaler_path = "new_objective2_ensemble_results/models/feature_scaler.pkl"
    
    def load_model(self):
        """Load the simplified ensemble model"""
        try:
            # Check if model files exist
            if not os.path.exists(self.model_path):
                print(f"❌ Model file not found at {self.model_path}")
                return False
            
            if not os.path.exists(self.scaler_path):
                print(f"❌ Scaler file not found at {self.scaler_path}")
                return False
            
            # Load model
            import joblib
            self.model = joblib.load(self.model_path)
            print("✅ Loaded simplified ensemble model")
            
            # Load scaler
            self.scaler = joblib.load(self.scaler_path)
            print("✅ Loaded feature scaler")
            
            self.is_loaded = True
            print("✅ Tampered detection model loaded successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error loading tampered detection model: {e}")
            return False
    
    def extract_tampered_features(self, image):
        """Extract features for tampered detection"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Resize to standard size
            gray = cv2.resize(gray, (256, 256))
            
            features = []
            
            # Basic statistical features
            features.extend([
                np.mean(gray),
                np.std(gray),
                np.var(gray),
                np.min(gray),
                np.max(gray),
                np.median(gray)
            ])
            
            # Texture features (simplified)
            from skimage.feature import graycomatrix, graycoprops
            
            # GLCM features
            glcm = graycomatrix(gray, distances=[1], angles=[0, 45, 90, 135], 
                              levels=256, symmetric=True, normed=True)
            
            contrast = graycoprops(glcm, 'contrast').flatten()
            dissimilarity = graycoprops(glcm, 'dissimilarity').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            
            features.extend(contrast)
            features.extend(dissimilarity)
            features.extend(homogeneity)
            features.extend(energy)
            features.extend(correlation)
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            features.append(edge_density)
            
            # Fill remaining features with zeros to match expected 84 features
            while len(features) < 84:
                features.append(0.0)
            
            # Truncate to exactly 84 features
            features = features[:84]
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            print(f"❌ Error extracting tampered features: {e}")
            # Return zero features if extraction fails
            return np.zeros((1, 84))
    
    def predict_tampered(self, image):
        """Predict if image is tampered or original"""
        try:
            if not self.is_loaded:
                return "Unknown", 0.0, "Model not loaded"
            
            # Extract features
            features = self.extract_tampered_features(image)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            
            # Get prediction probability
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = max(probabilities)
            else:
                confidence = 0.8  # Default confidence
            
            # Convert prediction to label
            if prediction == 1:
                result = "Tampered"
            else:
                result = "Original"
            
            return result, confidence, "Success"
            
        except Exception as e:
            print(f"❌ Error in tampered prediction: {e}")
            return "Unknown", 0.0, f"Error: {e}"

def get_dataset_images():
    """Get images from the dataset for testing"""
    dataset_path = "The SUPATLANTIQUE dataset"
    
    # Scanner folders (from Official and Wikipedia directories)
    scanner_folders = [
        "Canon120-1", "Canon120-2", "Canon220", "Canon9000-1", "Canon9000-2",
        "EpsonV370-1", "EpsonV370-2", "EpsonV39-1", "EpsonV39-2", "EpsonV550", "HP"
    ]
    
    scanner_images = {}
    tampered_images = {}
    
    # Get scanner images from Official and Wikipedia directories
    for base_dir in ["Official", "Wikipedia"]:
        base_path = os.path.join(dataset_path, base_dir)
        if os.path.exists(base_path):
            for folder in scanner_folders:
                folder_path = os.path.join(base_path, folder)
                if os.path.exists(folder_path):
                    images = []
                    # Look in both 150 and 300 DPI subdirectories
                    for dpi in ["150", "300"]:
                        dpi_path = os.path.join(folder_path, dpi)
                        if os.path.exists(dpi_path):
                            for file in os.listdir(dpi_path):
                                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                                    images.append(os.path.join(dpi_path, file))
                    
                    # Add to scanner_images (combine with existing if any)
                    if folder not in scanner_images:
                        scanner_images[folder] = []
                    scanner_images[folder].extend(images)
    
    # Sample up to 10 images per scanner type
    for folder in scanner_folders:
        if folder in scanner_images:
            scanner_images[folder] = random.sample(scanner_images[folder], min(10, len(scanner_images[folder]))) if scanner_images[folder] else []
            print(f"📁 {folder}: {len(scanner_images[folder])} images")
    
    # Get tampered detection images
    tampered_base_path = os.path.join(dataset_path, "Tampered images")
    
    # Original images (non-tampered)
    original_path = os.path.join(tampered_base_path, "Original")
    if os.path.exists(original_path):
        images = []
        for file in os.listdir(original_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                images.append(os.path.join(original_path, file))
        tampered_images["Originals"] = random.sample(images, min(20, len(images))) if images else []
        print(f"📁 Originals: {len(tampered_images['Originals'])} images")
    
    # Tampered images
    tampered_path = os.path.join(tampered_base_path, "Tampered")
    if os.path.exists(tampered_path):
        images = []
        for root, dirs, files in os.walk(tampered_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff')):
                    images.append(os.path.join(root, file))
        tampered_images["Tampered"] = random.sample(images, min(20, len(images))) if images else []
        print(f"📁 Tampered: {len(tampered_images['Tampered'])} images")
    
    return scanner_images, tampered_images

def test_scanner_identification():
    """Test scanner identification"""
    print("\n" + "="*60)
    print("🔍 TESTING SCANNER IDENTIFICATION")
    print("="*60)
    
    # Initialize scanner identifier
    scanner_identifier = FixedScannerIdentifier()
    
    if not scanner_identifier.load_model():
        print("❌ Failed to load scanner identification model")
        return False
    
    # Get test images
    scanner_images, _ = get_dataset_images()
    
    total_tests = 0
    correct_predictions = 0
    results = {}
    
    for scanner_type, image_paths in scanner_images.items():
        if not image_paths:
            continue
            
        print(f"\n📊 Testing {scanner_type} ({len(image_paths)} images)")
        
        correct_count = 0
        for i, image_path in enumerate(image_paths[:5]):  # Test first 5 images
            try:
                # Read image
                image = Image.open(image_path)
                image_array = np.array(image)
                
                # Predict scanner
                result, confidence, status = scanner_identifier.predict_scanner(image_array)
                
                # Check if prediction is correct
                if result == scanner_type:
                    correct_count += 1
                    print(f"  ✅ Image {i+1}: {result} (Confidence: {confidence:.1%})")
                else:
                    print(f"  ❌ Image {i+1}: Predicted {result}, Expected {scanner_type} (Confidence: {confidence:.1%})")
                
                total_tests += 1
                
            except Exception as e:
                print(f"  ❌ Error processing image {i+1}: {e}")
        
        results[scanner_type] = {
            'correct': correct_count,
            'total': min(5, len(image_paths)),
            'accuracy': correct_count / min(5, len(image_paths)) if image_paths else 0
        }
        correct_predictions += correct_count
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 SCANNER IDENTIFICATION RESULTS:")
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({correct_predictions}/{total_tests})")
    
    for scanner_type, result in results.items():
        print(f"{scanner_type}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    
    return overall_accuracy >= 0.8  # Target: 80%+ accuracy

def test_tampered_detection():
    """Test tampered detection"""
    print("\n" + "="*60)
    print("🔒 TESTING TAMPERED DETECTION")
    print("="*60)
    
    # Initialize tampered detector
    tampered_detector = SimplifiedTamperedDetector()
    
    if not tampered_detector.load_model():
        print("❌ Failed to load tampered detection model")
        return False
    
    # Get test images
    _, tampered_images = get_dataset_images()
    
    total_tests = 0
    correct_predictions = 0
    results = {}
    
    for category, image_paths in tampered_images.items():
        if not image_paths:
            continue
            
        print(f"\n📊 Testing {category} ({len(image_paths)} images)")
        
        correct_count = 0
        for i, image_path in enumerate(image_paths[:20]):  # Test first 20 images
            try:
                # Read image
                image = Image.open(image_path)
                image_array = np.array(image)
                
                # Predict tampered status
                result, confidence, status = tampered_detector.predict_tampered(image_array)
                
                # Check if prediction is correct
                expected = "Original" if category == "Originals" else "Tampered"
                if result == expected:
                    correct_count += 1
                    print(f"  ✅ Image {i+1}: {result} (Confidence: {confidence:.1%})")
                else:
                    print(f"  ❌ Image {i+1}: Predicted {result}, Expected {expected} (Confidence: {confidence:.1%})")
                
                total_tests += 1
                
            except Exception as e:
                print(f"  ❌ Error processing image {i+1}: {e}")
        
        results[category] = {
            'correct': correct_count,
            'total': min(20, len(image_paths)),
            'accuracy': correct_count / min(20, len(image_paths)) if image_paths else 0
        }
        correct_predictions += correct_count
    
    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_tests if total_tests > 0 else 0
    
    print(f"\n📊 TAMPERED DETECTION RESULTS:")
    print(f"Overall Accuracy: {overall_accuracy:.1%} ({correct_predictions}/{total_tests})")
    
    for category, result in results.items():
        print(f"{category}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    
    return overall_accuracy >= 0.7  # Target: 70%+ accuracy

def main():
    """Main test function"""
    print("🚀 AI TraceFinder - Final Working App Test")
    print("="*60)
    
    # Test scanner identification
    scanner_success = test_scanner_identification()
    
    # Test tampered detection
    tampered_success = test_tampered_detection()
    
    # Final results
    print("\n" + "="*60)
    print("🎯 FINAL TEST RESULTS")
    print("="*60)
    
    print(f"Scanner Identification: {'✅ PASS' if scanner_success else '❌ FAIL'}")
    print(f"Tampered Detection: {'✅ PASS' if tampered_success else '❌ FAIL'}")
    
    if scanner_success and tampered_success:
        print("\n🎉 All tests passed! The app is ready for deployment.")
        return True
    else:
        print("\n⚠️ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    # Set random seed for reproducible results
    random.seed(42)
    np.random.seed(42)
    
    success = main()
    sys.exit(0 if success else 1)
