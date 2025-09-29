#!/usr/bin/env python3
"""
Test script for Scanner Identification only (without TensorFlow dependency)
Tests Objective 1 with images from the dataset to ensure accuracy alignment
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
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
        
        self.confidence_threshold = 0.3  # Threshold for unknown scanner detection
    
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
            
            # 5. Calculate confidence
            confidence = self.calculate_confidence(features_selected)
            
            # 6. Check if unknown scanner
            if confidence < self.confidence_threshold:
                return "Unknown Scanner", confidence
            else:
                return scanner_name, confidence
                
        except Exception as e:
            print(f"❌ Error in scanner prediction: {e}")
            return "Prediction Error", 0.0
    
    def calculate_confidence(self, features):
        """Calculate confidence score using base estimators and meta-learner"""
        try:
            # Simplified confidence calculation - just use meta-learner
            if hasattr(self.model.final_estimator_, 'decision_function'):
                meta_decision = self.model.final_estimator_.decision_function(features)[0]
                # Convert decision function to probability-like score
                meta_confidence = 1.0 / (1.0 + np.exp(-meta_decision))
                return float(meta_confidence)
            else:
                # Default confidence based on prediction certainty
                return 0.7  # Assume good confidence for successful predictions
            
        except Exception as e:
            print(f"❌ Error calculating confidence: {e}")
            return 0.5

class ScannerTester:
    """Test scanner identification with dataset images"""
    
    def __init__(self):
        self.feature_extractor = CombinedFeatureExtractor()
        self.model_loader = ScannerModelLoaderOnly()
        self.results = []
        
        # Dataset paths
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        self.official_path = os.path.join(self.dataset_path, "Official")
        self.wikipedia_path = os.path.join(self.dataset_path, "Wikipedia")
    
    def load_models(self):
        """Load scanner model"""
        print("🔄 Loading scanner model...")
        success = self.model_loader.load_model()
        if success:
            print("✅ Scanner model loaded successfully!")
            return True
        else:
            print("❌ Failed to load scanner model!")
            return False
    
    def test_scanner_images(self, num_images=100):
        """Test scanner identification with dataset images"""
        print(f"\n🔍 Testing Scanner Identification with {num_images} images...")
        
        # Get all scanner images with balanced sampling from each scanner
        scanner_images = []
        
        # Collect images from each scanner type
        all_scanners = []
        
        # Flatfield images
        if os.path.exists(self.flatfield_path):
            for scanner_dir in os.listdir(self.flatfield_path):
                if scanner_dir not in all_scanners:
                    all_scanners.append(scanner_dir)
        
        # Official images
        if os.path.exists(self.official_path):
            for scanner_dir in os.listdir(self.official_path):
                if scanner_dir not in all_scanners:
                    all_scanners.append(scanner_dir)
        
        # Wikipedia images
        if os.path.exists(self.wikipedia_path):
            for scanner_dir in os.listdir(self.wikipedia_path):
                if scanner_dir not in all_scanners:
                    all_scanners.append(scanner_dir)
        
        print(f"Found {len(all_scanners)} scanner types: {all_scanners}")
        
        # Sample evenly from each scanner type
        images_per_scanner = max(1, num_images // len(all_scanners))
        
        for scanner_dir in all_scanners:
            scanner_images_found = []
            
            # Collect from all three sources for this scanner
            sources = [self.flatfield_path, self.official_path, self.wikipedia_path]
            
            for source_path in sources:
                if os.path.exists(source_path):
                    scanner_path = os.path.join(source_path, scanner_dir)
                    if os.path.isdir(scanner_path):
                        for dpi_dir in os.listdir(scanner_path):
                            if dpi_dir in ['150', '300']:
                                dpi_path = os.path.join(scanner_path, dpi_dir)
                                if os.path.isdir(dpi_path):
                                    for img_file in os.listdir(dpi_path):
                                        if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                            img_path = os.path.join(dpi_path, img_file)
                                            scanner_images_found.append((img_path, scanner_dir, dpi_dir))
            
            # Sample from this scanner
            if scanner_images_found:
                import random
                random.shuffle(scanner_images_found)
                selected = scanner_images_found[:images_per_scanner]
                scanner_images.extend(selected)
                print(f"Selected {len(selected)} images from {scanner_dir}")
        
        # If we have fewer images than requested, take more from each scanner
        if len(scanner_images) < num_images:
            for scanner_dir in all_scanners:
                if len(scanner_images) >= num_images:
                    break
                    
                scanner_images_found = []
                sources = [self.flatfield_path, self.official_path, self.wikipedia_path]
                
                for source_path in sources:
                    if os.path.exists(source_path):
                        scanner_path = os.path.join(source_path, scanner_dir)
                        if os.path.isdir(scanner_path):
                            for dpi_dir in os.listdir(scanner_path):
                                if dpi_dir in ['150', '300']:
                                    dpi_path = os.path.join(scanner_path, dpi_dir)
                                    if os.path.isdir(dpi_path):
                                        for img_file in os.listdir(dpi_path):
                                            if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                                img_path = os.path.join(dpi_path, img_file)
                                                scanner_images_found.append((img_path, scanner_dir, dpi_dir))
                
                # Add more images from this scanner
                import random
                random.shuffle(scanner_images_found)
                needed = min(len(scanner_images_found), num_images - len(scanner_images))
                if needed > 0:
                    scanner_images.extend(scanner_images_found[:needed])
        
        # Shuffle the final list and limit
        import random
        random.shuffle(scanner_images)
        if len(scanner_images) > num_images:
            scanner_images = scanner_images[:num_images]
        
        print(f"Found {len(scanner_images)} scanner images for testing")
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (img_path, expected_scanner, dpi) in enumerate(scanner_images):
            try:
                print(f"Testing {i+1}/{len(scanner_images)}: {os.path.basename(img_path)}")
                
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Extract features
                scanner_features = self.feature_extractor.extract_scanner_features(image)
                
                # Predict
                predicted_scanner, confidence = self.model_loader.predict_scanner(scanner_features)
                
                # Check if correct
                is_correct = predicted_scanner == expected_scanner
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                self.results.append({
                    'image_path': img_path,
                    'expected_scanner': expected_scanner,
                    'predicted_scanner': predicted_scanner,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'dpi': dpi
                })
                
                print(f"  Expected: {expected_scanner}, Predicted: {predicted_scanner}, Confidence: {confidence:.3f}, Correct: {is_correct}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n📊 Scanner Identification Results:")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Expected: 93.75%")
        
        return accuracy
    
    def load_image(self, image_path):
        """Load image from file"""
        try:
            import cv2
            import tifffile
            
            if image_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]  # Remove alpha channel
            
            return image
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def save_results(self, filename="scanner_test_results.csv"):
        """Save test results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")
            
            # Summary statistics
            accuracy = df['is_correct'].mean()
            print(f"\n📊 Summary Statistics:")
            print(f"   Scanner ID Accuracy: {accuracy:.1%}")
    
    def run_test(self, num_images=100):
        """Run scanner identification test"""
        print("🧪 AI TraceFinder - Scanner Identification Testing")
        print("=" * 50)
        
        # Load models
        if not self.load_models():
            return False
        
        # Test scanner identification
        accuracy = self.test_scanner_images(num_images)
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎯 Scanner Test Results:")
        print(f"   Accuracy: {accuracy:.1%} (Expected: 93.75%)")
        
        # Check if results are acceptable
        scanner_ok = accuracy >= 0.85  # Allow some tolerance
        
        if scanner_ok:
            print("✅ Scanner Test PASSED - Model is working correctly!")
            return True
        else:
            print("❌ Scanner Test FAILED - Model needs fixes!")
            return False

def main():
    """Main test function"""
    tester = ScannerTester()
    
    # Run scanner test
    success = tester.run_test(num_images=100)
    
    if success:
        print("\n🚀 Scanner identification is ready!")
        sys.exit(0)
    else:
        print("\n🔧 Scanner identification needs debugging!")
        sys.exit(1)

if __name__ == "__main__":
    main()
