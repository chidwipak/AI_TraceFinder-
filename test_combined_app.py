#!/usr/bin/env python3
"""
Test script for AI TraceFinder Combined Application
Tests both objectives with images from the dataset to ensure accuracy alignment
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
from combined_model_loaders import CombinedModelManager

class CombinedAppTester:
    """Test the combined application with dataset images"""
    
    def __init__(self):
        self.feature_extractor = CombinedFeatureExtractor()
        self.model_manager = CombinedModelManager()
        self.results = []
        
        # Dataset paths
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        self.official_path = os.path.join(self.dataset_path, "Official")
        self.wikipedia_path = os.path.join(self.dataset_path, "Wikipedia")
        self.tampered_path = os.path.join(self.dataset_path, "Tampered images")
        self.originals_path = os.path.join(self.dataset_path, "Originals")
    
    def load_models(self):
        """Load all models"""
        print("🔄 Loading models...")
        success = self.model_manager.load_all_models()
        if success:
            print("✅ Models loaded successfully!")
            return True
        else:
            print("❌ Failed to load models!")
            return False
    
    def test_scanner_images(self, num_images=100):
        """Test scanner identification with dataset images"""
        print(f"\n🔍 Testing Scanner Identification with {num_images} images...")
        
        # Get all scanner images
        scanner_images = []
        
        # Flatfield images
        for scanner_dir in os.listdir(self.flatfield_path):
            scanner_path = os.path.join(self.flatfield_path, scanner_dir)
            if os.path.isdir(scanner_path):
                for dpi_dir in os.listdir(scanner_path):
                    if dpi_dir in ['150', '300']:
                        dpi_path = os.path.join(scanner_path, dpi_dir)
                        if os.path.isdir(dpi_path):
                            for img_file in os.listdir(dpi_path):
                                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                    img_path = os.path.join(dpi_path, img_file)
                                    scanner_images.append((img_path, scanner_dir, dpi_dir))
        
        # Official images
        for scanner_dir in os.listdir(self.official_path):
            scanner_path = os.path.join(self.official_path, scanner_dir)
            if os.path.isdir(scanner_path):
                for dpi_dir in os.listdir(scanner_path):
                    if dpi_dir in ['150', '300']:
                        dpi_path = os.path.join(scanner_path, dpi_dir)
                        if os.path.isdir(dpi_path):
                            for img_file in os.listdir(dpi_path):
                                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                    img_path = os.path.join(dpi_path, img_file)
                                    scanner_images.append((img_path, scanner_dir, dpi_dir))
        
        # Wikipedia images
        for scanner_dir in os.listdir(self.wikipedia_path):
            scanner_path = os.path.join(self.wikipedia_path, scanner_dir)
            if os.path.isdir(scanner_path):
                for dpi_dir in os.listdir(scanner_path):
                    if dpi_dir in ['150', '300']:
                        dpi_path = os.path.join(scanner_path, dpi_dir)
                        if os.path.isdir(dpi_path):
                            for img_file in os.listdir(dpi_path):
                                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                    img_path = os.path.join(dpi_path, img_file)
                                    scanner_images.append((img_path, scanner_dir, dpi_dir))
        
        # Limit to requested number
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
                predicted_scanner, confidence = self.model_manager.scanner_loader.predict_scanner(scanner_features)
                
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
                    'objective': 'Scanner ID',
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
    
    def test_tampered_images(self, num_images=100):
        """Test tampered detection with dataset images"""
        print(f"\n🛡️ Testing Tampered Detection with {num_images} images...")
        
        # Get tampered images
        tampered_images = []
        original_images = []
        
        # Tampered images
        tampered_dir = os.path.join(self.tampered_path, "Tampered")
        if os.path.exists(tampered_dir):
            for tamper_type in os.listdir(tampered_dir):
                tamper_path = os.path.join(tampered_dir, tamper_type)
                if os.path.isdir(tamper_path):
                    for img_file in os.listdir(tamper_path):
                        if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                            img_path = os.path.join(tamper_path, img_file)
                            tampered_images.append((img_path, 1, tamper_type))  # 1 = tampered
        
        # Original images
        original_dir = os.path.join(self.tampered_path, "Original")
        if os.path.exists(original_dir):
            for img_file in os.listdir(original_dir):
                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                    img_path = os.path.join(original_dir, img_file)
                    original_images.append((img_path, 0, "none"))  # 0 = original
        
        # Combine and limit
        all_images = tampered_images + original_images
        if len(all_images) > num_images:
            all_images = all_images[:num_images]
        
        print(f"Found {len(all_images)} images for testing ({len(tampered_images)} tampered, {len(original_images)} original)")
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (img_path, expected_label, tamper_type) in enumerate(all_images):
            try:
                print(f"Testing {i+1}/{len(all_images)}: {os.path.basename(img_path)}")
                
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Extract features
                tampered_features = self.feature_extractor.extract_tampered_features(image)
                
                # Predict
                predicted_result, confidence, predicted_tamper_type = self.model_manager.tampered_loader.predict_tampered(tampered_features)
                
                # Convert prediction to label
                predicted_label = 1 if predicted_result == "Tampered" else 0
                
                # Check if correct
                is_correct = predicted_label == expected_label
                if is_correct:
                    correct_predictions += 1
                total_predictions += 1
                
                # Store result
                self.results.append({
                    'image_path': img_path,
                    'expected_label': expected_label,
                    'predicted_label': predicted_label,
                    'predicted_result': predicted_result,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'objective': 'Tampered Detection',
                    'tamper_type': tamper_type
                })
                
                print(f"  Expected: {'Tampered' if expected_label == 1 else 'Original'}, Predicted: {predicted_result}, Confidence: {confidence:.3f}, Correct: {is_correct}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n📊 Tampered Detection Results:")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Expected: 89.86%")
        
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
    
    def save_results(self, filename="test_results.csv"):
        """Save test results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")
            
            # Summary statistics
            scanner_results = df[df['objective'] == 'Scanner ID']
            tampered_results = df[df['objective'] == 'Tampered Detection']
            
            print(f"\n📊 Summary Statistics:")
            if len(scanner_results) > 0:
                scanner_accuracy = scanner_results['is_correct'].mean()
                print(f"   Scanner ID Accuracy: {scanner_accuracy:.1%}")
            
            if len(tampered_results) > 0:
                tampered_accuracy = tampered_results['is_correct'].mean()
                print(f"   Tampered Detection Accuracy: {tampered_accuracy:.1%}")
    
    def run_comprehensive_test(self, scanner_images=100, tampered_images=100):
        """Run comprehensive test on both objectives"""
        print("🧪 AI TraceFinder - Comprehensive Testing")
        print("=" * 50)
        
        # Load models
        if not self.load_models():
            return False
        
        # Test scanner identification
        scanner_accuracy = self.test_scanner_images(scanner_images)
        
        # Test tampered detection
        tampered_accuracy = self.test_tampered_images(tampered_images)
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎯 Final Test Results:")
        print(f"   Scanner ID: {scanner_accuracy:.1%} (Expected: 93.75%)")
        print(f"   Tampered Detection: {tampered_accuracy:.1%} (Expected: 89.86%)")
        
        # Check if results are acceptable
        scanner_ok = scanner_accuracy >= 0.85  # Allow some tolerance
        tampered_ok = tampered_accuracy >= 0.80  # Allow some tolerance
        
        if scanner_ok and tampered_ok:
            print("✅ Test PASSED - Application is working correctly!")
            return True
        else:
            print("❌ Test FAILED - Application needs fixes!")
            return False

def main():
    """Main test function"""
    tester = CombinedAppTester()
    
    # Run comprehensive test
    success = tester.run_comprehensive_test(
        scanner_images=100,  # Test 100 scanner images
        tampered_images=100  # Test 100 tampered images
    )
    
    if success:
        print("\n🚀 Application is ready for deployment!")
        sys.exit(0)
    else:
        print("\n🔧 Application needs debugging!")
        sys.exit(1)

if __name__ == "__main__":
    main()
