#!/usr/bin/env python3
"""
Final Validation Test for AI TraceFinder Combined Application
Tests both objectives with balanced sampling from all scanners and tampered images
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
import random
warnings.filterwarnings('ignore')

# Import our modules
from combined_feature_extractors import CombinedFeatureExtractor

class FinalValidator:
    """Final validation for both objectives"""
    
    def __init__(self):
        self.feature_extractor = CombinedFeatureExtractor()
        self.results = []
        
        # Dataset paths
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        self.official_path = os.path.join(self.dataset_path, "Official")
        self.wikipedia_path = os.path.join(self.dataset_path, "Wikipedia")
        self.tampered_path = os.path.join(self.dataset_path, "Tampered images")
    
    def get_all_scanner_images(self, num_per_scanner=10):
        """Get balanced sample of images from all scanners"""
        all_scanners = []
        
        # Collect unique scanner names
        for source_path in [self.flatfield_path, self.official_path, self.wikipedia_path]:
            if os.path.exists(source_path):
                for scanner_dir in os.listdir(source_path):
                    if scanner_dir not in all_scanners:
                        all_scanners.append(scanner_dir)
        
        print(f"Found {len(all_scanners)} scanner types: {all_scanners}")
        
        scanner_images = []
        
        for scanner_dir in all_scanners:
            scanner_images_found = []
            
            # Collect from all sources
            for source_path in [self.flatfield_path, self.official_path, self.wikipedia_path]:
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
                random.shuffle(scanner_images_found)
                selected = scanner_images_found[:num_per_scanner]
                scanner_images.extend(selected)
                print(f"Selected {len(selected)} images from {scanner_dir}")
        
        return scanner_images
    
    def get_tampered_images(self, num_images=50):
        """Get tampered and original images"""
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
                            tampered_images.append((img_path, 1, tamper_type))
        
        # Original images
        original_dir = os.path.join(self.tampered_path, "Original")
        if os.path.exists(original_dir):
            for img_file in os.listdir(original_dir):
                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                    img_path = os.path.join(original_dir, img_file)
                    original_images.append((img_path, 0, "none"))
        
        # Balance the dataset
        min_samples = min(len(tampered_images), len(original_images), num_images // 2)
        random.shuffle(tampered_images)
        random.shuffle(original_images)
        
        all_images = tampered_images[:min_samples] + original_images[:min_samples]
        random.shuffle(all_images)
        
        print(f"Found {len(tampered_images)} tampered and {len(original_images)} original images")
        print(f"Using {len(all_images)} images for testing")
        
        return all_images
    
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
                image = image[:, :, :3]
            
            return image
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def test_scanner_identification(self, num_per_scanner=10):
        """Test scanner identification with balanced sampling"""
        print(f"\n🔍 Testing Scanner Identification")
        print("=" * 50)
        
        # Get balanced sample of images
        scanner_images = self.get_all_scanner_images(num_per_scanner)
        
        if not scanner_images:
            print("❌ No scanner images found")
            return 0.0
        
        print(f"Testing with {len(scanner_images)} images")
        
        # Import the working scanner identifier
        sys.path.append('.')
        from streamlit_combined_app_working import ScannerIdentifier
        
        scanner_identifier = ScannerIdentifier()
        if not scanner_identifier.load_model():
            print("❌ Failed to load scanner model")
            return 0.0
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (img_path, expected_scanner, dpi) in enumerate(scanner_images):
            try:
                print(f"Testing {i+1}/{len(scanner_images)}: {os.path.basename(img_path)}")
                
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Convert to grayscale and normalize
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                gray = gray.astype(np.float64) / 255.0
                
                # Predict
                predicted_scanner, confidence = scanner_identifier.predict_scanner(gray)
                
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
    
    def test_tampered_detection(self, num_images=50):
        """Test tampered detection"""
        print(f"\n🛡️ Testing Tampered Detection")
        print("=" * 50)
        
        # Get tampered images
        tampered_images = self.get_tampered_images(num_images)
        
        if not tampered_images:
            print("❌ No tampered images found")
            return 0.0
        
        print(f"Testing with {len(tampered_images)} images")
        
        # For now, simulate tampered detection (we'll implement the actual model later)
        correct_predictions = 0
        total_predictions = 0
        
        for i, (img_path, expected_label, tamper_type) in enumerate(tampered_images):
            try:
                print(f"Testing {i+1}/{len(tampered_images)}: {os.path.basename(img_path)}")
                
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # For now, use a simple heuristic based on image characteristics
                # This is a placeholder - we'll implement the actual model
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                
                # Simple heuristic: if image has high variance, it might be tampered
                variance = np.var(gray)
                predicted_label = 1 if variance > 5000 else 0
                predicted_result = "Tampered" if predicted_label == 1 else "Original"
                confidence = 0.7  # Placeholder
                
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
    
    def save_results(self, filename="final_validation_results.csv"):
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
    
    def run_validation(self):
        """Run complete validation"""
        print("🧪 AI TraceFinder - Final Validation")
        print("=" * 60)
        
        # Test scanner identification
        scanner_accuracy = self.test_scanner_identification(num_per_scanner=5)  # 5 per scanner for quick test
        
        # Test tampered detection
        tampered_accuracy = self.test_tampered_detection(num_images=20)  # 20 images for quick test
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎯 Final Validation Results:")
        print(f"   Scanner ID: {scanner_accuracy:.1%} (Expected: 93.75%)")
        print(f"   Tampered Detection: {tampered_accuracy:.1%} (Expected: 89.86%)")
        
        # Check if results are acceptable
        scanner_ok = scanner_accuracy >= 0.80  # Allow some tolerance
        tampered_ok = tampered_accuracy >= 0.70  # Allow some tolerance for placeholder
        
        if scanner_ok:
            print("✅ Scanner identification is working well!")
        else:
            print("❌ Scanner identification needs improvement!")
        
        if tampered_ok:
            print("✅ Tampered detection placeholder is working!")
        else:
            print("❌ Tampered detection needs implementation!")
        
        return scanner_ok, tampered_ok

def main():
    """Main validation function"""
    validator = FinalValidator()
    scanner_ok, tampered_ok = validator.run_validation()
    
    if scanner_ok:
        print("\n🚀 Scanner identification is ready for deployment!")
        print("📱 Streamlit app is running at: http://localhost:8502")
    else:
        print("\n🔧 Scanner identification needs debugging!")
    
    if tampered_ok:
        print("🛡️ Tampered detection framework is ready!")
    else:
        print("🔧 Tampered detection needs implementation!")

if __name__ == "__main__":
    main()
