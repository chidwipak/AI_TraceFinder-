#!/usr/bin/env python3
"""
Final Combined Test for AI TraceFinder
Tests both Scanner ID and Tampered Detection with balanced sampling
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from scanner_identification_fixed import FixedScannerIdentifier
from streamlit_final_combined_app import TamperedDetector

class FinalCombinedTester:
    """Final test for both objectives"""
    
    def __init__(self):
        self.scanner_identifier = FixedScannerIdentifier()
        self.tampered_detector = TamperedDetector()
        self.results = []
        
        # Dataset paths
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        self.official_path = os.path.join(self.dataset_path, "Official")
        self.wikipedia_path = os.path.join(self.dataset_path, "Wikipedia")
        self.tampered_path = os.path.join(self.dataset_path, "Tampered images")
    
    def load_models(self):
        """Load both models"""
        print("🔄 Loading models...")
        
        # Load scanner identification model
        if not self.scanner_identifier.load_model():
            print("❌ Failed to load scanner identification model")
            return False
        
        # Load tampered detection model
        if not self.tampered_detector.load_model():
            print("❌ Failed to load tampered detection model")
            return False
        
        print("✅ Both models loaded successfully!")
        return True
    
    def get_scanner_test_images(self, num_per_scanner=5):
        """Get test images for scanner identification"""
        all_scanners = []
        
        # Collect unique scanner names
        for source_path in [self.flatfield_path, self.official_path, self.wikipedia_path]:
            if os.path.exists(source_path):
                for scanner_dir in os.listdir(source_path):
                    if scanner_dir not in all_scanners and scanner_dir != '.DS_Store':
                        all_scanners.append(scanner_dir)
        
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
        
        return scanner_images
    
    def get_tampered_test_images(self, num_images=30):
        """Get test images for tampered detection"""
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
        
        return all_images
    
    def load_image(self, image_path):
        """Load image from file"""
        try:
            import tifffile
            
            if image_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(image_path)
            else:
                import cv2
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Ensure image is RGB
            if len(image.shape) == 2:
                import cv2
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            return image
            
        except Exception as e:
            print(f"❌ Error loading image {image_path}: {e}")
            return None
    
    def test_scanner_identification(self, num_per_scanner=5):
        """Test scanner identification"""
        print(f"\n🔍 Testing Scanner Identification")
        print("=" * 50)
        
        # Get test images
        scanner_images = self.get_scanner_test_images(num_per_scanner)
        
        if not scanner_images:
            print("❌ No scanner images found")
            return 0.0
        
        print(f"Testing with {len(scanner_images)} images")
        
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
                import cv2
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                gray = gray.astype(np.float64) / 255.0
                
                # Preprocess for scanner identification
                processed_image = self.scanner_identifier.preprocess_image(gray)
                if processed_image is None:
                    continue
                
                # Predict
                predicted_scanner, confidence, status = self.scanner_identifier.predict_scanner(processed_image)
                
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
    
    def test_tampered_detection(self, num_images=30):
        """Test tampered detection"""
        print(f"\n🛡️ Testing Tampered Detection")
        print("=" * 50)
        
        # Get test images
        tampered_images = self.get_tampered_test_images(num_images)
        
        if not tampered_images:
            print("❌ No tampered images found")
            return 0.0
        
        print(f"Testing with {len(tampered_images)} images")
        
        correct_predictions = 0
        total_predictions = 0
        
        for i, (img_path, expected_label, tamper_type) in enumerate(tampered_images):
            try:
                print(f"Testing {i+1}/{len(tampered_images)}: {os.path.basename(img_path)}")
                
                # Load image
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Predict
                predicted_result, confidence = self.tampered_detector.predict_tampered(image)
                
                # Convert result to label
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
                
                expected_text = "Tampered" if expected_label == 1 else "Original"
                print(f"  Expected: {expected_text}, Predicted: {predicted_result}, Confidence: {confidence:.3f}, Correct: {is_correct}")
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        print(f"\n📊 Tampered Detection Results:")
        print(f"   Correct: {correct_predictions}/{total_predictions}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Expected: 89.86%")
        
        return accuracy
    
    def save_results(self, filename="final_combined_test_results.csv"):
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
    
    def run_final_test(self):
        """Run final combined test"""
        print("🧪 AI TraceFinder - Final Combined Test")
        print("=" * 60)
        
        # Load models
        if not self.load_models():
            return False
        
        # Test scanner identification
        scanner_accuracy = self.test_scanner_identification(num_per_scanner=3)  # 3 per scanner for quick test
        
        # Test tampered detection
        tampered_accuracy = self.test_tampered_detection(num_images=20)  # 20 images for quick test
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎯 Final Combined Test Results:")
        print(f"   Scanner ID: {scanner_accuracy:.1%} (Expected: 93.75%)")
        print(f"   Tampered Detection: {tampered_accuracy:.1%} (Expected: 89.86%)")
        
        # Check if results are acceptable
        scanner_ok = scanner_accuracy >= 0.70  # Allow some tolerance
        tampered_ok = tampered_accuracy >= 0.70  # Allow some tolerance
        
        if scanner_ok:
            print("✅ Scanner identification is working well!")
        else:
            print("❌ Scanner identification needs improvement!")
        
        if tampered_ok:
            print("✅ Tampered detection is working well!")
        else:
            print("❌ Tampered detection needs improvement!")
        
        return scanner_ok and tampered_ok

def main():
    """Main test function"""
    tester = FinalCombinedTester()
    success = tester.run_final_test()
    
    if success:
        print("\n🚀 Final combined testing PASSED!")
        print("📱 Streamlit app is ready for deployment!")
        print("🌐 Run: streamlit run streamlit_final_combined_app.py")
    else:
        print("\n🔧 Final combined testing needs improvements!")

if __name__ == "__main__":
    main()
