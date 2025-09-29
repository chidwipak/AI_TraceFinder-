#!/usr/bin/env python3
"""
Comprehensive Testing Script for AI TraceFinder
Tests all edge cases and accuracy across all dataset folders
"""

import os
import random
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import time
import warnings
warnings.filterwarnings('ignore')

from scanner_identification_fixed import FixedScannerIdentifier
from combined_feature_extractors import CombinedFeatureExtractor
import joblib

class ComprehensiveTester:
    """Comprehensive testing for all scenarios"""
    
    def __init__(self):
        self.scanner_identifier = FixedScannerIdentifier()
        self.feature_extractor = CombinedFeatureExtractor()
        self.tampered_model = None
        self.tampered_scaler = None
        self.results = []
        
        # Load models
        self.load_models()
        
        # Dataset paths
        self.dataset_root = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_root, "Flatfield")
        self.official_path = os.path.join(self.dataset_root, "Official")
        self.wikipedia_path = os.path.join(self.dataset_root, "Wikipedia")
        self.originals_path = os.path.join(self.dataset_root, "Originals")
        self.tampered_path = os.path.join(self.dataset_root, "Tampered images")
    
    def load_models(self):
        """Load all required models"""
        print("🔄 Loading models...")
        
        # Load scanner model
        if not self.scanner_identifier.load_model():
            print("❌ Failed to load scanner model")
            return False
        
        # Load tampered model (baseline)
        try:
            self.tampered_model = joblib.load("new_objective2_baseline_ml_results/models/best_model.pkl")
            self.tampered_scaler = joblib.load("new_objective2_baseline_ml_results/models/scaler.pkl")
            print("✅ All models loaded successfully")
            return True
        except Exception as e:
            print(f"❌ Failed to load tampered model: {e}")
            return False
    
    def load_image(self, image_path):
        """Load image with proper format handling"""
        try:
            if image_path.lower().endswith(('.tif', '.tiff')):
                import tifffile
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                return None
            
            # Ensure RGB format
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = image[:, :, :3]
            
            return image
        except Exception as e:
            print(f"❌ Error loading {image_path}: {e}")
            return None
    
    def test_scanner_identification(self, num_samples=50):
        """Test scanner identification across all folders"""
        print(f"\n🔍 Testing Scanner Identification ({num_samples} samples)")
        print("=" * 60)
        
        # Collect samples from all scanner folders
        scanner_samples = []
        
        for source_name, source_path in [
            ("Flatfield", self.flatfield_path),
            ("Official", self.official_path),
            ("Wikipedia", self.wikipedia_path)
        ]:
            if os.path.exists(source_path):
                for scanner_dir in os.listdir(source_path):
                    if scanner_dir == '.DS_Store':
                        continue
                    sp = os.path.join(source_path, scanner_dir)
                    if not os.path.isdir(sp):
                        continue
                    
                    # Collect images from both DPI folders
                    for dpi_dir in ['150', '300']:
                        dp = os.path.join(sp, dpi_dir)
                        if os.path.isdir(dp):
                            imgs = [os.path.join(dp, f) for f in os.listdir(dp) 
                                   if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))]
                            for img_path in imgs:
                                scanner_samples.append((img_path, scanner_dir, source_name, dpi_dir))
        
        # Sample randomly
        random.shuffle(scanner_samples)
        scanner_samples = scanner_samples[:num_samples]
        
        print(f"Testing with {len(scanner_samples)} scanner images")
        
        correct = 0
        total = 0
        confidence_scores = []
        
        for i, (img_path, expected_scanner, source, dpi) in enumerate(scanner_samples, 1):
            try:
                print(f"[{i:02d}/{len(scanner_samples)}] {os.path.basename(img_path)} ({expected_scanner})")
                
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Convert to grayscale for scanner ID
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                gray = gray.astype(np.float64) / 255.0
                
                # Predict
                pred_scanner, conf, status = self.scanner_identifier.predict_scanner(gray)
                
                total += 1
                is_correct = pred_scanner == expected_scanner
                if is_correct:
                    correct += 1
                
                confidence_scores.append(conf)
                
                print(f"  Expected: {expected_scanner}, Predicted: {pred_scanner}, Conf: {conf:.3f}, Correct: {is_correct}")
                
                # Store result
                self.results.append({
                    'type': 'scanner',
                    'image_path': img_path,
                    'expected': expected_scanner,
                    'predicted': pred_scanner,
                    'confidence': conf,
                    'correct': is_correct,
                    'source': source,
                    'dpi': dpi
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        print(f"\n📊 Scanner ID Results:")
        print(f"   Correct: {correct}/{total}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Target: >80%")
        
        return accuracy >= 0.8
    
    def test_tampered_detection(self, num_samples=50):
        """Test tampered detection across all folders"""
        print(f"\n🔒 Testing Tampered Detection ({num_samples} samples)")
        print("=" * 60)
        
        # Collect tampered and original samples
        tampered_samples = []
        original_samples = []
        
        # Tampered images
        tampered_dir = os.path.join(self.tampered_path, "Tampered")
        if os.path.exists(tampered_dir):
            for tamper_type in os.listdir(tampered_dir):
                tp = os.path.join(tampered_dir, tamper_type)
                if os.path.isdir(tp):
                    for img_file in os.listdir(tp):
                        if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(tp, img_file)
                            tampered_samples.append((img_path, 1, tamper_type))
        
        # Original images
        original_dir = os.path.join(self.tampered_path, "Original")
        if os.path.exists(original_dir):
            for img_file in os.listdir(original_dir):
                if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(original_dir, img_file)
                    original_samples.append((img_path, 0, "original"))
        
        # Balance samples
        min_samples = min(len(tampered_samples), len(original_samples), num_samples // 2)
        random.shuffle(tampered_samples)
        random.shuffle(original_samples)
        
        all_samples = tampered_samples[:min_samples] + original_samples[:min_samples]
        random.shuffle(all_samples)
        
        print(f"Testing with {len(all_samples)} images ({min_samples} tampered, {min_samples} original)")
        
        correct = 0
        total = 0
        confidence_scores = []
        
        for i, (img_path, expected_label, tamper_type) in enumerate(all_samples, 1):
            try:
                print(f"[{i:02d}/{len(all_samples)}] {os.path.basename(img_path)} ({'Tampered' if expected_label == 1 else 'Original'})")
                
                image = self.load_image(img_path)
                if image is None:
                    continue
                
                # Extract features and predict (baseline model doesn't use imputer)
                features = self.feature_extractor.extract_tampered_features(image)
                features = features.reshape(1, -1)
                features_scaled = self.tampered_scaler.transform(features)
                
                if hasattr(self.tampered_model, 'predict_proba'):
                    proba = self.tampered_model.predict_proba(features_scaled)[0]
                    pred_label = int(np.argmax(proba))
                    conf = float(np.max(proba))
                else:
                    pred_label = int(self.tampered_model.predict(features_scaled)[0])
                    conf = 0.5
                
                total += 1
                is_correct = pred_label == expected_label
                if is_correct:
                    correct += 1
                
                confidence_scores.append(conf)
                
                print(f"  Expected: {'Tampered' if expected_label == 1 else 'Original'}, Predicted: {'Tampered' if pred_label == 1 else 'Original'}, Conf: {conf:.3f}, Correct: {is_correct}")
                
                # Store result
                self.results.append({
                    'type': 'tampered',
                    'image_path': img_path,
                    'expected': expected_label,
                    'predicted': pred_label,
                    'confidence': conf,
                    'correct': is_correct,
                    'tamper_type': tamper_type
                })
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
        
        print(f"\n📊 Tampered Detection Results:")
        print(f"   Correct: {correct}/{total}")
        print(f"   Accuracy: {accuracy:.1%}")
        print(f"   Average Confidence: {avg_confidence:.3f}")
        print(f"   Target: >80%")
        
        return accuracy >= 0.8
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        print(f"\n⚠️ Testing Edge Cases")
        print("=" * 60)
        
        edge_cases = []
        
        # Test with very small images
        small_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        edge_cases.append(("Very small image", small_img, "scanner"))
        edge_cases.append(("Very small image", small_img, "tampered"))
        
        # Test with very large images
        large_img = np.random.randint(0, 255, (3000, 3000, 3), dtype=np.uint8)
        edge_cases.append(("Very large image", large_img, "scanner"))
        edge_cases.append(("Very large image", large_img, "tampered"))
        
        # Test with grayscale images
        gray_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        edge_cases.append(("Grayscale image", gray_img, "scanner"))
        edge_cases.append(("Grayscale image", gray_img, "tampered"))
        
        # Test with RGBA images
        rgba_img = np.random.randint(0, 255, (256, 256, 4), dtype=np.uint8)
        edge_cases.append(("RGBA image", rgba_img, "scanner"))
        edge_cases.append(("RGBA image", rgba_img, "tampered"))
        
        # Test with corrupted data
        corrupted_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        corrupted_img[100:150, 100:150] = 0  # Black square
        edge_cases.append(("Corrupted image", corrupted_img, "scanner"))
        edge_cases.append(("Corrupted image", corrupted_img, "tampered"))
        
        for case_name, img, test_type in edge_cases:
            try:
                print(f"Testing {case_name} for {test_type}...")
                
                if test_type == "scanner":
                    # Convert to grayscale
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    else:
                        gray = img
                    gray = gray.astype(np.float64) / 255.0
                    
                    pred, conf, status = self.scanner_identifier.predict_scanner(gray)
                    print(f"  Result: {pred}, Confidence: {conf:.3f}, Status: {status}")
                
                elif test_type == "tampered":
                    # Ensure RGB
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    elif img.shape[2] == 4:
                        img = img[:, :, :3]
                    
                    features = self.feature_extractor.extract_tampered_features(img)
                    features = features.reshape(1, -1)
                    features_scaled = self.tampered_scaler.transform(features)
                    
                    if hasattr(self.tampered_model, 'predict_proba'):
                        proba = self.tampered_model.predict_proba(features_scaled)[0]
                        pred = int(np.argmax(proba))
                        conf = float(np.max(proba))
                    else:
                        pred = int(self.tampered_model.predict(features_scaled)[0])
                        conf = 0.5
                    
                    print(f"  Result: {'Tampered' if pred == 1 else 'Original'}, Confidence: {conf:.3f}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        print("✅ Edge case testing completed")
    
    def save_results(self, filename="comprehensive_test_results.csv"):
        """Save test results to CSV"""
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(filename, index=False)
            print(f"\n💾 Results saved to {filename}")
            
            # Summary statistics
            scanner_results = df[df['type'] == 'scanner']
            tampered_results = df[df['type'] == 'tampered']
            
            print(f"\n📊 Summary Statistics:")
            if len(scanner_results) > 0:
                scanner_accuracy = scanner_results['correct'].mean()
                print(f"   Scanner ID Accuracy: {scanner_accuracy:.1%}")
            
            if len(tampered_results) > 0:
                tampered_accuracy = tampered_results['correct'].mean()
                print(f"   Tampered Detection Accuracy: {tampered_accuracy:.1%}")
    
    def run_comprehensive_test(self):
        """Run complete comprehensive test"""
        print("🧪 AI TraceFinder - Comprehensive Testing")
        print("=" * 80)
        
        # Test scanner identification
        scanner_success = self.test_scanner_identification(num_samples=30)
        
        # Test tampered detection
        tampered_success = self.test_tampered_detection(num_samples=30)
        
        # Test edge cases
        self.test_edge_cases()
        
        # Save results
        self.save_results()
        
        # Final summary
        print(f"\n🎯 Final Test Results:")
        print(f"   Scanner ID: {'✅ PASS' if scanner_success else '❌ FAIL'} (Target: >80%)")
        print(f"   Tampered Detection: {'✅ PASS' if tampered_success else '❌ FAIL'} (Target: >80%)")
        
        if scanner_success and tampered_success:
            print("🎉 All tests passed! The application is ready for deployment.")
        else:
            print("⚠️ Some tests failed. Review results and iterate improvements.")
        
        return scanner_success and tampered_success

def main():
    """Main testing function"""
    tester = ComprehensiveTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🚀 Perfect Streamlit app is ready!")
        print("📱 Run: streamlit run streamlit_perfect_app.py --server.port 8506")
    else:
        print("\n🔧 Further improvements needed. Check results and iterate.")

if __name__ == "__main__":
    main()
