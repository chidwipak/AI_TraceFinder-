#!/usr/bin/env python3
"""
Proper test using actual Streamlit app models
"""

import os
import sys
import random
import numpy as np
import cv2
import tifffile
from sklearn.metrics import accuracy_score, classification_report

# Add current directory to path
sys.path.append('.')
from scanner_identification_fixed import FixedScannerIdentifier
from streamlit_separate_buttons import BalancedTamperedDetector

class ProperTester:
    def __init__(self):
        self.scanner_identifier = None
        self.tampered_detector = None
        
    def load_models(self):
        """Load both models"""
        print("🔄 Loading Models")
        print("=" * 25)
        
        # Load scanner identifier
        try:
            self.scanner_identifier = FixedScannerIdentifier()
            if self.scanner_identifier.load_model():
                print("✅ Scanner identification model loaded")
            else:
                print("❌ Failed to load scanner model")
                return False
        except Exception as e:
            print(f"❌ Scanner model error: {e}")
            return False
        
        # Load tampered detector
        try:
            self.tampered_detector = BalancedTamperedDetector()
            if self.tampered_detector.load_model():
                print("✅ Tampered detection model loaded")
            else:
                print("❌ Failed to load tampered model")
                return False
        except Exception as e:
            print(f"❌ Tampered model error: {e}")
            return False
        
        return True
    
    def test_scanner_identification(self, test_images=20):
        """Test scanner identification with correct paths"""
        print(f"\n🖨️ Testing Scanner Identification ({test_images} images)")
        print("=" * 55)
        
        # Test with actual scanner directories (they have subdirectories)
        test_cases = [
            ("The SUPATLANTIQUE dataset/Wikipedia/HP", "HP"),
            ("The SUPATLANTIQUE dataset/Wikipedia/Canon120-2", "Canon120-2"),
            ("The SUPATLANTIQUE dataset/Wikipedia/EpsonV550", "EpsonV550"),
            ("The SUPATLANTIQUE dataset/Wikipedia/Canon220", "Canon220"),
            ("The SUPATLANTIQUE dataset/Wikipedia/EpsonV39-1", "EpsonV39-1")
        ]
        
        correct = 0
        total = 0
        results = []
        
        for test_dir, expected_scanner in test_cases:
            if os.path.exists(test_dir):
                # Find TIF files in subdirectories
                tif_files = []
                for root, dirs, files in os.walk(test_dir):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff')):
                            tif_files.append(os.path.join(root, file))
                
                files = tif_files[:4]  # Test 4 per scanner
                
                for file_path in files:
                    try:
                        image = tifffile.imread(file_path)
                        
                        result, conf, status, top3 = self.scanner_identifier.predict_scanner_with_top3(image)
                        
                        is_correct = result == expected_scanner
                        if is_correct:
                            correct += 1
                        
                        total += 1
                        results.append({
                            'file': file,
                            'expected': expected_scanner,
                            'predicted': result,
                            'confidence': conf,
                            'correct': is_correct
                        })
                        
                        print(f"📷 {os.path.basename(file_path)}: {result} (conf: {conf:.3f}) {'✅' if is_correct else '❌'}")
                        
                    except Exception as e:
                        print(f"   ❌ Test failed: {e}")
        
        if total > 0:
            accuracy = correct / total
            print(f"\n📊 Scanner ID Results:")
            print(f"   Accuracy: {correct}/{total} ({accuracy:.1%})")
            return accuracy >= 0.8
        else:
            print("❌ No scanner tests completed")
            return False
    
    def test_tampered_detection(self, test_images=20):
        """Test tampered detection with actual model"""
        print(f"\n🔒 Testing Tampered Detection ({test_images} images)")
        print("=" * 50)
        
        # Test with balanced dataset
        original_files = []
        original_dir = "The SUPATLANTIQUE dataset/Tampered images/Original"
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    original_files.append((os.path.join(original_dir, file), 0))  # 0 for original
        
        tampered_files = []
        tampered_dirs = [
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Copy-move",
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching",
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Splicing"
        ]
        
        for tampered_dir in tampered_dirs:
            if os.path.exists(tampered_dir):
                for file in os.listdir(tampered_dir):
                    if file.lower().endswith(('.tif', '.tiff')):
                        tampered_files.append((os.path.join(tampered_dir, file), 1))  # 1 for tampered
        
        # Sample test images
        test_originals = random.sample(original_files, min(10, len(original_files)))
        test_tampered = random.sample(tampered_files, min(10, len(tampered_files)))
        test_images = test_originals + test_tampered
        
        correct = 0
        total = 0
        results = []
        
        for img_path, true_label in test_images:
            try:
                image = tifffile.imread(img_path)
                
                # Use actual tampered detection model
                result, confidence, status = self.tampered_detector.predict_tampered(image)
                pred_label = 1 if result == "Tampered" else 0
                
                is_correct = pred_label == true_label
                if is_correct:
                    correct += 1
                
                total += 1
                results.append({
                    'file': os.path.basename(img_path),
                    'true_label': 'Original' if true_label == 0 else 'Tampered',
                    'predicted': 'Original' if pred_label == 0 else 'Tampered',
                    'confidence': confidence,
                    'correct': is_correct
                })
                
                print(f"🔍 {os.path.basename(img_path)}: {'Original' if pred_label == 0 else 'Tampered'} (conf: {confidence:.3f}) {'✅' if is_correct else '❌'}")
                
            except Exception as e:
                print(f"   ❌ Test failed: {e}")
        
        if total > 0:
            accuracy = correct / total
            print(f"\n📊 Tampered Detection Results:")
            print(f"   Accuracy: {correct}/{total} ({accuracy:.1%})")
            return accuracy >= 0.8
        else:
            print("❌ No tampered detection tests completed")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test"""
        print("🧪 Proper Comprehensive Test")
        print("=" * 50)
        
        # Load models
        if not self.load_models():
            print("❌ Failed to load models")
            return False
        
        # Test scanner identification
        scanner_success = self.test_scanner_identification()
        
        # Test tampered detection
        tampered_success = self.test_tampered_detection()
        
        # Final results
        print("\n🎯 Final Test Results")
        print("=" * 25)
        print(f"Scanner ID: {'✅' if scanner_success else '❌'}")
        print(f"Tampered Detection: {'✅' if tampered_success else '❌'}")
        
        if scanner_success and tampered_success:
            print("\n🎉 All tests passed!")
            return True
        else:
            print("\n⚠️ Some tests failed - needs attention")
            return False

if __name__ == "__main__":
    tester = ProperTester()
    success = tester.run_comprehensive_test()
    sys.exit(0 if success else 1)
