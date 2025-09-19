#!/usr/bin/env python3
"""
Quick test of the Streamlit app models
"""

import os
import sys
import random
import numpy as np
import cv2
import tifffile

# Add current directory to path
sys.path.append('.')
from scanner_identification_fixed import FixedScannerIdentifier
from streamlit_separate_buttons import BalancedTamperedDetector

def quick_test():
    print("🧪 Quick Test of Streamlit App Models")
    print("=" * 50)
    
    # Load models
    print("🔄 Loading Models...")
    scanner_id = FixedScannerIdentifier()
    scanner_loaded = scanner_id.load_model()
    
    tampered_detector = BalancedTamperedDetector()
    tampered_loaded = tampered_detector.load_model()
    
    if not scanner_loaded:
        print("❌ Scanner model failed to load")
        return
    if not tampered_loaded:
        print("❌ Tampered model failed to load")
        return
    
    print("✅ Both models loaded successfully!")
    
    # Test scanner identification
    print("\n🖨️ Testing Scanner ID (5 images)...")
    scanner_files = [
        "The SUPATLANTIQUE dataset/Wikipedia/HP/300/s11_23.tif",
        "The SUPATLANTIQUE dataset/Wikipedia/Canon120-2/300/s2_106.tif", 
        "The SUPATLANTIQUE dataset/Wikipedia/EpsonV550/300/s10_9.tif",
        "The SUPATLANTIQUE dataset/Wikipedia/Canon220/300/s3_53.tif",
        "The SUPATLANTIQUE dataset/Wikipedia/EpsonV39-1/300/s9_52.tif"
    ]
    
    scanner_correct = 0
    for i, file_path in enumerate(scanner_files):
        if os.path.exists(file_path):
            try:
                image = tifffile.imread(file_path)
                result, conf, status, top3 = scanner_id.predict_scanner_with_top3(image)
                expected = ["HP", "Canon120-2", "EpsonV550", "Canon220", "EpsonV39-1"][i]
                is_correct = result == expected
                if is_correct:
                    scanner_correct += 1
                print(f"  {os.path.basename(file_path)}: {result} (conf: {conf:.3f}) {'✅' if is_correct else '❌'}")
            except Exception as e:
                print(f"  {os.path.basename(file_path)}: Error - {e}")
    
    scanner_accuracy = scanner_correct / len(scanner_files)
    print(f"Scanner ID Accuracy: {scanner_correct}/{len(scanner_files)} ({scanner_accuracy:.1%})")
    
    # Test tampered detection
    print("\n🔒 Testing Tampered Detection (10 images)...")
    
    # Get some original images
    original_files = []
    original_dir = "The SUPATLANTIQUE dataset/Tampered images/Original"
    if os.path.exists(original_dir):
        files = [f for f in os.listdir(original_dir) if f.lower().endswith('.tif')][:5]
        original_files = [(os.path.join(original_dir, f), 0) for f in files]
    
    # Get some tampered images
    tampered_files = []
    tampered_dirs = [
        "The SUPATLANTIQUE dataset/Tampered images/Tampered/Copy-move",
        "The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching", 
        "The SUPATLANTIQUE dataset/Tampered images/Tampered/Splicing"
    ]
    
    for tampered_dir in tampered_dirs:
        if os.path.exists(tampered_dir):
            files = [f for f in os.listdir(tampered_dir) if f.lower().endswith('.tif')][:2]
            tampered_files.extend([(os.path.join(tampered_dir, f), 1) for f in files])
    
    test_files = original_files + tampered_files[:5]  # 5 original + 5 tampered
    
    tampered_correct = 0
    for file_path, true_label in test_files:
        try:
            image = tifffile.imread(file_path)
            result, confidence, status = tampered_detector.predict_tampered(image)
            pred_label = 1 if result == "Tampered" else 0
            is_correct = pred_label == true_label
            if is_correct:
                tampered_correct += 1
            print(f"  {os.path.basename(file_path)}: {result} (conf: {confidence:.3f}) {'✅' if is_correct else '❌'}")
        except Exception as e:
            print(f"  {os.path.basename(file_path)}: Error - {e}")
    
    tampered_accuracy = tampered_correct / len(test_files) if test_files else 0
    print(f"Tampered Detection Accuracy: {tampered_correct}/{len(test_files)} ({tampered_accuracy:.1%})")
    
    # Final results
    print(f"\n🎯 Final Results:")
    print(f"  Scanner ID: {scanner_accuracy:.1%} {'✅' if scanner_accuracy >= 0.8 else '❌'}")
    print(f"  Tampered Detection: {tampered_accuracy:.1%} {'✅' if tampered_accuracy >= 0.8 else '❌'}")
    
    return scanner_accuracy >= 0.8 and tampered_accuracy >= 0.8

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)
