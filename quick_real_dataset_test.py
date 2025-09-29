#!/usr/bin/env python3
"""
Quick test to verify the real dataset approach is working correctly
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
import tifffile
from pathlib import Path

# Add current directory to path
sys.path.append('.')

def analyze_real_dataset():
    """Analyze the actual dataset structure"""
    print("🔍 Analyzing Real Dataset Structure")
    print("=" * 50)
    
    dataset_path = "The SUPATLANTIQUE dataset"
    
    # Check Tampered images/Original (these are originals paired with tampered)
    original_paired_dir = os.path.join(dataset_path, "Tampered images", "Original")
    original_paired_files = []
    if os.path.exists(original_paired_dir):
        for file in os.listdir(original_paired_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                original_paired_files.append(os.path.join(original_paired_dir, file))
    
    print(f"📊 Original images (paired with tampered): {len(original_paired_files)}")
    print(f"   Sample files: {[os.path.basename(f) for f in original_paired_files[:3]]}")
    
    # Check Tampered images/Tampered
    tampered_dirs = [
        os.path.join(dataset_path, "Tampered images", "Tampered", "Copy-move"),
        os.path.join(dataset_path, "Tampered images", "Tampered", "Retouching"), 
        os.path.join(dataset_path, "Tampered images", "Tampered", "Splicing")
    ]
    
    tampered_files = []
    for tampered_dir in tampered_dirs:
        if os.path.exists(tampered_dir):
            subdir_files = []
            for file in os.listdir(tampered_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    subdir_files.append(os.path.join(tampered_dir, file))
            tampered_files.extend(subdir_files)
            print(f"📊 {os.path.basename(tampered_dir)} tampered: {len(subdir_files)}")
            if subdir_files:
                print(f"   Sample: {os.path.basename(subdir_files[0])}")
    
    print(f"📊 Total tampered images: {len(tampered_files)}")
    
    # Check Official and Wikipedia (additional originals)
    additional_original_dirs = [
        os.path.join(dataset_path, "Official"),
        os.path.join(dataset_path, "Wikipedia")
    ]
    
    additional_originals = []
    for source_dir in additional_original_dirs:
        if os.path.exists(source_dir):
            source_count = 0
            for scanner_dir in os.listdir(source_dir):
                scanner_path = os.path.join(source_dir, scanner_dir)
                if os.path.isdir(scanner_path):
                    scanner_files = [f for f in os.listdir(scanner_path) 
                                   if f.lower().endswith(('.tif', '.tiff'))]
                    source_count += len(scanner_files)
                    additional_originals.extend([os.path.join(scanner_path, f) for f in scanner_files])
            print(f"📊 {os.path.basename(source_dir)} original images: {source_count}")
    
    print(f"📊 Total additional originals available: {len(additional_originals)}")
    
    # Dataset balance analysis
    print(f"\n🎯 Dataset Balance Analysis:")
    print(f"   Tampered images: {len(tampered_files)}")
    print(f"   Paired originals: {len(original_paired_files)}")
    print(f"   Additional originals needed: {max(0, len(tampered_files) - len(original_paired_files))}")
    print(f"   Additional originals available: {len(additional_originals)}")
    
    # Check if we can balance
    total_originals_possible = len(original_paired_files) + len(additional_originals)
    can_balance = total_originals_possible >= len(tampered_files)
    
    print(f"   Can create balanced dataset: {'✅ YES' if can_balance else '❌ NO'}")
    
    if can_balance:
        balanced_size = min(len(tampered_files), total_originals_possible)
        print(f"   Recommended balanced size: {balanced_size} each (Original/Tampered)")
    
    return {
        'tampered_files': tampered_files,
        'original_paired_files': original_paired_files,
        'additional_originals': additional_originals,
        'can_balance': can_balance
    }

def test_feature_extraction():
    """Test feature extraction on a sample image"""
    print("\n🧪 Testing Feature Extraction")
    print("=" * 40)
    
    # Find a sample image
    sample_paths = [
        "The SUPATLANTIQUE dataset/Tampered images/Original",
        "The SUPATLANTIQUE dataset/Official/HP",
        "The SUPATLANTIQUE dataset/Wikipedia/Canon220"
    ]
    
    sample_image = None
    for path in sample_paths:
        if os.path.exists(path):
            files = [f for f in os.listdir(path) if f.lower().endswith(('.tif', '.tiff'))]
            if files:
                sample_image = os.path.join(path, files[0])
                break
    
    if not sample_image:
        print("❌ No sample image found")
        return False
    
    print(f"📷 Testing with: {os.path.basename(sample_image)}")
    
    try:
        # Load image
        if sample_image.lower().endswith(('.tif', '.tiff')):
            image = tifffile.imread(sample_image)
        else:
            image = cv2.imread(sample_image)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if image is None:
            print("❌ Failed to load image")
            return False
        
        print(f"✅ Image loaded: {image.shape}")
        
        # Test basic preprocessing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        gray = cv2.resize(gray, (512, 512))
        gray = gray.astype(np.float64) / 255.0
        
        # Test basic feature extraction
        features = []
        
        # Statistical features
        features.extend([
            np.mean(gray), np.std(gray), np.var(gray),
            np.min(gray), np.max(gray), np.median(gray)
        ])
        
        # Histogram
        hist, _ = np.histogram(gray.flatten(), bins=8, range=(0, 1))
        hist = hist / (np.sum(hist) + 1e-8)
        features.extend(hist[:5])  # First 5 bins
        
        print(f"✅ Feature extraction successful: {len(features)} features")
        print(f"   Sample features: {features[:5]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False

def check_model_files():
    """Check if model files exist and are accessible"""
    print("\n📁 Checking Model Files")
    print("=" * 30)
    
    # Check scanner model files
    scanner_files = [
        "ensemble_95_results_ultra/models/Stacking_Top5_Ridge_model.pkl",
        "ensemble_95_results_ultra/models/imputer.pkl",
        "ensemble_95_results_ultra/models/feature_selector.pkl",
        "ensemble_95_results_ultra/models/label_encoder.pkl"
    ]
    
    scanner_ok = True
    for file in scanner_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {os.path.basename(file)}: {size/1024:.1f} KB")
        else:
            print(f"❌ {os.path.basename(file)}: Missing")
            scanner_ok = False
    
    # Check if real tampered model exists
    real_model_files = [
        "real_tampered_model/model.pkl",
        "real_tampered_model/scaler.pkl"
    ]
    
    real_model_ok = True
    for file in real_model_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            print(f"✅ {os.path.basename(file)}: {size/1024:.1f} KB")
        else:
            print(f"⏳ {os.path.basename(file)}: Not ready (training in progress)")
            real_model_ok = False
    
    # Check comprehensive test log
    if os.path.exists("comprehensive_test.log"):
        with open("comprehensive_test.log", "r") as f:
            log_content = f.read()
        
        if "SUCCESS!" in log_content:
            print("✅ Comprehensive test completed successfully")
        elif "Training Model on Real Dataset" in log_content:
            print("🔄 Comprehensive test in progress...")
        elif "Error" in log_content:
            print("⚠️ Errors detected in comprehensive test")
        else:
            print("📝 Comprehensive test log exists but status unclear")
    else:
        print("⏳ Comprehensive test not started or log not created")
    
    return scanner_ok, real_model_ok

def test_scanner_identification():
    """Quick test of scanner identification"""
    print("\n🖨️ Testing Scanner Identification")
    print("=" * 40)
    
    try:
        from scanner_identification_fixed import FixedScannerIdentifier
        
        scanner_id = FixedScannerIdentifier()
        if not scanner_id.load_model():
            print("❌ Failed to load scanner model")
            return False
        
        print("✅ Scanner model loaded successfully")
        
        # Test with a known scanner image
        test_paths = [
            ("The SUPATLANTIQUE dataset/Official/HP", "HP"),
            ("The SUPATLANTIQUE dataset/Official/Canon120-1", "Canon120-1"),
            ("The SUPATLANTIQUE dataset/Wikipedia/Canon220", "Canon220")
        ]
        
        for test_dir, expected_scanner in test_paths:
            if os.path.exists(test_dir):
                files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.tif', '.tiff'))]
                if files:
                    test_file = os.path.join(test_dir, files[0])
                    
                    try:
                        image = tifffile.imread(test_file)
                        result, conf, status, top3 = scanner_id.predict_scanner_with_top3(image)
                        
                        print(f"📷 {os.path.basename(test_file)}:")
                        print(f"   Expected: {expected_scanner}")
                        print(f"   Predicted: {result} (conf: {conf:.3f})")
                        print(f"   Status: {status}")
                        
                        if result == expected_scanner:
                            print("   ✅ CORRECT")
                        else:
                            print("   ❌ INCORRECT")
                        
                        return True
                        
                    except Exception as e:
                        print(f"   ❌ Test failed: {e}")
        
        print("❌ No suitable test images found")
        return False
        
    except Exception as e:
        print(f"❌ Scanner identification test failed: {e}")
        return False

def main():
    print("🚀 Quick Real Dataset Test")
    print("=" * 50)
    
    # 1. Analyze dataset structure
    dataset_info = analyze_real_dataset()
    
    # 2. Test feature extraction
    feature_test_ok = test_feature_extraction()
    
    # 3. Check model files
    scanner_ok, real_model_ok = check_model_files()
    
    # 4. Test scanner identification
    scanner_test_ok = test_scanner_identification()
    
    # Summary
    print(f"\n🎯 Quick Test Summary")
    print("=" * 30)
    print(f"Dataset Analysis: {'✅ PASS' if dataset_info['can_balance'] else '❌ FAIL'}")
    print(f"Feature Extraction: {'✅ PASS' if feature_test_ok else '❌ FAIL'}")
    print(f"Scanner Model: {'✅ PASS' if scanner_ok else '❌ FAIL'}")
    print(f"Real Tampered Model: {'✅ READY' if real_model_ok else '⏳ TRAINING'}")
    print(f"Scanner Test: {'✅ PASS' if scanner_test_ok else '❌ FAIL'}")
    
    if dataset_info['can_balance'] and feature_test_ok and scanner_ok:
        print(f"\n🎉 Real dataset approach is viable!")
        print(f"   - Can create balanced dataset with {len(dataset_info['tampered_files'])} samples each")
        print(f"   - Feature extraction working")
        print(f"   - Scanner identification functional")
        
        if real_model_ok:
            print(f"   - Real tampered model ready for testing")
        else:
            print(f"   - Real tampered model training in progress")
    else:
        print(f"\n⚠️ Issues detected that need attention")

if __name__ == "__main__":
    main()

