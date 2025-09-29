#!/usr/bin/env python3
"""
Quick test to verify the balanced dataset and check training progress
"""

import os
import sys
import random
import numpy as np
import cv2
import tifffile

def test_balanced_dataset():
    """Test the balanced dataset"""
    print("🧪 Testing Balanced Dataset")
    print("=" * 40)
    
    # Count original images
    original_dir = "The SUPATLANTIQUE dataset/Tampered images/Original"
    original_files = []
    if os.path.exists(original_dir):
        for file in os.listdir(original_dir):
            if file.lower().endswith(('.tif', '.tiff')):
                original_files.append(os.path.join(original_dir, file))
    
    # Count tampered images
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
                    tampered_files.append(os.path.join(tampered_dir, file))
    
    print(f"📊 Dataset Balance:")
    print(f"   Original images: {len(original_files)}")
    print(f"   Tampered images: {len(tampered_files)}")
    print(f"   Balance ratio: {len(original_files)/len(tampered_files):.2f}:1")
    
    if len(original_files) >= len(tampered_files):
        print("   ✅ Dataset is balanced!")
    else:
        print("   ⚠️ Dataset still imbalanced")
    
    # Test a few images
    print(f"\n🔍 Testing Sample Images:")
    
    # Test original image
    if original_files:
        test_original = random.choice(original_files)
        try:
            image = tifffile.imread(test_original)
            print(f"   Original: {os.path.basename(test_original)} - {image.shape} ✅")
        except Exception as e:
            print(f"   Original: {os.path.basename(test_original)} - Error: {e} ❌")
    
    # Test tampered image
    if tampered_files:
        test_tampered = random.choice(tampered_files)
        try:
            image = tifffile.imread(test_tampered)
            print(f"   Tampered: {os.path.basename(test_tampered)} - {image.shape} ✅")
        except Exception as e:
            print(f"   Tampered: {os.path.basename(test_tampered)} - Error: {e} ❌")
    
    return len(original_files), len(tampered_files)

def check_training_progress():
    """Check training progress"""
    print(f"\n🔄 Checking Training Progress")
    print("=" * 35)
    
    if os.path.exists("training_pipeline.log"):
        with open("training_pipeline.log", "r") as f:
            log_content = f.read()
        
        if "Training Complete!" in log_content:
            print("✅ Training completed successfully!")
            return True
        elif "Training" in log_content:
            print("🔄 Training in progress...")
            # Show last few lines
            lines = log_content.split('\n')
            recent_lines = [line for line in lines[-10:] if line.strip()]
            for line in recent_lines:
                print(f"   {line}")
            return False
        else:
            print("⏳ Training not started or log empty")
            return False
    else:
        print("❌ Training log not found")
        return False

def check_model_files():
    """Check if model files exist"""
    print(f"\n📁 Checking Model Files")
    print("=" * 25)
    
    model_dir = "balanced_tampered_models"
    if os.path.exists(model_dir):
        files = os.listdir(model_dir)
        print(f"   Model directory exists with {len(files)} files:")
        for file in files:
            size = os.path.getsize(os.path.join(model_dir, file))
            print(f"   - {file}: {size/1024:.1f} KB")
        
        # Check for specific model files
        model_files = [f for f in files if f.endswith('_model.pkl')]
        if model_files:
            print(f"   ✅ Found {len(model_files)} trained model(s)")
            return True
        else:
            print(f"   ⏳ No trained models found yet")
            return False
    else:
        print(f"   ❌ Model directory not found")
        return False

def main():
    print("🚀 Quick Balanced Dataset Test")
    print("=" * 50)
    
    # Test balanced dataset
    orig_count, tamp_count = test_balanced_dataset()
    
    # Check training progress
    training_complete = check_training_progress()
    
    # Check model files
    models_ready = check_model_files()
    
    # Summary
    print(f"\n🎯 Summary")
    print("=" * 15)
    print(f"Dataset balanced: {'✅' if orig_count >= tamp_count else '❌'}")
    print(f"Training complete: {'✅' if training_complete else '⏳'}")
    print(f"Models ready: {'✅' if models_ready else '⏳'}")
    
    if orig_count >= tamp_count and training_complete and models_ready:
        print(f"\n🎉 Everything ready for Streamlit app!")
        print(f"   Access: http://localhost:8512")
    elif orig_count >= tamp_count and not training_complete:
        print(f"\n⏳ Dataset balanced, training in progress...")
        print(f"   Check back in a few minutes")
    else:
        print(f"\n⚠️ Some issues detected")

if __name__ == "__main__":
    main()

