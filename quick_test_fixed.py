#!/usr/bin/env python3
"""
Quick test of the fixed models
"""

import sys
sys.path.append('.')
from scanner_identification_fixed import FixedScannerIdentifier
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
import cv2
import tifffile

class FixedTamperedDetector:
    """Same as in streamlit_fixed_final.py"""
    
    def __init__(self):
        self.model = joblib.load("new_objective2_baseline_ml_results/models/best_model.pkl")
        self.scaler = joblib.load("new_objective2_baseline_ml_results/models/scaler.pkl")
    
    def extract_simple_tampered_features(self, image):
        """Extract simplified tampered detection features"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize to standard size
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Basic statistical features
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray)
            ])
            
            # 2. Histogram features
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 1))
            hist = hist / np.sum(hist)  # Normalize
            features.extend(hist)
            
            # 3. Edge features
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            edges = edges.astype(np.float64) / 255.0
            features.extend([
                np.mean(edges), np.std(edges), np.sum(edges) / (edges.shape[0] * edges.shape[1])
            ])
            
            # 4. Texture features (LBP simplified)
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=10)
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            features.extend(lbp_hist)
            
            # 5. Frequency domain features
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            features.extend([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude)
            ])
            
            # 6. Gradient features
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            features.extend([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude)
            ])
            
            # 7. GLCM features (simplified)
            from skimage.feature import graycomatrix, graycoprops
            gray_int = (gray * 255).astype(np.uint8)
            glcm = graycomatrix(gray_int, [1], [0], levels=256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0]
            ])
            
            # Pad or truncate to 84 features
            features = np.array(features)
            if len(features) < 84:
                padding = np.zeros(84 - len(features))
                features = np.concatenate([features, padding])
            elif len(features) > 84:
                features = features[:84]
            
            return features
            
        except Exception as e:
            print(f"Error in feature extraction: {e}")
            return np.zeros(84)
    
    def predict_tampered(self, image):
        features = self.extract_simple_tampered_features(image)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(features_scaled)[0]
            pred = int(np.argmax(proba))
            confidence = float(np.max(proba))
        else:
            pred = int(self.model.predict(features_scaled)[0])
            confidence = 0.6
        
        return pred, confidence

def quick_test():
    print("🧪 Quick Test of Fixed Models")
    print("=" * 50)
    
    # Load models
    print("Loading models...")
    scanner_id = FixedScannerIdentifier()
    if not scanner_id.load_model():
        print("❌ Failed to load scanner model")
        return
    
    tampered_detector = FixedTamperedDetector()
    print("✅ Models loaded")
    
    # Test on a few images from the actual dataset structure
    test_images = [
        ("The SUPATLANTIQUE dataset/Official/HP/s11_42.tif", "HP", "Original"),
        ("The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching/s11_19_c.tif", "HP", "Tampered"),
        ("The SUPATLANTIQUE dataset/Official/Canon120-1/s1_69.tif", "Canon120-1", "Original"),
    ]
    
    scanner_correct = 0
    tampered_correct = 0
    total = 0
    
    for img_path, expected_scanner, expected_tampered_label in test_images:
        try:
            # Load image
            if img_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(img_path)
            else:
                image = cv2.imread(img_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                print(f"❌ Failed to load {img_path}")
                continue
            
            print(f"\n📁 Testing: {img_path}")
            print(f"Expected: Scanner={expected_scanner}, Tampered={expected_tampered_label}")
            
            # Scanner prediction
            scanner_pred, scanner_conf, _, _ = scanner_id.predict_scanner_with_top3(image)
            print(f"Scanner: {scanner_pred} (conf: {scanner_conf:.3f})")
            
            # Check scanner accuracy
            scanner_correct += (scanner_pred == expected_scanner)
            
            # Tampered prediction
            tampered_pred, tampered_conf = tampered_detector.predict_tampered(image)
            tampered_label = "Tampered" if tampered_pred == 1 else "Original"
            print(f"Tampered: {tampered_label} (conf: {tampered_conf:.3f})")
            
            # Check tampered accuracy
            expected_tampered = 1 if expected_tampered_label == "Tampered" else 0
            tampered_correct += (tampered_pred == expected_tampered)
            total += 1
            
        except Exception as e:
            print(f"❌ Error testing {img_path}: {e}")
    
    print(f"\n📊 Results:")
    if total > 0:
        print(f"Scanner accuracy: {scanner_correct}/{total} ({scanner_correct/total*100:.1f}%)")
        print(f"Tampered accuracy: {tampered_correct}/{total} ({tampered_correct/total*100:.1f}%)")
        
        if scanner_correct/total >= 0.8 and tampered_correct/total >= 0.8:
            print("✅ Both models meeting target accuracy!")
        else:
            print("⚠️ Models need further improvement")
    else:
        print("❌ No images were successfully tested")

if __name__ == "__main__":
    quick_test()
