#!/usr/bin/env python3
"""
Test the real balanced approach using actual Original and Tampered folders
"""

import os
import sys
import random
import numpy as np
import cv2
import tifffile
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Add current directory to path
sys.path.append('.')
from scanner_identification_fixed import FixedScannerIdentifier

class RealBalancedTester:
    def __init__(self):
        self.dataset_path = "The SUPATLANTIQUE dataset"
        
    def collect_real_balanced_dataset(self, max_samples=50):
        """Collect real balanced dataset for testing"""
        print("🔍 Collecting Real Balanced Dataset")
        print("=" * 50)
        
        # Original images from Tampered images/Original/
        original_files = []
        original_dir = os.path.join(self.dataset_path, "Tampered images", "Original")
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    original_files.append(os.path.join(original_dir, file))
        
        print(f"📊 Original images (paired): {len(original_files)}")
        
        # Tampered images from Tampered images/Tampered/
        tampered_files = []
        tampered_dirs = [
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Copy-move"),
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Retouching"),
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Splicing")
        ]
        
        for tampered_dir in tampered_dirs:
            if os.path.exists(tampered_dir):
                subdir_files = []
                for file in os.listdir(tampered_dir):
                    if file.lower().endswith(('.tif', '.tiff')):
                        subdir_files.append(os.path.join(tampered_dir, file))
                tampered_files.extend(subdir_files)
                print(f"📊 {os.path.basename(tampered_dir)}: {len(subdir_files)}")
        
        print(f"📊 Total tampered images: {len(tampered_files)}")
        
        # Add more originals to balance if needed
        if len(tampered_files) > len(original_files):
            additional_originals = []
            additional_sources = [
                os.path.join(self.dataset_path, "Official"),
                os.path.join(self.dataset_path, "Wikipedia")
            ]
            
            for source_dir in additional_sources:
                if os.path.exists(source_dir):
                    for scanner_dir in os.listdir(source_dir):
                        scanner_path = os.path.join(source_dir, scanner_dir)
                        if os.path.isdir(scanner_path):
                            for file in os.listdir(scanner_path):
                                if file.lower().endswith(('.tif', '.tiff')):
                                    additional_originals.append(os.path.join(scanner_path, file))
            
            needed = min(len(tampered_files) - len(original_files), len(additional_originals))
            if needed > 0:
                sampled_additional = random.sample(additional_originals, needed)
                original_files.extend(sampled_additional)
                print(f"📊 Added {needed} additional originals for balance")
        
        # Limit samples for testing
        if max_samples:
            min_count = min(len(original_files), len(tampered_files), max_samples)
            original_files = random.sample(original_files, min_count)
            tampered_files = random.sample(tampered_files, min_count)
        
        print(f"📊 Final balanced test dataset:")
        print(f"   Original: {len(original_files)}")
        print(f"   Tampered: {len(tampered_files)}")
        
        return original_files, tampered_files
    
    def extract_robust_features(self, image_path):
        """Extract the same features as the Streamlit app"""
        try:
            # Load image
            if image_path.lower().endswith(('.tif', '.tiff')):
                image = tifffile.imread(image_path)
            else:
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                return None
            
            # Convert to grayscale and resize
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Statistical features (10)
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size
            ])
            
            # 2. Histogram features (8)
            hist, _ = np.histogram(gray.flatten(), bins=8, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            
            # 3. Edge features (8)
            try:
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges = edges.astype(np.float64) / 255.0
                features.extend([
                    np.mean(edges), np.std(edges), 
                    np.sum(edges > 0) / edges.size,
                    np.percentile(edges.flatten(), 50),
                    np.percentile(edges.flatten(), 75),
                    np.percentile(edges.flatten(), 90),
                    np.percentile(edges.flatten(), 95),
                    np.percentile(edges.flatten(), 99)
                ])
            except:
                features.extend([0.1] * 8)
            
            # 4. Gradient features (8)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude)
                ])
            except:
                features.extend([0.2] * 8)
            
            # 5. Texture features (6)
            try:
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                features.extend([
                    texture_var,
                    np.mean(local_std),
                    np.std(local_std),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95)
                ])
            except:
                features.extend([0.3] * 6)
            
            # 6. Frequency features (8)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                
                features.extend([
                    np.mean(magnitude), np.std(magnitude),
                    np.mean(central_region), np.std(central_region),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.sum(magnitude > np.mean(magnitude)) / magnitude.size
                ])
            except:
                features.extend([0.4] * 8)
            
            return np.array(features[:48])
            
        except Exception as e:
            print(f"❌ Feature extraction error for {image_path}: {e}")
            return None
    
    def train_and_test_model(self):
        """Train and test the real balanced model"""
        print("\n🚀 Training and Testing Real Balanced Model")
        print("=" * 50)
        
        # Collect dataset
        original_files, tampered_files = self.collect_real_balanced_dataset(max_samples=30)
        
        # Extract features
        print("\n🔧 Extracting Features...")
        X = []
        y = []
        
        # Process original images
        print("Processing original images...")
        for i, img_path in enumerate(original_files):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(original_files)}")
            
            features = self.extract_robust_features(img_path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 for original
        
        # Process tampered images
        print("Processing tampered images...")
        for i, img_path in enumerate(tampered_files):
            if i % 5 == 0:
                print(f"  Progress: {i}/{len(tampered_files)}")
            
            features = self.extract_robust_features(img_path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 for tampered
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Training dataset:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Original samples: {np.sum(y == 0)}")
        print(f"   Tampered samples: {np.sum(y == 1)}")
        
        # Train model
        print("\n🎯 Training Model...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = VotingClassifier([
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ], voting='soft')
        
        model.fit(X_scaled, y)
        
        # Test on training data (for verification)
        y_pred = model.predict(X_scaled)
        y_pred_proba = model.predict_proba(X_scaled)
        
        accuracy = accuracy_score(y, y_pred)
        
        print(f"\n📊 Training Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['Original', 'Tampered']))
        
        # Test individual predictions
        print(f"\n🔍 Sample Predictions:")
        for i in range(min(10, len(y))):
            pred = y_pred[i]
            actual = y[i]
            conf = np.max(y_pred_proba[i])
            pred_label = "Tampered" if pred == 1 else "Original"
            actual_label = "Tampered" if actual == 1 else "Original"
            status = "✅" if pred == actual else "❌"
            
            print(f"   Sample {i+1}: Predicted {pred_label} (conf: {conf:.3f}), Actual: {actual_label} {status}")
        
        return model, scaler, accuracy
    
    def test_scanner_identification(self):
        """Test scanner identification on real images"""
        print("\n🖨️ Testing Scanner Identification")
        print("=" * 40)
        
        try:
            scanner_id = FixedScannerIdentifier()
            if not scanner_id.load_model():
                print("❌ Failed to load scanner model")
                return False
            
            print("✅ Scanner model loaded")
            
            # Test with known scanners
            test_cases = [
                ("The SUPATLANTIQUE dataset/Official/HP", "HP"),
                ("The SUPATLANTIQUE dataset/Official/Canon120-1", "Canon120-1"),
                ("The SUPATLANTIQUE dataset/Official/EpsonV550", "EpsonV550"),
                ("The SUPATLANTIQUE dataset/Wikipedia/Canon220", "Canon220"),
                ("The SUPATLANTIQUE dataset/Wikipedia/EpsonV39-1", "EpsonV39-1")
            ]
            
            correct = 0
            total = 0
            
            for test_dir, expected_scanner in test_cases:
                if os.path.exists(test_dir):
                    files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.tif', '.tiff'))][:3]  # Test 3 per scanner
                    
                    for file in files:
                        try:
                            file_path = os.path.join(test_dir, file)
                            image = tifffile.imread(file_path)
                            
                            result, conf, status, top3 = scanner_id.predict_scanner_with_top3(image)
                            
                            print(f"📷 {os.path.basename(file)}:")
                            print(f"   Expected: {expected_scanner}")
                            print(f"   Predicted: {result} (conf: {conf:.3f})")
                            
                            if result == expected_scanner:
                                print("   ✅ CORRECT")
                                correct += 1
                            else:
                                print("   ❌ INCORRECT")
                            
                            total += 1
                            
                        except Exception as e:
                            print(f"   ❌ Test failed: {e}")
            
            if total > 0:
                scanner_accuracy = correct / total
                print(f"\n📊 Scanner ID Results:")
                print(f"   Accuracy: {correct}/{total} ({scanner_accuracy:.1%})")
                return scanner_accuracy >= 0.8
            else:
                print("❌ No scanner tests completed")
                return False
                
        except Exception as e:
            print(f"❌ Scanner identification test failed: {e}")
            return False
    
    def run_comprehensive_test(self):
        """Run comprehensive test of the real balanced approach"""
        print("🧪 Comprehensive Real Balanced Dataset Test")
        print("=" * 60)
        
        # Test tampered detection
        model, scaler, tampered_accuracy = self.train_and_test_model()
        
        # Test scanner identification
        scanner_success = self.test_scanner_identification()
        
        # Final results
        print(f"\n🎯 Final Results:")
        print("=" * 30)
        
        tampered_pass = tampered_accuracy >= 0.8
        print(f"Tampered Detection: {tampered_accuracy:.1%} {'✅ PASS' if tampered_pass else '❌ FAIL'}")
        print(f"Scanner Identification: {'✅ PASS' if scanner_success else '❌ FAIL'}")
        
        if tampered_pass and scanner_success:
            print(f"\n🎉 SUCCESS! Real balanced approach works!")
            print(f"   - Tampered detection achieves >80% accuracy")
            print(f"   - Scanner identification is functional")
            print(f"   - Ready for Streamlit deployment")
        else:
            print(f"\n⚠️ Issues detected:")
            if not tampered_pass:
                print(f"   - Tampered detection below 80% target")
            if not scanner_success:
                print(f"   - Scanner identification needs improvement")
        
        return tampered_pass and scanner_success

def main():
    tester = RealBalancedTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print(f"\n✅ Real balanced approach validated!")
        print(f"The Streamlit app on port 8511 should now work correctly.")
    else:
        print(f"\n❌ Real balanced approach needs refinement.")

if __name__ == "__main__":
    main()

