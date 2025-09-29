#!/usr/bin/env python3
"""
Comprehensive test using actual dataset to identify and fix tampered detection issues
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
import tifffile
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scanner_identification_fixed import FixedScannerIdentifier

class ComprehensiveDatasetTester:
    def __init__(self):
        self.dataset_path = "The SUPATLANTIQUE dataset"
        
    def collect_real_dataset(self):
        """Collect the actual dataset with proper labels"""
        print("🔍 Collecting Real Dataset")
        print("=" * 50)
        
        # Original images from Tampered images/Original/
        original_tampered_dir = []
        original_dir = os.path.join(self.dataset_path, "Tampered images", "Original")
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    original_tampered_dir.append(os.path.join(original_dir, file))
        
        print(f"📊 Original images (from Tampered images/Original): {len(original_tampered_dir)}")
        
        # Tampered images from Tampered images/Tampered/
        tampered_files = []
        tampered_dirs = [
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Copy-move"),
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Retouching"),
            os.path.join(self.dataset_path, "Tampered images", "Tampered", "Splicing")
        ]
        
        for tampered_dir in tampered_dirs:
            if os.path.exists(tampered_dir):
                for file in os.listdir(tampered_dir):
                    if file.lower().endswith(('.tif', '.tiff')):
                        tampered_files.append(os.path.join(tampered_dir, file))
        
        print(f"📊 Tampered images: {len(tampered_files)}")
        
        # Additional original images from Official and Wikipedia (sample to balance)
        additional_originals = []
        original_sources = [
            os.path.join(self.dataset_path, "Official"),
            os.path.join(self.dataset_path, "Wikipedia")
        ]
        
        for source_dir in original_sources:
            if os.path.exists(source_dir):
                for scanner_dir in os.listdir(source_dir):
                    scanner_path = os.path.join(source_dir, scanner_dir)
                    if os.path.isdir(scanner_path):
                        for file in os.listdir(scanner_path):
                            if file.lower().endswith(('.tif', '.tiff')):
                                additional_originals.append(os.path.join(scanner_path, file))
        
        # Balance the dataset - we need equal originals and tampered
        target_count = len(tampered_files)
        total_originals_needed = target_count - len(original_tampered_dir)
        
        if total_originals_needed > 0 and len(additional_originals) > 0:
            sampled_additional = random.sample(
                additional_originals, 
                min(total_originals_needed, len(additional_originals))
            )
            all_originals = original_tampered_dir + sampled_additional
        else:
            all_originals = original_tampered_dir
        
        # Balance by sampling tampered to match originals if needed
        if len(tampered_files) > len(all_originals):
            tampered_files = random.sample(tampered_files, len(all_originals))
        elif len(all_originals) > len(tampered_files):
            all_originals = random.sample(all_originals, len(tampered_files))
        
        print(f"📊 Final balanced dataset:")
        print(f"   Original images: {len(all_originals)}")
        print(f"   Tampered images: {len(tampered_files)}")
        
        return all_originals, tampered_files
    
    def extract_robust_features(self, image_path):
        """Extract robust features that work with real images"""
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
            
            # Convert to grayscale and resize for consistency
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Resize to manageable size
            gray = cv2.resize(gray, (512, 512))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Basic statistical features (10)
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size  # High intensity ratio
            ])
            
            # 2. Histogram features (8)
            hist, _ = np.histogram(gray.flatten(), bins=8, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            
            # 3. Edge detection features (8)
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
            
            # 5. Texture features (simplified to avoid warnings)
            try:
                # Use integer image for LBP
                gray_int = (gray * 255).astype(np.uint8)
                # Simple texture measure using variance of local patches
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                # Local standard deviation
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(sqr_filtered - mean_filtered**2)
                
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
            
            # 6. Frequency domain features (8)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                
                # Focus on central region
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
            
            return np.array(features[:48])  # Exactly 48 features
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def train_real_model(self):
        """Train model on real dataset"""
        print("\n🚀 Training Model on Real Dataset")
        print("=" * 50)
        
        # Collect real balanced dataset
        original_files, tampered_files = self.collect_real_dataset()
        
        # Extract features
        print("\n🔧 Extracting Features from Real Images...")
        X = []
        y = []
        
        # Process original images
        print("Processing original images...")
        for i, img_path in enumerate(original_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(original_files)}")
            
            features = self.extract_robust_features(img_path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 for original
        
        # Process tampered images
        print("Processing tampered images...")
        for i, img_path in enumerate(tampered_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tampered_files)}")
            
            features = self.extract_robust_features(img_path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 for tampered
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Final training dataset:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Original samples: {np.sum(y == 0)}")
        print(f"   Tampered samples: {np.sum(y == 1)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble model
        print("\n🎯 Training Ensemble Model...")
        
        ensemble = VotingClassifier([
            ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
        ], voting='soft')
        
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = ensemble.predict(X_test_scaled)
        y_pred_proba = ensemble.predict_proba(X_test_scaled)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   Test samples: {len(y_test)}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Original', 'Tampered']))
        
        # Save model
        os.makedirs("real_tampered_model", exist_ok=True)
        joblib.dump(ensemble, "real_tampered_model/model.pkl")
        joblib.dump(scaler, "real_tampered_model/scaler.pkl")
        
        print("✅ Real model saved to real_tampered_model/")
        
        return ensemble, scaler, accuracy
    
    def test_comprehensive_accuracy(self):
        """Test both scanner and tampered detection comprehensively"""
        print("\n🧪 Comprehensive Accuracy Test")
        print("=" * 50)
        
        # Load scanner model
        scanner_id = FixedScannerIdentifier()
        if not scanner_id.load_model():
            print("❌ Failed to load scanner model")
            return
        
        # Train and load tampered model
        tampered_model, tampered_scaler, _ = self.train_real_model()
        
        # Test scanner identification
        print("\n🖨️ Testing Scanner Identification...")
        scanner_test_dirs = [
            ("The SUPATLANTIQUE dataset/Official/HP", "HP"),
            ("The SUPATLANTIQUE dataset/Official/Canon120-1", "Canon120-1"),
            ("The SUPATLANTIQUE dataset/Official/EpsonV550", "EpsonV550"),
            ("The SUPATLANTIQUE dataset/Wikipedia/Canon220", "Canon220"),
            ("The SUPATLANTIQUE dataset/Wikipedia/EpsonV39-1", "EpsonV39-1")
        ]
        
        scanner_correct = 0
        scanner_total = 0
        
        for test_dir, expected_scanner in scanner_test_dirs:
            if os.path.exists(test_dir):
                files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.tif', '.tiff'))][:5]  # Test 5 per scanner
                
                for file in files:
                    try:
                        file_path = os.path.join(test_dir, file)
                        image = tifffile.imread(file_path)
                        
                        result, conf, status, top3 = scanner_id.predict_scanner_with_top3(image)
                        
                        print(f"Scanner Test: {file} -> Expected: {expected_scanner}, Got: {result}, Conf: {conf:.3f}")
                        
                        if result == expected_scanner:
                            scanner_correct += 1
                        scanner_total += 1
                        
                    except Exception as e:
                        print(f"❌ Scanner test error for {file}: {e}")
        
        # Test tampered detection
        print("\n🔒 Testing Tampered Detection...")
        
        # Test original images
        original_test_files = []
        original_dir = os.path.join(self.dataset_path, "Tampered images", "Original")
        if os.path.exists(original_dir):
            files = [f for f in os.listdir(original_dir) if f.lower().endswith(('.tif', '.tiff'))][:15]
            for file in files:
                original_test_files.append((os.path.join(original_dir, file), 0, "Original"))
        
        # Test tampered images
        tampered_test_files = []
        tampered_dirs = [
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Copy-move",
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching",
            "The SUPATLANTIQUE dataset/Tampered images/Tampered/Splicing"
        ]
        
        for tampered_dir in tampered_dirs:
            if os.path.exists(tampered_dir):
                files = [f for f in os.listdir(tampered_dir) if f.lower().endswith(('.tif', '.tiff'))][:5]
                for file in files:
                    tampered_test_files.append((os.path.join(tampered_dir, file), 1, "Tampered"))
        
        # Test tampered detection
        tampered_correct = 0
        tampered_total = 0
        
        all_test_files = original_test_files + tampered_test_files
        random.shuffle(all_test_files)
        
        for file_path, expected_label, expected_name in all_test_files[:30]:  # Test 30 total
            try:
                features = self.extract_robust_features(file_path)
                if features is not None:
                    features_scaled = tampered_scaler.transform(features.reshape(1, -1))
                    pred = tampered_model.predict(features_scaled)[0]
                    proba = tampered_model.predict_proba(features_scaled)[0]
                    conf = np.max(proba)
                    
                    pred_name = "Tampered" if pred == 1 else "Original"
                    
                    print(f"Tampered Test: {os.path.basename(file_path)} -> Expected: {expected_name}, Got: {pred_name}, Conf: {conf:.3f}")
                    
                    if pred == expected_label:
                        tampered_correct += 1
                    tampered_total += 1
                    
            except Exception as e:
                print(f"❌ Tampered test error for {file_path}: {e}")
        
        # Final results
        print(f"\n🎯 Final Comprehensive Results:")
        
        if scanner_total > 0:
            scanner_accuracy = scanner_correct / scanner_total
            print(f"Scanner ID: {scanner_correct}/{scanner_total} ({scanner_accuracy:.1%})")
        else:
            scanner_accuracy = 0
            print("Scanner ID: No tests completed")
        
        if tampered_total > 0:
            tampered_accuracy = tampered_correct / tampered_total
            print(f"Tampered Detection: {tampered_correct}/{tampered_total} ({tampered_accuracy:.1%})")
        else:
            tampered_accuracy = 0
            print("Tampered Detection: No tests completed")
        
        # Check if targets met
        scanner_pass = scanner_accuracy >= 0.8
        tampered_pass = tampered_accuracy >= 0.8
        
        print(f"\n🏆 Target Achievement:")
        print(f"Scanner ID (>80%): {'✅ PASS' if scanner_pass else '❌ FAIL'}")
        print(f"Tampered Detection (>80%): {'✅ PASS' if tampered_pass else '❌ FAIL'}")
        
        if scanner_pass and tampered_pass:
            print("\n🎉 SUCCESS! Both models meet target accuracy!")
        else:
            print("\n⚠️ Some models need improvement")
        
        return scanner_accuracy, tampered_accuracy

def main():
    tester = ComprehensiveDatasetTester()
    tester.test_comprehensive_accuracy()

if __name__ == "__main__":
    main()

