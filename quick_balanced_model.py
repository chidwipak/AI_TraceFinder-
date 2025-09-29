#!/usr/bin/env python3
"""
Quick solution: Create balanced tampered detection model
"""

import os
import pandas as pd
import numpy as np
import cv2
import glob
import random
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, f1_score
from imblearn.over_sampling import SMOTE
import tifffile

class QuickBalancedModel:
    def __init__(self):
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.output_path = "quick_balanced_model"
        os.makedirs(self.output_path, exist_ok=True)
        
    def collect_balanced_data(self):
        """Collect 102 original + 102 tampered images"""
        print("🔍 Collecting Balanced Dataset")
        print("=" * 50)
        
        # Get tampered images (all 102)
        tampered_files = []
        tampered_path = os.path.join(self.dataset_path, "Tampered images", "Tampered")
        for root, dirs, files in os.walk(tampered_path):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff')):
                    tampered_files.append(os.path.join(root, file))
        
        print(f"📊 Tampered images: {len(tampered_files)}")
        
        # Get original images (sample 102 from existing)
        original_files = []
        original_dirs = [
            os.path.join(self.dataset_path, "Official"),
            os.path.join(self.dataset_path, "Wikipedia"),
            os.path.join(self.dataset_path, "Flatfield")
        ]
        
        for img_dir in original_dirs:
            if os.path.exists(img_dir):
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff')):
                            original_files.append(os.path.join(root, file))
        
        # Sample exactly 102 original images to match tampered
        target_originals = len(tampered_files)
        if len(original_files) > target_originals:
            original_files = random.sample(original_files, target_originals)
        
        print(f"📊 Original images: {len(original_files)}")
        print(f"📊 Total balanced dataset: {len(original_files) + len(tampered_files)}")
        
        return original_files, tampered_files
    
    def extract_fast_features(self, image_path):
        """Extract fast but effective features"""
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
            
            # Resize to standard size for faster processing
            gray = cv2.resize(gray, (256, 256))
            gray = gray.astype(np.float64) / 255.0
            
            features = []
            
            # 1. Statistical features (6)
            features.extend([
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray)
            ])
            
            # 2. Histogram features (16)
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 1))
            hist = hist / np.sum(hist)
            features.extend(hist)
            
            # 3. Edge features (6)
            edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
            edges = edges.astype(np.float64) / 255.0
            features.extend([
                np.mean(edges), np.std(edges), np.sum(edges) / edges.size,
                np.percentile(edges.flatten(), 25),
                np.percentile(edges.flatten(), 75),
                np.percentile(edges.flatten(), 90)
            ])
            
            # 4. Texture features - LBP (10)
            from skimage.feature import local_binary_pattern
            lbp = local_binary_pattern(gray, 8, 1, method='uniform')
            lbp_hist, _ = np.histogram(lbp.flatten(), bins=10)
            lbp_hist = lbp_hist / np.sum(lbp_hist)
            features.extend(lbp_hist)
            
            # 5. Frequency domain features (8)
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude = np.abs(f_shift)
            phase = np.angle(f_shift)
            features.extend([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude), np.min(magnitude),
                np.mean(phase), np.std(phase), np.max(phase), np.min(phase)
            ])
            
            # 6. Gradient features (8)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            direction = np.arctan2(grad_y, grad_x)
            features.extend([
                np.mean(magnitude), np.std(magnitude), np.max(magnitude), np.min(magnitude),
                np.mean(direction), np.std(direction), np.max(direction), np.min(direction)
            ])
            
            # 7. GLCM features (4)
            from skimage.feature import graycomatrix, graycoprops
            gray_int = (gray * 255).astype(np.uint8)
            glcm = graycomatrix(gray_int, [1], [0], levels=256, symmetric=True, normed=True)
            features.extend([
                graycoprops(glcm, 'contrast')[0, 0],
                graycoprops(glcm, 'dissimilarity')[0, 0],
                graycoprops(glcm, 'homogeneity')[0, 0],
                graycoprops(glcm, 'energy')[0, 0]
            ])
            
            # 8. Additional statistical features (6)
            features.extend([
                np.percentile(gray.flatten(), 10),
                np.percentile(gray.flatten(), 90),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.5) / gray.size,  # High intensity ratio
                np.sum(gray < 0.1) / gray.size,  # Low intensity ratio
                np.sum((gray > 0.4) & (gray < 0.6)) / gray.size  # Mid intensity ratio
            ])
            
            return np.array(features[:64])  # Exactly 64 features
            
        except Exception as e:
            print(f"❌ Error extracting features from {image_path}: {e}")
            return None
    
    def create_and_train_model(self):
        """Create balanced dataset and train model"""
        print("\n🚀 Creating and Training Balanced Model")
        print("=" * 50)
        
        # Collect balanced data
        original_files, tampered_files = self.collect_balanced_data()
        
        # Extract features
        print("\n🔧 Extracting Features...")
        X = []
        y = []
        
        # Process original images
        for i, img_path in enumerate(original_files):
            if i % 20 == 0:
                print(f"Processing original images: {i}/{len(original_files)}")
            
            features = self.extract_fast_features(img_path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 for original
        
        # Process tampered images
        for i, img_path in enumerate(tampered_files):
            if i % 20 == 0:
                print(f"Processing tampered images: {i}/{len(tampered_files)}")
            
            features = self.extract_fast_features(img_path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 for tampered
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Final dataset: {X.shape[0]} images with {X.shape[1]} features")
        print(f"📊 Class distribution: Original={np.sum(y==0)}, Tampered={np.sum(y==1)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"📊 Train set: {X_train.shape[0]} samples")
        print(f"📊 Test set: {X_test.shape[0]} samples")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train multiple models
        print("\n🎯 Training Models...")
        
        models = {
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced'),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
            'SVM': SVC(random_state=42, probability=True, class_weight='balanced')
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"  {name}: Accuracy={accuracy:.3f}, F1={f1:.3f}")
            
            if accuracy > best_score:
                best_score = accuracy
                best_model = model
                best_name = name
        
        # Create ensemble
        print("\n🔗 Creating Ensemble...")
        ensemble = VotingClassifier(
            estimators=[
                ('lr', LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')),
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
                ('svm', SVC(random_state=42, probability=True, class_weight='balanced'))
            ],
            voting='soft'
        )
        
        ensemble.fit(X_train_scaled, y_train)
        
        # Evaluate ensemble
        y_pred_ensemble = ensemble.predict(X_test_scaled)
        ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
        ensemble_f1 = f1_score(y_test, y_pred_ensemble)
        
        print(f"Ensemble: Accuracy={ensemble_accuracy:.3f}, F1={ensemble_f1:.3f}")
        
        # Choose best model
        if ensemble_accuracy > best_score:
            final_model = ensemble
            final_name = "Ensemble"
            final_accuracy = ensemble_accuracy
        else:
            final_model = best_model
            final_name = best_name
            final_accuracy = best_score
        
        print(f"\n🏆 Best Model: {final_name} with {final_accuracy:.3f} accuracy")
        
        # Save model and scaler
        model_path = os.path.join(self.output_path, "quick_tampered_model.pkl")
        scaler_path = os.path.join(self.output_path, "quick_tampered_scaler.pkl")
        
        joblib.dump(final_model, model_path)
        joblib.dump(scaler, scaler_path)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Scaler saved to {scaler_path}")
        
        # Detailed evaluation
        y_pred_final = final_model.predict(X_test_scaled)
        print(f"\n📊 Final Model Performance:")
        print(classification_report(y_test, y_pred_final, target_names=['Original', 'Tampered']))
        
        return final_model, scaler, final_accuracy

def main():
    model_creator = QuickBalancedModel()
    model_creator.create_and_train_model()

if __name__ == "__main__":
    main()

