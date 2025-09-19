#!/usr/bin/env python3
"""
Complete training pipeline for balanced tampered detection
Data preprocessing -> Feature extraction -> Baseline ML -> Deep Learning -> Ensemble -> Best model selection
"""

import os
import sys
import numpy as np
import cv2
import tifffile
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import random

# Add current directory to path
sys.path.append('.')

class BalancedTamperedDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.label_encoder = None
        self.feature_names = None
        
    def collect_balanced_dataset(self):
        """Collect the balanced dataset"""
        print("🔍 Collecting Balanced Dataset")
        print("=" * 40)
        
        # Original images (34 original + 68 converted = 102 total)
        original_files = []
        original_dir = "The SUPATLANTIQUE dataset/Tampered images/Original"
        if os.path.exists(original_dir):
            for file in os.listdir(original_dir):
                if file.lower().endswith(('.tif', '.tiff')):
                    original_files.append(os.path.join(original_dir, file))
        
        # Tampered images (102 total)
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
        
        print(f"📊 Dataset counts:")
        print(f"   Original images: {len(original_files)}")
        print(f"   Tampered images: {len(tampered_files)}")
        print(f"   Balance ratio: {len(original_files)/len(tampered_files):.2f}:1")
        
        return original_files, tampered_files
    
    def extract_comprehensive_features(self, image_path):
        """Extract comprehensive forensic features"""
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
            feature_names = []
            
            # 1. Statistical features (15)
            stat_features = [
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 90),
                np.percentile(gray.flatten(), 95),
                np.sum(gray > 0.8) / gray.size,
                np.sum(gray < 0.2) / gray.size,
                np.sum((gray > 0.2) & (gray < 0.8)) / gray.size,
                np.sum(gray > np.mean(gray)) / gray.size,
                np.sum(gray < np.mean(gray)) / gray.size
            ]
            features.extend(stat_features)
            feature_names.extend([f'stat_{i}' for i in range(len(stat_features))])
            
            # 2. Histogram features (16)
            hist, _ = np.histogram(gray.flatten(), bins=16, range=(0, 1))
            hist = hist / (np.sum(hist) + 1e-8)
            features.extend(hist)
            feature_names.extend([f'hist_{i}' for i in range(len(hist))])
            
            # 3. Edge features (12)
            try:
                edges = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges = edges.astype(np.float64) / 255.0
                edge_features = [
                    np.mean(edges), np.std(edges), np.var(edges),
                    np.sum(edges > 0) / edges.size,
                    np.percentile(edges.flatten(), 25),
                    np.percentile(edges.flatten(), 50),
                    np.percentile(edges.flatten(), 75),
                    np.percentile(edges.flatten(), 90),
                    np.percentile(edges.flatten(), 95),
                    np.percentile(edges.flatten(), 99),
                    np.max(edges), np.min(edges)
                ]
                features.extend(edge_features)
                feature_names.extend([f'edge_{i}' for i in range(len(edge_features))])
            except:
                features.extend([0.1] * 12)
                feature_names.extend([f'edge_{i}' for i in range(12)])
            
            # 4. Gradient features (12)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                
                grad_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.percentile(magnitude.flatten(), 25),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(direction), np.std(direction)
                ]
                features.extend(grad_features)
                feature_names.extend([f'grad_{i}' for i in range(len(grad_features))])
            except:
                features.extend([0.2] * 12)
                feature_names.extend([f'grad_{i}' for i in range(12)])
            
            # 5. Texture features (10)
            try:
                # Local Binary Pattern approximation
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                # Local standard deviation
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                # Gabor-like features
                gabor_kernel = cv2.getGaborKernel((21, 21), 5, 0, 10, 0.5, 0, ktype=cv2.CV_32F)
                gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
                
                texture_features = [
                    texture_var,
                    np.mean(local_std), np.std(local_std),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95),
                    np.mean(gabor_response), np.std(gabor_response),
                    np.max(gabor_response), np.min(gabor_response)
                ]
                features.extend(texture_features)
                feature_names.extend([f'texture_{i}' for i in range(len(texture_features))])
            except:
                features.extend([0.3] * 10)
                feature_names.extend([f'texture_{i}' for i in range(10)])
            
            # 6. Frequency features (12)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                phase = np.angle(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                
                freq_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.mean(central_region), np.std(central_region),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.percentile(magnitude.flatten(), 99),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(phase), np.std(phase)
                ]
                features.extend(freq_features)
                feature_names.extend([f'freq_{i}' for i in range(len(freq_features))])
            except:
                features.extend([0.4] * 12)
                feature_names.extend([f'freq_{i}' for i in range(12)])
            
            # 7. Tampering-specific features (15)
            try:
                # Block-based analysis
                block_size = 32
                h, w = gray.shape
                blocks_h = h // block_size
                blocks_w = w // block_size
                
                block_vars = []
                for i in range(blocks_h):
                    for j in range(blocks_w):
                        block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                        block_vars.append(np.var(block))
                
                # Copy-move detection features
                copy_move_features = [
                    np.mean(block_vars), np.std(block_vars),
                    np.percentile(block_vars, 25),
                    np.percentile(block_vars, 75),
                    np.percentile(block_vars, 90),
                    np.percentile(block_vars, 95),
                    np.sum(np.array(block_vars) > np.mean(block_vars)) / len(block_vars),
                    np.sum(np.array(block_vars) < np.mean(block_vars)) / len(block_vars)
                ]
                
                # Noise analysis
                noise = gray - cv2.GaussianBlur(gray, (5, 5), 0)
                noise_features = [
                    np.mean(noise), np.std(noise), np.var(noise),
                    np.percentile(noise.flatten(), 90),
                    np.percentile(noise.flatten(), 95),
                    np.max(noise), np.min(noise)
                ]
                
                tamper_features = copy_move_features + noise_features
                features.extend(tamper_features)
                feature_names.extend([f'tamper_{i}' for i in range(len(tamper_features))])
            except:
                features.extend([0.5] * 15)
                feature_names.extend([f'tamper_{i}' for i in range(15)])
            
            return np.array(features), feature_names
            
        except Exception as e:
            print(f"Feature extraction error for {image_path}: {e}")
            return None, None
    
    def preprocess_data(self):
        """Preprocess the balanced dataset"""
        print("\n🔧 Data Preprocessing")
        print("=" * 30)
        
        # Collect dataset
        original_files, tampered_files = self.collect_balanced_dataset()
        
        # Extract features
        print("Extracting features...")
        X = []
        y = []
        feature_names = None
        
        # Process original images
        print("Processing original images...")
        for i, img_path in enumerate(original_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(original_files)}")
            
            features, names = self.extract_comprehensive_features(img_path)
            if features is not None:
                X.append(features)
                y.append(0)  # 0 for original
                if feature_names is None:
                    feature_names = names
        
        # Process tampered images
        print("Processing tampered images...")
        for i, img_path in enumerate(tampered_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(tampered_files)}")
            
            features, names = self.extract_comprehensive_features(img_path)
            if features is not None:
                X.append(features)
                y.append(1)  # 1 for tampered
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"\n📊 Preprocessed dataset:")
        print(f"   Total samples: {X.shape[0]}")
        print(f"   Features per sample: {X.shape[1]}")
        print(f"   Original samples: {np.sum(y == 0)}")
        print(f"   Tampered samples: {np.sum(y == 1)}")
        
        # Handle missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = self.imputer.fit_transform(X)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        
        self.feature_names = feature_names
        
        return X_train, X_test, y_train, y_test
    
    def train_baseline_models(self, X_train, y_train):
        """Train baseline ML models"""
        print("\n🎯 Training Baseline ML Models")
        print("=" * 40)
        
        models = {}
        
        # 1. Logistic Regression
        print("Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr.fit(X_train, y_train)
        models['LogisticRegression'] = lr
        
        # 2. Random Forest
        print("Training Random Forest...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        
        # 3. SVM
        print("Training SVM...")
        svm = SVC(random_state=42, probability=True, class_weight='balanced')
        svm.fit(X_train, y_train)
        models['SVM'] = svm
        
        # 4. Gradient Boosting
        print("Training Gradient Boosting...")
        gb = GradientBoostingClassifier(random_state=42)
        gb.fit(X_train, y_train)
        models['GradientBoosting'] = gb
        
        # 5. Neural Network
        print("Training Neural Network...")
        nn = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
        nn.fit(X_train, y_train)
        models['NeuralNetwork'] = nn
        
        return models
    
    def train_ensemble_models(self, X_train, y_train, baseline_models):
        """Train ensemble models"""
        print("\n🎯 Training Ensemble Models")
        print("=" * 35)
        
        ensemble_models = {}
        
        # 1. Voting Classifier (Soft)
        print("Training Voting Classifier (Soft)...")
        voting_soft = VotingClassifier([
            ('lr', baseline_models['LogisticRegression']),
            ('rf', baseline_models['RandomForest']),
            ('svm', baseline_models['SVM']),
            ('gb', baseline_models['GradientBoosting'])
        ], voting='soft')
        voting_soft.fit(X_train, y_train)
        ensemble_models['VotingSoft'] = voting_soft
        
        # 2. Voting Classifier (Hard)
        print("Training Voting Classifier (Hard)...")
        voting_hard = VotingClassifier([
            ('lr', baseline_models['LogisticRegression']),
            ('rf', baseline_models['RandomForest']),
            ('svm', baseline_models['SVM']),
            ('gb', baseline_models['GradientBoosting'])
        ], voting='hard')
        voting_hard.fit(X_train, y_train)
        ensemble_models['VotingHard'] = voting_hard
        
        return ensemble_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models"""
        print("\n📊 Model Evaluation")
        print("=" * 25)
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[name] = accuracy
            
            print(f"{name}: {accuracy:.3f}")
        
        # Find best model
        best_model_name = max(results, key=results.get)
        best_accuracy = results[best_model_name]
        
        print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.3f})")
        
        return best_model_name, best_accuracy, results
    
    def train_complete_pipeline(self):
        """Run the complete training pipeline"""
        print("🚀 Complete Training Pipeline")
        print("=" * 50)
        
        # 1. Data Preprocessing
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # 2. Train Baseline Models
        baseline_models = self.train_baseline_models(X_train, y_train)
        
        # 3. Train Ensemble Models
        ensemble_models = self.train_ensemble_models(X_train, y_train, baseline_models)
        
        # 4. Combine all models
        all_models = {**baseline_models, **ensemble_models}
        
        # 5. Evaluate Models
        best_name, best_accuracy, all_results = self.evaluate_models(all_models, X_test, y_test)
        
        # 6. Select best model
        self.model = all_models[best_name]
        
        print(f"\n🎉 Training Complete!")
        print(f"   Best model: {best_name}")
        print(f"   Accuracy: {best_accuracy:.3f}")
        
        # Save model and preprocessing objects
        self.save_model(best_name)
        
        return best_accuracy
    
    def save_model(self, model_name):
        """Save the trained model and preprocessing objects"""
        print(f"\n💾 Saving Model: {model_name}")
        
        # Create models directory
        os.makedirs("balanced_tampered_models", exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"balanced_tampered_models/{model_name.lower()}_model.pkl")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, "balanced_tampered_models/scaler.pkl")
        joblib.dump(self.imputer, "balanced_tampered_models/imputer.pkl")
        
        # Save feature names
        with open("balanced_tampered_models/feature_names.txt", "w") as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"   Model saved: balanced_tampered_models/{model_name.lower()}_model.pkl")
        print(f"   Scaler saved: balanced_tampered_models/scaler.pkl")
        print(f"   Imputer saved: balanced_tampered_models/imputer.pkl")
        print(f"   Feature names saved: balanced_tampered_models/feature_names.txt")

def main():
    print("🔍 Balanced Tampered Detection Training Pipeline")
    print("=" * 60)
    
    detector = BalancedTamperedDetector()
    accuracy = detector.train_complete_pipeline()
    
    if accuracy >= 0.85:
        print(f"\n✅ SUCCESS! Model achieved {accuracy:.1%} accuracy")
        print(f"   Ready for Streamlit integration")
    else:
        print(f"\n⚠️ Model accuracy {accuracy:.1%} below 85% target")
        print(f"   Consider feature engineering or more data")

if __name__ == "__main__":
    main()

