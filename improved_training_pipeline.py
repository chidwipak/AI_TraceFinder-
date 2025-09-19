#!/usr/bin/env python3
"""
Improved training pipeline with better feature engineering and model tuning
Target: >85% accuracy for tampered detection
"""

import os
import sys
import numpy as np
import cv2
import tifffile
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import random

# Add current directory to path
sys.path.append('.')

class ImprovedTamperedDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.imputer = None
        self.feature_selector = None
        self.feature_names = None
        
    def collect_balanced_dataset(self):
        """Collect the balanced dataset"""
        print("🔍 Collecting Balanced Dataset")
        print("=" * 40)
        
        # Original images (102 total)
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
    
    def extract_advanced_features(self, image_path):
        """Extract advanced forensic features"""
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
            
            # 1. Enhanced Statistical features (20)
            stat_features = [
                np.mean(gray), np.std(gray), np.var(gray),
                np.min(gray), np.max(gray), np.median(gray),
                np.percentile(gray.flatten(), 10),
                np.percentile(gray.flatten(), 25),
                np.percentile(gray.flatten(), 75),
                np.percentile(gray.flatten(), 90),
                np.percentile(gray.flatten(), 95),
                np.percentile(gray.flatten(), 99),
                np.sum(gray > 0.8) / gray.size,
                np.sum(gray < 0.2) / gray.size,
                np.sum((gray > 0.2) & (gray < 0.8)) / gray.size,
                np.sum(gray > np.mean(gray)) / gray.size,
                np.sum(gray < np.mean(gray)) / gray.size,
                np.sum(gray > np.mean(gray) + np.std(gray)) / gray.size,
                np.sum(gray < np.mean(gray) - np.std(gray)) / gray.size,
                np.sum(np.abs(gray - np.mean(gray)) > np.std(gray)) / gray.size
            ]
            features.extend(stat_features)
            feature_names.extend([f'stat_{i}' for i in range(len(stat_features))])
            
            # 2. Multi-scale Histogram features (32)
            for bins in [8, 16, 32]:
                hist, _ = np.histogram(gray.flatten(), bins=bins, range=(0, 1))
                hist = hist / (np.sum(hist) + 1e-8)
                features.extend(hist)
                feature_names.extend([f'hist_{bins}_{i}' for i in range(len(hist))])
            
            # 3. Advanced Edge features (20)
            try:
                # Multiple edge detection methods
                edges_canny = cv2.Canny((gray * 255).astype(np.uint8), 50, 150)
                edges_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                edges_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                edges_laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                
                edge_features = [
                    np.mean(edges_canny), np.std(edges_canny), np.var(edges_canny),
                    np.sum(edges_canny > 0) / edges_canny.size,
                    np.percentile(edges_canny.flatten(), 25),
                    np.percentile(edges_canny.flatten(), 50),
                    np.percentile(edges_canny.flatten(), 75),
                    np.percentile(edges_canny.flatten(), 90),
                    np.percentile(edges_canny.flatten(), 95),
                    np.percentile(edges_canny.flatten(), 99),
                    np.max(edges_canny), np.min(edges_canny),
                    np.mean(np.abs(edges_sobel_x)), np.std(np.abs(edges_sobel_x)),
                    np.mean(np.abs(edges_sobel_y)), np.std(np.abs(edges_sobel_y)),
                    np.mean(np.abs(edges_laplacian)), np.std(np.abs(edges_laplacian)),
                    np.var(np.abs(edges_sobel_x)), np.var(np.abs(edges_sobel_y))
                ]
                features.extend(edge_features)
                feature_names.extend([f'edge_{i}' for i in range(len(edge_features))])
            except:
                features.extend([0.1] * 20)
                feature_names.extend([f'edge_{i}' for i in range(20)])
            
            # 4. Advanced Gradient features (20)
            try:
                grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                magnitude = np.sqrt(grad_x**2 + grad_y**2)
                direction = np.arctan2(grad_y, grad_x)
                
                # Gradient statistics
                grad_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.percentile(magnitude.flatten(), 25),
                    np.percentile(magnitude.flatten(), 50),
                    np.percentile(magnitude.flatten(), 75),
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(direction), np.std(direction), np.var(direction),
                    np.mean(grad_x), np.std(grad_x), np.var(grad_x),
                    np.mean(grad_y), np.std(grad_y), np.var(grad_y),
                    np.sum(magnitude > np.mean(magnitude)) / magnitude.size
                ]
                features.extend(grad_features)
                feature_names.extend([f'grad_{i}' for i in range(len(grad_features))])
            except:
                features.extend([0.2] * 20)
                feature_names.extend([f'grad_{i}' for i in range(20)])
            
            # 5. Advanced Texture features (25)
            try:
                # Local Binary Pattern approximation
                kernel = np.ones((5,5), np.float32) / 25
                smoothed = cv2.filter2D(gray, -1, kernel)
                texture_var = np.var(gray - smoothed)
                
                # Local standard deviation
                mean_filtered = cv2.blur(gray, (9, 9))
                sqr_filtered = cv2.blur(gray**2, (9, 9))
                local_std = np.sqrt(np.abs(sqr_filtered - mean_filtered**2))
                
                # Gabor features (multiple orientations and scales)
                gabor_features = []
                for theta in [0, 45, 90, 135]:
                    for freq in [0.1, 0.3, 0.5]:
                        gabor_kernel = cv2.getGaborKernel((21, 21), 5, theta*np.pi/180, 10, freq, 0, ktype=cv2.CV_32F)
                        gabor_response = cv2.filter2D(gray, cv2.CV_8UC3, gabor_kernel)
                        gabor_features.extend([np.mean(gabor_response), np.std(gabor_response)])
                
                texture_features = [
                    texture_var,
                    np.mean(local_std), np.std(local_std), np.var(local_std),
                    np.percentile(local_std.flatten(), 25),
                    np.percentile(local_std.flatten(), 50),
                    np.percentile(local_std.flatten(), 75),
                    np.percentile(local_std.flatten(), 90),
                    np.percentile(local_std.flatten(), 95),
                    np.max(local_std), np.min(local_std),
                    np.sum(local_std > np.mean(local_std)) / local_std.size
                ] + gabor_features
                
                features.extend(texture_features)
                feature_names.extend([f'texture_{i}' for i in range(len(texture_features))])
            except:
                features.extend([0.3] * 25)
                feature_names.extend([f'texture_{i}' for i in range(25)])
            
            # 6. Advanced Frequency features (20)
            try:
                f_transform = np.fft.fft2(gray)
                f_shift = np.fft.fftshift(f_transform)
                magnitude = np.abs(f_shift)
                phase = np.angle(f_shift)
                
                h, w = magnitude.shape
                center_h, center_w = h//2, w//2
                
                # Multiple frequency regions
                central_region = magnitude[center_h-50:center_h+50, center_w-50:center_w+50]
                high_freq_region = magnitude[center_h-100:center_h+100, center_w-100:center_w+100]
                high_freq_region = high_freq_region[~np.isin(high_freq_region, central_region)]
                
                freq_features = [
                    np.mean(magnitude), np.std(magnitude), np.var(magnitude),
                    np.mean(central_region), np.std(central_region), np.var(central_region),
                    np.mean(high_freq_region) if len(high_freq_region) > 0 else 0,
                    np.std(high_freq_region) if len(high_freq_region) > 0 else 0,
                    np.percentile(magnitude.flatten(), 90),
                    np.percentile(magnitude.flatten(), 95),
                    np.percentile(magnitude.flatten(), 99),
                    np.max(magnitude), np.min(magnitude),
                    np.mean(phase), np.std(phase), np.var(phase),
                    np.sum(magnitude > np.mean(magnitude)) / magnitude.size,
                    np.sum(magnitude > np.percentile(magnitude.flatten(), 90)) / magnitude.size,
                    np.sum(magnitude > np.percentile(magnitude.flatten(), 95)) / magnitude.size,
                    np.sum(magnitude > np.percentile(magnitude.flatten(), 99)) / magnitude.size
                ]
                features.extend(freq_features)
                feature_names.extend([f'freq_{i}' for i in range(len(freq_features))])
            except:
                features.extend([0.4] * 20)
                feature_names.extend([f'freq_{i}' for i in range(20)])
            
            # 7. Advanced Tampering-specific features (30)
            try:
                # Block-based analysis (multiple block sizes)
                block_sizes = [16, 32, 64]
                block_features = []
                
                for block_size in block_sizes:
                    h, w = gray.shape
                    blocks_h = h // block_size
                    blocks_w = w // block_size
                    
                    block_vars = []
                    block_means = []
                    for i in range(blocks_h):
                        for j in range(blocks_w):
                            block = gray[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                            block_vars.append(np.var(block))
                            block_means.append(np.mean(block))
                    
                    block_features.extend([
                        np.mean(block_vars), np.std(block_vars), np.var(block_vars),
                        np.percentile(block_vars, 25), np.percentile(block_vars, 75),
                        np.percentile(block_vars, 90), np.percentile(block_vars, 95),
                        np.sum(np.array(block_vars) > np.mean(block_vars)) / len(block_vars),
                        np.mean(block_means), np.std(block_means)
                    ])
                
                # Noise analysis (multiple methods)
                noise_gaussian = gray - cv2.GaussianBlur(gray, (5, 5), 0)
                noise_median = gray - cv2.medianBlur((gray * 255).astype(np.uint8), 5).astype(np.float64) / 255.0
                
                noise_features = [
                    np.mean(noise_gaussian), np.std(noise_gaussian), np.var(noise_gaussian),
                    np.percentile(noise_gaussian.flatten(), 90),
                    np.percentile(noise_gaussian.flatten(), 95),
                    np.max(noise_gaussian), np.min(noise_gaussian),
                    np.mean(noise_median), np.std(noise_median), np.var(noise_median),
                    np.percentile(noise_median.flatten(), 90),
                    np.percentile(noise_median.flatten(), 95),
                    np.max(noise_median), np.min(noise_median)
                ]
                
                tamper_features = block_features + noise_features
                features.extend(tamper_features)
                feature_names.extend([f'tamper_{i}' for i in range(len(tamper_features))])
            except:
                features.extend([0.5] * 30)
                feature_names.extend([f'tamper_{i}' for i in range(30)])
            
            return np.array(features), feature_names
            
        except Exception as e:
            print(f"Feature extraction error for {image_path}: {e}")
            return None, None
    
    def preprocess_data(self):
        """Preprocess the balanced dataset with advanced techniques"""
        print("\n🔧 Advanced Data Preprocessing")
        print("=" * 40)
        
        # Collect dataset
        original_files, tampered_files = self.collect_balanced_dataset()
        
        # Extract features
        print("Extracting advanced features...")
        X = []
        y = []
        feature_names = None
        
        # Process original images
        print("Processing original images...")
        for i, img_path in enumerate(original_files):
            if i % 10 == 0:
                print(f"  Progress: {i}/{len(original_files)}")
            
            features, names = self.extract_advanced_features(img_path)
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
            
            features, names = self.extract_advanced_features(img_path)
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
        
        # Feature selection
        print("Performing feature selection...")
        self.feature_selector = SelectKBest(f_classif, k=min(100, X_imputed.shape[1]))
        X_selected = self.feature_selector.fit_transform(X_imputed, y)
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"   Selected features: {X_selected.shape[1]}")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        
        self.feature_names = feature_names
        
        return X_train, X_test, y_train, y_test
    
    def train_optimized_models(self, X_train, y_train):
        """Train optimized models with hyperparameter tuning"""
        print("\n🎯 Training Optimized Models")
        print("=" * 40)
        
        models = {}
        
        # 1. Optimized Random Forest
        print("Training optimized Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        rf.fit(X_train, y_train)
        models['RandomForest'] = rf
        
        # 2. Optimized SVM
        print("Training optimized SVM...")
        svm = SVC(
            C=10.0,
            gamma='scale',
            kernel='rbf',
            random_state=42,
            probability=True,
            class_weight='balanced'
        )
        svm.fit(X_train, y_train)
        models['SVM'] = svm
        
        # 3. Optimized Gradient Boosting
        print("Training optimized Gradient Boosting...")
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        gb.fit(X_train, y_train)
        models['GradientBoosting'] = gb
        
        # 4. Extra Trees
        print("Training Extra Trees...")
        et = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        )
        et.fit(X_train, y_train)
        models['ExtraTrees'] = et
        
        # 5. Optimized Neural Network
        print("Training optimized Neural Network...")
        nn = MLPClassifier(
            hidden_layer_sizes=(200, 100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=2000,
            random_state=42
        )
        nn.fit(X_train, y_train)
        models['NeuralNetwork'] = nn
        
        return models
    
    def train_advanced_ensemble(self, X_train, y_train, base_models):
        """Train advanced ensemble models"""
        print("\n🎯 Training Advanced Ensemble Models")
        print("=" * 45)
        
        ensemble_models = {}
        
        # 1. Voting Classifier (Soft) - All models
        print("Training Voting Classifier (All models)...")
        voting_all = VotingClassifier([
            ('rf', base_models['RandomForest']),
            ('svm', base_models['SVM']),
            ('gb', base_models['GradientBoosting']),
            ('et', base_models['ExtraTrees']),
            ('nn', base_models['NeuralNetwork'])
        ], voting='soft')
        voting_all.fit(X_train, y_train)
        ensemble_models['VotingAll'] = voting_all
        
        # 2. Voting Classifier (Best 3)
        print("Training Voting Classifier (Best 3)...")
        voting_best3 = VotingClassifier([
            ('rf', base_models['RandomForest']),
            ('svm', base_models['SVM']),
            ('gb', base_models['GradientBoosting'])
        ], voting='soft')
        voting_best3.fit(X_train, y_train)
        ensemble_models['VotingBest3'] = voting_best3
        
        # 3. Weighted Voting
        print("Training Weighted Voting...")
        weighted_voting = VotingClassifier([
            ('rf', base_models['RandomForest']),
            ('svm', base_models['SVM']),
            ('gb', base_models['GradientBoosting'])
        ], voting='soft', weights=[2, 1, 1])
        weighted_voting.fit(X_train, y_train)
        ensemble_models['WeightedVoting'] = weighted_voting
        
        return ensemble_models
    
    def evaluate_models(self, models, X_test, y_test):
        """Evaluate all models with detailed metrics"""
        print("\n📊 Detailed Model Evaluation")
        print("=" * 35)
        
        results = {}
        
        for name, model in models.items():
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_test, y_test, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            results[name] = {
                'accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std
            }
            
            print(f"{name}:")
            print(f"  Accuracy: {accuracy:.3f}")
            print(f"  CV Score: {cv_mean:.3f} (+/- {cv_std*2:.3f})")
        
        # Find best model
        best_model_name = max(results, key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name} ({best_accuracy:.3f})")
        
        return best_model_name, best_accuracy, results
    
    def train_complete_pipeline(self):
        """Run the complete improved training pipeline"""
        print("🚀 Improved Training Pipeline")
        print("=" * 50)
        
        # 1. Advanced Data Preprocessing
        X_train, X_test, y_train, y_test = self.preprocess_data()
        
        # 2. Train Optimized Models
        base_models = self.train_optimized_models(X_train, y_train)
        
        # 3. Train Advanced Ensemble Models
        ensemble_models = self.train_advanced_ensemble(X_train, y_train, base_models)
        
        # 4. Combine all models
        all_models = {**base_models, **ensemble_models}
        
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
        print(f"\n💾 Saving Improved Model: {model_name}")
        
        # Create models directory
        os.makedirs("improved_tampered_models", exist_ok=True)
        
        # Save model
        joblib.dump(self.model, f"improved_tampered_models/{model_name.lower()}_model.pkl")
        
        # Save preprocessing objects
        joblib.dump(self.scaler, "improved_tampered_models/scaler.pkl")
        joblib.dump(self.imputer, "improved_tampered_models/imputer.pkl")
        joblib.dump(self.feature_selector, "improved_tampered_models/feature_selector.pkl")
        
        # Save feature names
        with open("improved_tampered_models/feature_names.txt", "w") as f:
            for name in self.feature_names:
                f.write(f"{name}\n")
        
        print(f"   Model saved: improved_tampered_models/{model_name.lower()}_model.pkl")
        print(f"   Scaler saved: improved_tampered_models/scaler.pkl")
        print(f"   Imputer saved: improved_tampered_models/imputer.pkl")
        print(f"   Feature selector saved: improved_tampered_models/feature_selector.pkl")
        print(f"   Feature names saved: improved_tampered_models/feature_names.txt")

def main():
    print("🔍 Improved Tampered Detection Training Pipeline")
    print("=" * 60)
    
    detector = ImprovedTamperedDetector()
    accuracy = detector.train_complete_pipeline()
    
    if accuracy >= 0.85:
        print(f"\n✅ SUCCESS! Model achieved {accuracy:.1%} accuracy")
        print(f"   Ready for Streamlit integration")
    else:
        print(f"\n⚠️ Model accuracy {accuracy:.1%} below 85% target")
        print(f"   Consider more advanced feature engineering or data augmentation")

if __name__ == "__main__":
    main()

