#!/usr/bin/env python3
"""
ACHIEVE 95% ACCURACY - Scanner Identification System
Comprehensive implementation of state-of-the-art techniques
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import pickle
import logging
from datetime import datetime
import warnings
from pathlib import Path

# Core ML imports
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Advanced ML imports
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Feature extraction
from skimage.feature import local_binary_pattern, hog, corner_harris, corner_peaks
from skimage.filters import gabor, sobel
from skimage.measure import regionprops, label
from scipy.stats import skew, kurtosis
from scipy.fftpack import dct
import pywt

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

class Achieve95PercentAccuracy:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "achieve_95_results"
        
        # Create output directories
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_path}/enhanced_features", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        
    def load_and_fix_test_data(self):
        """Fix the test data loading issue and create proper train/test split"""
        logger.info("="*80)
        logger.info("FIXING TEST DATA LOADING ISSUE")
        logger.info("="*80)
        
        # Load existing features
        features_file = "features/feature_vectors/scanner_features.csv"
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Load the original preprocessed metadata to get correct split
        train_meta = pd.read_csv("preprocessed/metadata/metadata_train.csv")
        test_meta = pd.read_csv("preprocessed/metadata/metadata_test.csv")
        
        logger.info(f"Original split: {len(train_meta)} train, {len(test_meta)} test")
        
        # Create mapping from original paths to preprocessed paths
        def get_preprocessed_path(original_path, split):
            parts = original_path.split('/')
            scanner = None
            dpi = None
            filename = parts[-1].replace('.tif', '.png')
            
            # Extract scanner and DPI
            for part in parts:
                if any(scanner_name in part for scanner_name in ['Canon', 'Epson', 'HP']):
                    scanner = part
                elif part in ['150', '300']:
                    dpi = part
            
            if scanner and dpi:
                return f"preprocessed/{split}/{scanner}/{dpi}/{filename}"
            return None
        
        # Fix the split column
        df['split'] = 'unknown'
        
        for idx, row in train_meta.iterrows():
            preprocessed_path = get_preprocessed_path(row['filepath'], 'train')
            if preprocessed_path:
                mask = df['filepath'] == preprocessed_path
                df.loc[mask, 'split'] = 'train'
        
        for idx, row in test_meta.iterrows():
            preprocessed_path = get_preprocessed_path(row['filepath'], 'test')
            if preprocessed_path:
                mask = df['filepath'] == preprocessed_path
                df.loc[mask, 'split'] = 'test'
        
        # If we still have issues, create a stratified split
        if df['split'].value_counts().get('test', 0) < 200:
            logger.info("Creating new stratified test split...")
            from sklearn.model_selection import train_test_split
            
            # Reset split
            df['split'] = 'train'
            
            # Create stratified split
            train_idx, test_idx = train_test_split(
                range(len(df)), 
                test_size=0.15, 
                random_state=self.random_seed,
                stratify=df['scanner_id']
            )
            
            df.iloc[test_idx, df.columns.get_loc('split')] = 'test'
        
        split_counts = df['split'].value_counts()
        logger.info(f"Fixed split: {split_counts}")
        
        return df
    
    def extract_advanced_features(self, df):
        """Extract cutting-edge features for 95%+ accuracy"""
        logger.info("="*80)
        logger.info("EXTRACTING ADVANCED FEATURES")
        logger.info("="*80)
        
        # Load images and extract advanced features
        enhanced_features = []
        
        for idx, row in df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing image {idx}/{len(df)}")
            
            try:
                # Load image
                if os.path.exists(row['filepath']):
                    img = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue
                    
                    # Resize for consistency
                    img = cv2.resize(img, (256, 256))
                    img_float = img.astype(np.float32) / 255.0
                    
                    # Extract advanced features
                    features = self.extract_comprehensive_features(img_float)
                    features.update({
                        'filepath': row['filepath'],
                        'scanner_id': row['scanner_id'],
                        'dpi': row['dpi'],
                        'source': row['source'],
                        'split': row['split']
                    })
                    enhanced_features.append(features)
                    
            except Exception as e:
                logger.warning(f"Failed to process {row['filepath']}: {e}")
                continue
        
        enhanced_df = pd.DataFrame(enhanced_features)
        logger.info(f"Enhanced features: {len(enhanced_df)} samples, {len(enhanced_df.columns)-5} features")
        
        # Save enhanced features
        enhanced_df.to_csv(f"{self.output_path}/enhanced_features/enhanced_features.csv", index=False)
        
        return enhanced_df
    
    def extract_comprehensive_features(self, img):
        """Extract comprehensive features using state-of-the-art techniques"""
        features = {}
        
        # 1. Edge-based features
        edges_canny = cv2.Canny((img * 255).astype(np.uint8), 50, 150)
        edges_sobel = sobel(img)
        
        features['edge_density_canny'] = np.mean(edges_canny > 0)
        features['edge_density_sobel'] = np.mean(edges_sobel > np.mean(edges_sobel))
        features['edge_variance_canny'] = np.var(edges_canny)
        features['edge_variance_sobel'] = np.var(edges_sobel)
        
        # 2. Corner detection
        corners = corner_harris(img)
        corner_peaks_coords = corner_peaks(corners, min_distance=5, threshold_rel=0.1)
        features['corner_count'] = len(corner_peaks_coords)
        features['corner_density'] = len(corner_peaks_coords) / (img.shape[0] * img.shape[1])
        
        # 3. Advanced texture features
        # LBP with multiple radii
        for radius in [1, 2, 3]:
            n_points = 8 * radius
            lbp = local_binary_pattern(img, n_points, radius, method='uniform')
            lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
            for i, val in enumerate(lbp_hist):
                features[f'lbp_r{radius}_hist_{i}'] = val
        
        # HOG features
        hog_features = hog(img, orientations=9, pixels_per_cell=(16, 16), 
                          cells_per_block=(2, 2), feature_vector=True)
        for i, val in enumerate(hog_features[:50]):  # Use first 50 HOG features
            features[f'hog_{i}'] = val
        
        # 4. Gabor filters
        for freq in [0.1, 0.3, 0.5]:
            for theta in [0, 45, 90, 135]:
                filtered_real, _ = gabor(img, frequency=freq, theta=np.radians(theta))
                features[f'gabor_f{freq}_t{theta}_mean'] = np.mean(filtered_real)
                features[f'gabor_f{freq}_t{theta}_std'] = np.std(filtered_real)
                features[f'gabor_f{freq}_t{theta}_energy'] = np.sum(filtered_real**2)
        
        # 5. Wavelet features
        coeffs = pywt.wavedec2(img, 'db4', levels=3)
        features['wavelet_approx_mean'] = np.mean(coeffs[0])
        features['wavelet_approx_std'] = np.std(coeffs[0])
        features['wavelet_approx_energy'] = np.sum(coeffs[0]**2)
        
        for i, detail in enumerate(coeffs[1:], 1):
            for j, coeff in enumerate(detail):
                features[f'wavelet_detail_l{i}_c{j}_mean'] = np.mean(coeff)
                features[f'wavelet_detail_l{i}_c{j}_std'] = np.std(coeff)
                features[f'wavelet_detail_l{i}_c{j}_energy'] = np.sum(coeff**2)
        
        # 6. DCT features
        dct_coeffs = dct(dct(img, axis=0), axis=1)
        features['dct_dc'] = dct_coeffs[0, 0]
        features['dct_low_freq_energy'] = np.sum(dct_coeffs[:32, :32]**2)
        features['dct_mid_freq_energy'] = np.sum(dct_coeffs[32:128, 32:128]**2)
        features['dct_high_freq_energy'] = np.sum(dct_coeffs[128:, 128:]**2)
        
        # 7. Statistical moments
        features['moment_2'] = np.mean((img - np.mean(img))**2)
        features['moment_3'] = np.mean((img - np.mean(img))**3)
        features['moment_4'] = np.mean((img - np.mean(img))**4)
        features['central_moment_2'] = np.var(img)
        features['central_moment_3'] = skew(img.ravel())
        features['central_moment_4'] = kurtosis(img.ravel())
        
        # 8. Morphological features
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img_uint8 = (img * 255).astype(np.uint8)
        
        opening = cv2.morphologyEx(img_uint8, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(img_uint8, cv2.MORPH_CLOSE, kernel)
        
        features['morph_opening_mean'] = np.mean(opening)
        features['morph_closing_mean'] = np.mean(closing)
        features['morph_opening_std'] = np.std(opening)
        features['morph_closing_std'] = np.std(closing)
        
        # 9. Frequency domain features
        fft = np.fft.fft2(img)
        fft_magnitude = np.abs(fft)
        fft_phase = np.angle(fft)
        
        features['fft_mag_mean'] = np.mean(fft_magnitude)
        features['fft_mag_std'] = np.std(fft_magnitude)
        features['fft_mag_skew'] = skew(fft_magnitude.ravel())
        features['fft_mag_kurtosis'] = kurtosis(fft_magnitude.ravel())
        features['fft_phase_std'] = np.std(fft_phase)
        
        # 10. Regional properties
        # Threshold image for region analysis
        thresh = img > np.mean(img)
        labeled_img = label(thresh)
        regions = regionprops(labeled_img)
        
        if regions:
            areas = [r.area for r in regions]
            eccentricities = [r.eccentricity for r in regions if hasattr(r, 'eccentricity')]
            
            features['region_count'] = len(regions)
            features['region_area_mean'] = np.mean(areas) if areas else 0
            features['region_area_std'] = np.std(areas) if areas else 0
            features['region_eccentricity_mean'] = np.mean(eccentricities) if eccentricities else 0
        else:
            features['region_count'] = 0
            features['region_area_mean'] = 0
            features['region_area_std'] = 0
            features['region_eccentricity_mean'] = 0
        
        return features
    
    def create_state_of_art_models(self):
        """Create state-of-the-art models for 95%+ accuracy"""
        logger.info("="*80)
        logger.info("CREATING STATE-OF-THE-ART MODELS")
        logger.info("="*80)
        
        models = {}
        
        # 1. Advanced Random Forest
        models['AdvancedRandomForest'] = RandomForestClassifier(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            oob_score=True,
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # 2. Extra Trees (Extremely Randomized Trees)
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # 3. XGBoost
        models['XGBoost'] = XGBClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # 4. LightGBM
        models['LightGBM'] = LGBMClassifier(
            n_estimators=1000,
            max_depth=10,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=-1
        )
        
        # 5. Advanced SVM
        models['AdvancedSVM'] = SVC(
            C=1000,
            gamma='scale',
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=self.random_seed
        )
        
        # 6. Neural Network
        models['NeuralNetwork'] = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            random_state=self.random_seed
        )
        
        # 7. Linear Discriminant Analysis
        models['LDA'] = LinearDiscriminantAnalysis()
        
        return models
    
    def create_ultimate_ensemble(self, models):
        """Create ultimate ensemble for maximum accuracy"""
        logger.info("Creating ultimate ensemble...")
        
        # Select best performing models for ensemble
        ensemble_models = [
            ('rf', models['AdvancedRandomForest']),
            ('et', models['ExtraTrees']),
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM']),
            ('svm', models['AdvancedSVM']),
            ('nn', models['NeuralNetwork'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',
            n_jobs=-1
        )
        
        return ensemble
    
    def comprehensive_evaluation(self, models, X_train, X_test, y_train, y_test):
        """Comprehensive evaluation with detailed analysis"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE MODEL EVALUATION")
        logger.info("="*80)
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # Cross-validation on training set
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_seed),
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'y_pred': y_pred,
                    'model': model
                }
                
                logger.info(f"{name}: Accuracy={accuracy:.4f}, CV={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")
                continue
        
        return results
    
    def save_final_results(self, results):
        """Save final results and create comprehensive report"""
        logger.info("Saving final results...")
        
        # Save models
        for name, result in results.items():
            model_path = f"{self.output_path}/models/{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Create results summary
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Test_Accuracy': result['accuracy'],
                'Test_Precision': result['precision'],
                'Test_Recall': result['recall'],
                'Test_F1': result['f1_score'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        summary_df.to_csv(f"{self.output_path}/results/final_results.csv", index=False)
        
        return summary_df
    
    def run_achieve_95_pipeline(self):
        """Run the complete pipeline to achieve 95%+ accuracy"""
        logger.info("="*80)
        logger.info("🎯 ACHIEVE 95% ACCURACY PIPELINE")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Fix test data loading issue
        df = self.load_and_fix_test_data()
        
        # 2. Check if we need to extract enhanced features
        enhanced_features_file = f"{self.output_path}/enhanced_features/enhanced_features.csv"
        if os.path.exists(enhanced_features_file):
            logger.info("Loading existing enhanced features...")
            enhanced_df = pd.read_csv(enhanced_features_file)
        else:
            # 3. Extract advanced features
            enhanced_df = self.extract_advanced_features(df)
        
        # 4. Prepare data
        feature_columns = [col for col in enhanced_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = enhanced_df[feature_columns].values
        y = enhanced_df['scanner_id'].values
        split = enhanced_df['split'].values
        
        # Handle missing values
        self.imputer = KNNImputer(n_neighbors=5)
        X = self.imputer.fit_transform(X)
        
        # Split data
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Feature selection
        logger.info("Performing feature selection...")
        selector = SelectKBest(mutual_info_classif, k=min(200, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train_encoded)
        X_test_selected = selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        logger.info(f"Final data: Train={X_train_scaled.shape}, Test={X_test_scaled.shape}")
        logger.info(f"Classes: {len(self.label_encoder.classes_)}")
        
        # 5. Create state-of-the-art models
        models = self.create_state_of_art_models()
        
        # 6. Create ultimate ensemble
        ensemble = self.create_ultimate_ensemble(models)
        models['UltimateEnsemble'] = ensemble
        
        # 7. Comprehensive evaluation
        results = self.comprehensive_evaluation(models, X_train_scaled, X_test_scaled, 
                                              y_train_encoded, y_test_encoded)
        
        # 8. Save results
        summary_df = self.save_final_results(results)
        
        # 9. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🏆 FINAL RESULTS - ACHIEVE 95% ACCURACY")
        print("="*80)
        
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        best_model = summary_df.iloc[0]
        best_accuracy = best_model['Test_Accuracy']
        
        print(f"\n🥇 BEST MODEL: {best_model['Model']}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"⏱️  TRAINING TIME: {duration}")
        
        if best_accuracy >= 0.95:
            print("\n🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉")
            print("✅ TARGET REACHED!")
            print("🏅 CONGRATULATIONS!")
        elif best_accuracy >= 0.90:
            gap = 0.95 - best_accuracy
            print(f"\n🔥 EXCELLENT! Almost there! Need {gap:.4f} ({gap*100:.2f}%) more")
            print("💪 Very close to target!")
        else:
            gap = 0.95 - best_accuracy
            print(f"\n📈 PROGRESS! Need {gap:.4f} ({gap*100:.2f}%) more accuracy")
            print("🚀 Keep improving!")
        
        print(f"\n📊 ANALYSIS:")
        print(f"• Training samples: {len(y_train_encoded)}")
        print(f"• Test samples: {len(y_test_encoded)}")
        print(f"• Scanner classes: {len(self.label_encoder.classes_)}")
        print(f"• Final features: {X_train_scaled.shape[1]}")
        print(f"• Models evaluated: {len(results)}")
        
        logger.info("🎯 Achieve 95% accuracy pipeline complete!")
        
        return results, best_accuracy

if __name__ == "__main__":
    # Run the achieve 95% accuracy pipeline
    system = Achieve95PercentAccuracy()
    results, best_accuracy = system.run_achieve_95_pipeline()
