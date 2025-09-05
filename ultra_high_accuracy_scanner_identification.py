#!/usr/bin/env python3
"""
Ultra High Accuracy Scanner Identification System for AI_TraceFinder
Combines PRNU fingerprints, CNN-based features, advanced preprocessing, and ensemble methods
Target: 95%+ accuracy

Based on research:
- "Beyond PRNU: Learning Robust Device-Specific Fingerprint" 
- "ResNet101-based scanner identification without PRNU"
- "Multi-modal feature fusion for device identification"
- "Noiseprint: CNN-based camera model fingerprint"
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet101V2, EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import logging
from datetime import datetime
import warnings
import pywt
from scipy import ndimage, signal
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
import tifffile
import imageio.v2 as imageio
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class UltraHighAccuracyScannerID:
    def __init__(self, dataset_path="The SUPATLANTIQUE dataset", output_path="ultra_models"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.random_seed = 42
        self.img_size = (256, 256)
        self.batch_size = 16  # Smaller batch for better training
        self.epochs = 150
        self.learning_rate = 0.0005
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.prnu_fingerprints = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/trained_models",
            f"{self.output_path}/evaluation_results", 
            f"{self.output_path}/visualizations",
            f"{self.output_path}/metadata",
            f"{self.output_path}/checkpoints",
            f"{self.output_path}/preprocessed_data"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def enhanced_preprocessing(self, force_reprocess=False):
        """Enhanced preprocessing with better image loading and augmentation"""
        logger.info("Starting enhanced preprocessing...")
        
        metadata_file = f"{self.output_path}/preprocessed_data/enhanced_metadata.csv"
        
        if os.path.exists(metadata_file) and not force_reprocess:
            logger.info("Loading existing enhanced metadata...")
            return pd.read_csv(metadata_file)
        
        # Collect all images
        image_data = []
        
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.tif') and ('Official' in root or 'Wikipedia' in root):
                    # Skip Flatfield images for now
                    if 'Flatfield' in root:
                        continue
                        
                    file_path = os.path.join(root, file)
                    parts = root.split(os.sep)
                    
                    # Extract scanner info
                    scanner_id = None
                    source = None
                    dpi = None
                    
                    for part in parts:
                        if part in ['Official', 'Wikipedia']:
                            source = part
                        elif part in ['150', '300']:
                            dpi = int(part)
                        elif any(scanner in part for scanner in ['Canon', 'Epson', 'HP']):
                            scanner_id = part
                    
                    if scanner_id and source and dpi:
                        image_data.append({
                            'filepath': file_path,
                            'scanner_id': scanner_id, 
                            'dpi': dpi,
                            'source': source,
                            'filename': file
                        })
        
        df = pd.DataFrame(image_data)
        logger.info(f"Found {len(df)} images for processing")
        
        # Enhanced stratified split
        from sklearn.model_selection import train_test_split
        
        # Create stratification key
        df['strat_key'] = df['scanner_id'] + '_' + df['dpi'].astype(str)
        
        train_df, test_df = train_test_split(
            df, test_size=0.15, random_state=self.random_seed,
            stratify=df['strat_key']
        )
        
        train_df['split'] = 'train'
        test_df['split'] = 'test'
        
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Process and save images
        processed_data = []
        
        for idx, row in full_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing image {idx}/{len(full_df)}")
            
            try:
                # Load image with multiple attempts
                image = self.robust_image_load(row['filepath'])
                if image is None:
                    continue
                
                # Enhanced preprocessing
                processed_image = self.enhanced_image_preprocessing(image)
                
                # Save processed image
                output_dir = os.path.join(
                    self.output_path, "preprocessed_data", row['split'],
                    row['scanner_id'], str(row['dpi'])
                )
                os.makedirs(output_dir, exist_ok=True)
                
                output_path = os.path.join(output_dir, f"{row['filename'].replace('.tif', '.png')}")
                cv2.imwrite(output_path, processed_image)
                
                processed_data.append({
                    'filepath': output_path,
                    'original_filepath': row['filepath'],
                    'scanner_id': row['scanner_id'],
                    'dpi': row['dpi'],
                    'source': row['source'],
                    'split': row['split']
                })
                
            except Exception as e:
                logger.warning(f"Failed to process {row['filepath']}: {e}")
                continue
        
        processed_df = pd.DataFrame(processed_data)
        processed_df.to_csv(metadata_file, index=False)
        
        logger.info(f"Enhanced preprocessing complete. Processed {len(processed_df)} images")
        return processed_df
    
    def robust_image_load(self, filepath):
        """Robust image loading with multiple fallback methods"""
        try:
            # Try tifffile first
            image = tifffile.imread(filepath)
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            return image
        except:
            try:
                # Try imageio
                image = imageio.imread(filepath)
                if len(image.shape) == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                return image
            except:
                try:
                    # Try PIL
                    image = Image.open(filepath).convert('L')
                    return np.array(image)
                except:
                    try:
                        # Try OpenCV
                        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                        return image
                    except:
                        return None
    
    def enhanced_image_preprocessing(self, image):
        """Enhanced image preprocessing with noise reduction and normalization"""
        # Resize to target size
        image = cv2.resize(image, self.img_size)
        
        # Noise reduction
        image = cv2.bilateralFilter(image, 9, 75, 75)
        
        # Histogram equalization
        image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(image)
        
        # Normalize to [0, 255]
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image.astype(np.uint8)
    
    def extract_enhanced_prnu_fingerprints(self, metadata_df):
        """Extract enhanced PRNU fingerprints"""
        logger.info("Extracting enhanced PRNU fingerprints...")
        
        # Load flatfield images
        flatfield_images = {}
        
        for root, dirs, files in os.walk(self.dataset_path):
            if 'Flatfield' in root:
                for file in files:
                    if file.endswith('.tif'):
                        file_path = os.path.join(root, file)
                        parts = root.split(os.sep)
                        
                        scanner_id = None
                        dpi = None
                        
                        for part in parts:
                            if part in ['150', '300']:
                                dpi = int(part)
                            elif any(scanner in part for scanner in ['Canon', 'Epson', 'HP']):
                                scanner_id = part
                        
                        if scanner_id and dpi:
                            key = f"{scanner_id}_{dpi}"
                            if key not in flatfield_images:
                                flatfield_images[key] = []
                            
                            image = self.robust_image_load(file_path)
                            if image is not None:
                                flatfield_images[key].append(image)
        
        # Extract PRNU fingerprints
        prnu_fingerprints = {}
        
        for key, images in flatfield_images.items():
            if len(images) > 0:
                logger.info(f"Processing PRNU for {key}")
                
                residuals = []
                for image in images:
                    image = cv2.resize(image, self.img_size)
                    residual = self.extract_noise_residual(image)
                    residuals.append(residual)
                
                # Average residuals to create fingerprint
                if residuals:
                    fingerprint = np.mean(residuals, axis=0)
                    # Normalize fingerprint
                    fingerprint = (fingerprint - np.mean(fingerprint)) / (np.std(fingerprint) + 1e-8)
                    prnu_fingerprints[key] = fingerprint
        
        # Save fingerprints
        fingerprint_path = f"{self.output_path}/preprocessed_data/enhanced_prnu_fingerprints.pkl"
        with open(fingerprint_path, 'wb') as f:
            pickle.dump(prnu_fingerprints, f)
        
        self.prnu_fingerprints = prnu_fingerprints
        logger.info(f"Extracted PRNU fingerprints for {len(prnu_fingerprints)} scanner-DPI combinations")
        
        return prnu_fingerprints
    
    def extract_noise_residual(self, image):
        """Extract noise residual using advanced wavelet denoising"""
        # Convert to float
        image_float = image.astype(np.float32) / 255.0
        
        # Wavelet denoising
        coeffs = pywt.wavedec2(image_float, 'db8', mode='symmetric')
        sigma = 0.1
        coeffs_thresh = list(coeffs)
        coeffs_thresh[1:] = [pywt.threshold(detail, sigma, mode='soft') for detail in coeffs[1:]]
        
        # Reconstruct denoised image
        denoised = pywt.waverec2(coeffs_thresh, 'db8')
        
        # Calculate residual
        residual = image_float - denoised
        
        return residual
    
    def extract_comprehensive_features(self, metadata_df):
        """Extract comprehensive features for each image"""
        logger.info("Extracting comprehensive features...")
        
        features_list = []
        
        for idx, row in metadata_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Extracting features for image {idx}/{len(metadata_df)}")
            
            try:
                # Load preprocessed image
                image = cv2.imread(row['filepath'], cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue
                
                image = image.astype(np.float32) / 255.0
                
                # Extract all features
                feature_vector = self.extract_all_features(image, row['scanner_id'], row['dpi'])
                feature_vector.update({
                    'filepath': row['filepath'],
                    'scanner_id': row['scanner_id'],
                    'dpi': row['dpi'],
                    'source': row['source'],
                    'split': row['split']
                })
                
                features_list.append(feature_vector)
                
            except Exception as e:
                logger.warning(f"Failed to extract features from {row['filepath']}: {e}")
                continue
        
        features_df = pd.DataFrame(features_list)
        
        # Save features
        features_path = f"{self.output_path}/preprocessed_data/comprehensive_features.csv"
        features_df.to_csv(features_path, index=False)
        
        logger.info(f"Extracted features from {len(features_df)} images")
        return features_df
    
    def extract_all_features(self, image, scanner_id, dpi):
        """Extract all types of features from an image"""
        features = {}
        
        # 1. PRNU Correlation Features
        residual = self.extract_noise_residual(image)
        fingerprint_key = f"{scanner_id}_{dpi}"
        
        if fingerprint_key in self.prnu_fingerprints:
            fingerprint = self.prnu_fingerprints[fingerprint_key]
            # Resize fingerprint to match residual if needed
            if fingerprint.shape != residual.shape:
                fingerprint = cv2.resize(fingerprint, residual.shape[::-1])
            
            # Normalized cross-correlation
            correlation = signal.correlate2d(residual, fingerprint, mode='valid')
            features['prnu_correlation_max'] = np.max(correlation)
            features['prnu_correlation_mean'] = np.mean(correlation)
            features['prnu_correlation_std'] = np.std(correlation)
        else:
            features['prnu_correlation_max'] = 0
            features['prnu_correlation_mean'] = 0  
            features['prnu_correlation_std'] = 0
        
        # 2. Texture Features
        texture_features = self.extract_enhanced_texture_features(image)
        features.update(texture_features)
        
        # 3. Frequency Features
        frequency_features = self.extract_enhanced_frequency_features(image)
        features.update(frequency_features)
        
        # 4. Statistical Features
        statistical_features = self.extract_enhanced_statistical_features(image)
        features.update(statistical_features)
        
        # 5. Noise Features
        noise_features = self.extract_noise_features(image, residual)
        features.update(noise_features)
        
        return features
    
    def extract_enhanced_texture_features(self, image):
        """Extract enhanced texture features"""
        features = {}
        
        # Convert to uint8 for texture analysis
        image_uint8 = (image * 255).astype(np.uint8)
        
        # LBP features
        radius = 3
        n_points = 8 * radius
        lbp = local_binary_pattern(image_uint8, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, density=True)
        
        for i, val in enumerate(lbp_hist):
            features[f'lbp_hist_{i}'] = val
        
        # GLCM features
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        for dist in distances:
            for angle in angles:
                glcm = graycomatrix(image_uint8, [dist], [np.radians(angle)], 
                                 levels=256, symmetric=True, normed=True)
                
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                
                features[f'glcm_contrast_d{dist}_a{angle}'] = contrast
                features[f'glcm_dissimilarity_d{dist}_a{angle}'] = dissimilarity
                features[f'glcm_homogeneity_d{dist}_a{angle}'] = homogeneity
                features[f'glcm_energy_d{dist}_a{angle}'] = energy
                features[f'glcm_correlation_d{dist}_a{angle}'] = correlation
        
        # Gabor features
        frequencies = [0.1, 0.3, 0.5]
        orientations = [0, 45, 90, 135]
        
        for freq in frequencies:
            for orientation in orientations:
                filtered_real, _ = gabor(image, frequency=freq, 
                                       theta=np.radians(orientation))
                features[f'gabor_mean_f{freq}_o{orientation}'] = np.mean(filtered_real)
                features[f'gabor_std_f{freq}_o{orientation}'] = np.std(filtered_real)
        
        return features
    
    def extract_enhanced_frequency_features(self, image):
        """Extract enhanced frequency domain features"""
        features = {}
        
        # FFT analysis
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shifted)
        
        # Frequency bands analysis
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Low frequency (center region)
        low_freq = magnitude_spectrum[center_h-32:center_h+32, center_w-32:center_w+32]
        features['fft_low_freq_energy'] = np.sum(low_freq**2)
        features['fft_low_freq_mean'] = np.mean(low_freq)
        features['fft_low_freq_std'] = np.std(low_freq)
        
        # High frequency (edges)
        high_freq_mask = np.ones_like(magnitude_spectrum)
        high_freq_mask[center_h-32:center_h+32, center_w-32:center_w+32] = 0
        high_freq = magnitude_spectrum * high_freq_mask
        features['fft_high_freq_energy'] = np.sum(high_freq**2)
        features['fft_high_freq_mean'] = np.mean(high_freq)
        features['fft_high_freq_std'] = np.std(high_freq)
        
        # DCT analysis
        dct_coeffs = dct(dct(image, axis=0), axis=1)
        
        # DCT energy in different regions
        features['dct_total_energy'] = np.sum(dct_coeffs**2)
        features['dct_low_freq_energy'] = np.sum(dct_coeffs[:64, :64]**2)
        features['dct_high_freq_energy'] = np.sum(dct_coeffs[64:, 64:]**2)
        
        return features
    
    def extract_enhanced_statistical_features(self, image):
        """Extract enhanced statistical features"""
        features = {}
        
        # Basic statistics
        features['mean'] = np.mean(image)
        features['std'] = np.std(image)
        features['variance'] = np.var(image)
        features['skewness'] = skew(image.ravel())
        features['kurtosis'] = kurtosis(image.ravel())
        
        # Histogram features
        hist, _ = np.histogram(image.ravel(), bins=256, density=True)
        features['hist_entropy'] = -np.sum(hist * np.log(hist + 1e-8))
        
        # Percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        for p in percentiles:
            features[f'percentile_{p}'] = np.percentile(image, p)
        
        # Gradient statistics
        grad_x = np.gradient(image, axis=1)
        grad_y = np.gradient(image, axis=0)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        features['gradient_mean'] = np.mean(grad_magnitude)
        features['gradient_std'] = np.std(grad_magnitude)
        features['gradient_max'] = np.max(grad_magnitude)
        
        return features
    
    def extract_noise_features(self, image, residual):
        """Extract noise-specific features"""
        features = {}
        
        # Residual statistics
        features['residual_mean'] = np.mean(residual)
        features['residual_std'] = np.std(residual)
        features['residual_skewness'] = skew(residual.ravel())
        features['residual_kurtosis'] = kurtosis(residual.ravel())
        
        # Noise variance estimation
        features['noise_variance'] = np.var(residual)
        
        # Local noise patterns
        residual_uint8 = ((residual - np.min(residual)) / (np.max(residual) - np.min(residual)) * 255).astype(np.uint8)
        
        # LBP on residual
        lbp_residual = local_binary_pattern(residual_uint8, 8, 1, method='uniform')
        lbp_hist_residual, _ = np.histogram(lbp_residual.ravel(), bins=10, density=True)
        
        for i, val in enumerate(lbp_hist_residual):
            features[f'residual_lbp_hist_{i}'] = val
        
        return features
    
    def build_ultra_high_accuracy_model(self, num_classes, feature_dim):
        """Build ultra high accuracy model with advanced architecture"""
        logger.info("Building ultra high accuracy model...")
        
        # Image input branch
        img_input = layers.Input(shape=(self.img_size[0], self.img_size[1], 3), name='image_input')
        
        # Feature input branch
        feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
        
        # Advanced CNN branch with ResNet101V2
        base_model = ResNet101V2(
            weights='imagenet',
            include_top=False,
            input_tensor=img_input
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # CNN feature extraction
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        cnn_features = layers.Dense(256, activation='relu', name='cnn_features')(x)
        
        # Advanced feature branch
        feature_branch = layers.Dense(512, activation='relu')(feature_input)
        feature_branch = layers.BatchNormalization()(feature_branch)
        feature_branch = layers.Dropout(0.4)(feature_branch)
        feature_branch = layers.Dense(256, activation='relu')(feature_branch)
        feature_branch = layers.BatchNormalization()(feature_branch)
        feature_branch = layers.Dropout(0.3)(feature_branch)
        feature_branch = layers.Dense(128, activation='relu', name='feature_features')(feature_branch)
        
        # Attention mechanism for feature fusion
        attention = layers.Dense(384, activation='tanh')(layers.Concatenate()([cnn_features, feature_branch]))
        attention = layers.Dense(384, activation='sigmoid')(attention)
        
        # Apply attention
        attended_features = layers.Multiply()([layers.Concatenate()([cnn_features, feature_branch]), attention])
        
        # Final classification layers
        combined = layers.Dropout(0.5)(attended_features)
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.4)(combined)
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='output')(combined)
        
        # Create model
        model = models.Model(inputs=[img_input, feature_input], outputs=output)
        
        # Compile with advanced optimizer
        optimizer = optimizers.AdamW(learning_rate=self.learning_rate, weight_decay=0.01)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_ensemble_models(self, X_train, X_test, y_train, y_test):
        """Train ensemble of traditional ML models"""
        logger.info("Training ensemble models...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=500, max_depth=20, min_samples_split=2,
                min_samples_leaf=1, random_state=self.random_seed, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=self.random_seed
            ),
            'SVM': SVC(
                kernel='rbf', C=10, gamma='scale',
                random_state=self.random_seed, probability=True
            ),
            'LogisticRegression': LogisticRegression(
                C=1.0, random_state=self.random_seed, max_iter=1000
            )
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_scaled, y_train, 
                                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                                      scoring='f1_weighted')
            
            results[name] = {
                'model': model,
                'test_accuracy': accuracy,
                'test_f1': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}, CV F1: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
        
        return results
    
    def run_ultra_pipeline(self):
        """Run the complete ultra high accuracy pipeline"""
        logger.info("Starting ultra high accuracy scanner identification pipeline...")
        
        # 1. Enhanced preprocessing
        metadata_df = self.enhanced_preprocessing(force_reprocess=True)
        
        # 2. Extract PRNU fingerprints
        self.extract_enhanced_prnu_fingerprints(metadata_df)
        
        # 3. Extract comprehensive features
        features_df = self.extract_comprehensive_features(metadata_df)
        
        # 4. Prepare data for training
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = features_df[feature_columns].values
        y = features_df['scanner_id'].values
        split = features_df['split'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X[train_mask]
        y_train = y_encoded[train_mask]
        X_test = X[test_mask]
        y_test = y_encoded[test_mask]
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        logger.info(f"Feature dimension: {len(feature_columns)}")
        
        # 5. Train ensemble models
        ensemble_results = self.train_ensemble_models(X_train, X_test, y_train, y_test)
        
        # 6. Load image data for deep learning
        logger.info("Loading image data for deep learning...")
        train_df = features_df[features_df['split'] == 'train'].copy()
        test_df = features_df[features_df['split'] == 'test'].copy()
        
        X_train_img = self.load_images_for_training(train_df)
        X_test_img = self.load_images_for_training(test_df)
        
        if len(X_train_img) > 0 and len(X_test_img) > 0:
            # 7. Train deep learning model
            num_classes = len(self.label_encoder.classes_)
            y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
            y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
            
            ultra_model = self.build_ultra_high_accuracy_model(num_classes, len(feature_columns))
            
            # Train with callbacks
            callbacks = [
                ModelCheckpoint(
                    filepath=f"{self.output_path}/checkpoints/ultra_model_best.h5",
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=20,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=10,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            logger.info("Training ultra high accuracy model...")
            history = ultra_model.fit(
                [X_train_img, X_train],
                y_train_cat,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=([X_test_img, X_test], y_test_cat),
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate deep learning model
            y_pred_proba = ultra_model.predict([X_test_img, X_test])
            y_pred = np.argmax(y_pred_proba, axis=1)
            y_true = np.argmax(y_test_cat, axis=1)
            
            dl_accuracy = accuracy_score(y_true, y_pred)
            dl_f1 = f1_score(y_true, y_pred, average='weighted')
            
            logger.info(f"Deep Learning Model - Accuracy: {dl_accuracy:.4f}, F1: {dl_f1:.4f}")
        
        # 8. Print final results
        print("\n" + "="*80)
        print("ULTRA HIGH ACCURACY RESULTS")
        print("="*80)
        
        best_traditional_model = max(ensemble_results.items(), key=lambda x: x[1]['test_accuracy'])
        best_traditional_name = best_traditional_model[0]
        best_traditional_acc = best_traditional_model[1]['test_accuracy']
        
        print(f"Best Traditional Model: {best_traditional_name}")
        print(f"Best Traditional Accuracy: {best_traditional_acc:.4f}")
        
        if 'dl_accuracy' in locals():
            print(f"Deep Learning Model Accuracy: {dl_accuracy:.4f}")
            
            best_overall_acc = max(best_traditional_acc, dl_accuracy)
            best_overall_model = best_traditional_name if best_traditional_acc > dl_accuracy else "Deep Learning"
        else:
            best_overall_acc = best_traditional_acc
            best_overall_model = best_traditional_name
        
        print(f"\nBest Overall Model: {best_overall_model}")
        print(f"Best Overall Accuracy: {best_overall_acc:.4f}")
        
        if best_overall_acc >= 0.95:
            print("🎉 TARGET ACHIEVED: 95%+ accuracy reached!")
        else:
            print(f"⚠️  Target not reached. Need {0.95 - best_overall_acc:.4f} more accuracy.")
            print("Recommendations:")
            print("1. Collect more training data")
            print("2. Try advanced data augmentation")
            print("3. Experiment with different model architectures")
            print("4. Ensemble multiple models")
        
        logger.info("Ultra high accuracy pipeline complete!")
    
    def load_images_for_training(self, df):
        """Load images for deep learning training"""
        images = []
        
        for idx, row in df.iterrows():
            try:
                image = cv2.imread(row['filepath'])
                if image is not None:
                    image = cv2.resize(image, self.img_size)
                    image = image.astype(np.float32) / 255.0
                    images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {row['filepath']}: {e}")
                continue
        
        if images:
            return np.array(images)
        else:
            return np.array([])

if __name__ == "__main__":
    # Run ultra high accuracy scanner identification
    ultra_pipeline = UltraHighAccuracyScannerID()
    ultra_pipeline.run_ultra_pipeline()
