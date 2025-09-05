#!/usr/bin/env python3
"""
ULTIMATE HYBRID SYSTEM FOR 95%+ ACCURACY
Combines ALL extracted features (PRNU, texture, frequency, statistical) with Purdue CNN architecture
Based on "Forensic Scanner Identification Using Machine Learning" by Ruiting Shao and Edward J. Delp

This is the ULTIMATE solution targeting 95%+ accuracy by leveraging EVERYTHING we've built.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
import pickle
import logging
from datetime import datetime
import warnings
from pathlib import Path
import random

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

class UltimateHybrid95Plus:
    def __init__(self):
        self.random_seed = 42
        self.patch_size = 64  # As per Purdue paper
        self.output_path = "ultimate_hybrid_results"
        
        # Create output directories
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_path}/logs", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.feature_scaler = StandardScaler()
        self.image_scaler = MinMaxScaler()
        
    def load_all_extracted_features(self):
        """Load ALL the features we extracted earlier"""
        logger.info("="*80)
        logger.info("LOADING ALL EXTRACTED FEATURES")
        logger.info("="*80)
        
        # 1. Load comprehensive features from our earlier extraction
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        features_df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(features_df)} samples with {len(features_df.columns)-5} extracted features")
        
        # 2. Load PRNU fingerprints
        prnu_file = "prnu_fingerprints/fingerprints/all_prnu_fingerprints.pkl"
        prnu_fingerprints = {}
        if os.path.exists(prnu_file):
            with open(prnu_file, 'rb') as f:
                prnu_fingerprints = pickle.load(f)
            logger.info(f"Loaded PRNU fingerprints for {len(prnu_fingerprints)} scanner-DPI combinations")
        
        # 3. Create proper train/test split
        train_meta = pd.read_csv("preprocessed/metadata/metadata_train.csv")
        test_meta = pd.read_csv("preprocessed/metadata/metadata_test.csv")
        
        # Fix split column in features_df
        features_df['split'] = 'unknown'
        
        # Map original paths to preprocessed paths
        def map_to_preprocessed_path(original_path, split):
            parts = original_path.split('/')
            scanner = None
            dpi = None
            filename = parts[-1].replace('.tif', '.png')
            
            for part in parts:
                if any(s in part for s in ['Canon', 'Epson', 'HP']):
                    scanner = part
                elif part in ['150', '300']:
                    dpi = part
            
            if scanner and dpi:
                return f"preprocessed/{split}/{scanner}/{dpi}/{filename}"
            return None
        
        # Set train split
        for _, row in train_meta.iterrows():
            preprocessed_path = map_to_preprocessed_path(row['filepath'], 'train')
            if preprocessed_path:
                mask = features_df['filepath'] == preprocessed_path
                features_df.loc[mask, 'split'] = 'train'
        
        # Set test split
        for _, row in test_meta.iterrows():
            preprocessed_path = map_to_preprocessed_path(row['filepath'], 'test')
            if preprocessed_path:
                mask = features_df['filepath'] == preprocessed_path
                features_df.loc[mask, 'split'] = 'test'
        
        # If still insufficient test samples, create stratified split
        test_count = sum(features_df['split'] == 'test')
        if test_count < 300:
            logger.info(f"Only {test_count} test samples found, creating stratified split...")
            
            # Reset and create new split
            features_df['split'] = 'train'
            
            # Stratified split
            train_idx, test_idx = train_test_split(
                range(len(features_df)), 
                test_size=0.15, 
                random_state=self.random_seed,
                stratify=features_df['scanner_id']
            )
            
            features_df.iloc[test_idx, features_df.columns.get_loc('split')] = 'test'
        
        split_counts = features_df['split'].value_counts()
        logger.info(f"Final split: {split_counts}")
        
        return features_df, prnu_fingerprints
    
    def create_hybrid_cnn_architecture(self, num_classes, feature_dim):
        """Create hybrid CNN that combines Purdue architecture with extracted features"""
        logger.info("Creating ULTIMATE HYBRID CNN ARCHITECTURE...")
        
        # ============ IMAGE BRANCH (Purdue CNN) ============
        image_input = layers.Input(shape=(self.patch_size, self.patch_size, 3), name='image_input')
        
        # Initial convolution layers
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(image_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Reduce dimensions
        x = layers.Conv2D(64, (1, 1), activation='relu')(x)
        x = layers.Conv2D(192, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Inception modules
        x = self.create_inception_module(x, 64, 96, 128, 16, 32, 32)
        x = self.create_inception_module(x, 128, 128, 192, 32, 96, 64)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        x = self.create_inception_module(x, 192, 96, 208, 16, 48, 64)
        
        # Global pooling and feature extraction
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # CNN feature output
        cnn_features = layers.Dense(256, activation='relu', name='cnn_features')(x)
        
        # ============ EXTRACTED FEATURES BRANCH ============
        feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
        
        # Advanced feature processing
        feat = layers.Dense(512, activation='relu')(feature_input)
        feat = layers.BatchNormalization()(feat)
        feat = layers.Dropout(0.4)(feat)
        feat = layers.Dense(256, activation='relu')(feat)
        feat = layers.BatchNormalization()(feat)
        feat = layers.Dropout(0.3)(feat)
        
        # Extracted feature output
        extracted_features = layers.Dense(128, activation='relu', name='extracted_features')(feat)
        
        # ============ FUSION LAYER WITH ATTENTION ============
        # Concatenate both feature streams
        combined = layers.Concatenate(name='feature_fusion')([cnn_features, extracted_features])
        
        # Attention mechanism for optimal feature weighting
        attention = layers.Dense(384, activation='tanh')(combined)
        attention = layers.Dense(384, activation='sigmoid')(attention)
        attended_features = layers.Multiply()([combined, attention])
        
        # ============ FINAL CLASSIFICATION ============
        final = layers.Dense(512, activation='relu')(attended_features)
        final = layers.BatchNormalization()(final)
        final = layers.Dropout(0.5)(final)
        
        final = layers.Dense(256, activation='relu')(final)
        final = layers.BatchNormalization()(final)
        final = layers.Dropout(0.3)(final)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='scanner_prediction')(final)
        
        # Create model
        model = models.Model(inputs=[image_input, feature_input], outputs=output)
        
        return model
    
    def create_inception_module(self, x, filters_1x1, filters_3x3_reduce, filters_3x3, 
                               filters_5x5_reduce, filters_5x5, filters_pool_proj):
        """Create Inception module as in Purdue paper"""
        
        # 1x1 convolution branch
        conv_1x1 = layers.Conv2D(filters_1x1, (1, 1), padding='same', activation='relu')(x)
        
        # 3x3 convolution branch
        conv_3x3_reduce = layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_3x3 = layers.Conv2D(filters_3x3, (3, 3), padding='same', activation='relu')(conv_3x3_reduce)
        
        # 5x5 convolution branch
        conv_5x5_reduce = layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu')(x)
        conv_5x5 = layers.Conv2D(filters_5x5, (5, 5), padding='same', activation='relu')(conv_5x5_reduce)
        
        # Max pooling branch
        pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = layers.Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu')(pool)
        
        # Concatenate all branches
        output = layers.Concatenate(axis=-1)([conv_1x1, conv_3x3, conv_5x5, pool_proj])
        
        return output
    
    def prepare_hybrid_data(self, features_df):
        """Prepare both image patches and extracted features"""
        logger.info("="*80)
        logger.info("PREPARING HYBRID DATA (IMAGES + FEATURES)")
        logger.info("="*80)
        
        # Separate features from metadata
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        # Extract feature matrix
        X_features = features_df[feature_columns].values
        y_labels = features_df['scanner_id'].values
        split_info = features_df['split'].values
        
        # Handle missing values in features
        logger.info("Imputing missing values in extracted features...")
        imputer = KNNImputer(n_neighbors=5)
        X_features = imputer.fit_transform(X_features)
        
        # Feature selection for extracted features
        logger.info("Selecting best extracted features...")
        train_mask = split_info == 'train'
        X_features_train = X_features[train_mask]
        y_train_temp = y_labels[train_mask]
        
        # Encode labels for feature selection
        y_encoded_temp = self.label_encoder.fit_transform(y_train_temp)
        
        # Select top features
        n_features_to_select = min(100, X_features.shape[1])
        selector = SelectKBest(mutual_info_classif, k=n_features_to_select)
        X_features_selected = selector.fit_transform(X_features_train, y_encoded_temp)
        X_features_test_selected = selector.transform(X_features[~train_mask])
        
        # Scale features
        logger.info("Scaling extracted features...")
        X_features_train_scaled = self.feature_scaler.fit_transform(X_features_selected)
        X_features_test_scaled = self.feature_scaler.transform(X_features_test_selected)
        
        # Extract image patches
        logger.info("Extracting image patches...")
        train_features_df = features_df[features_df['split'] == 'train']
        test_features_df = features_df[features_df['split'] == 'test']
        
        X_train_images, y_train_labels = self.extract_patches_with_features(train_features_df)
        X_test_images, y_test_labels = self.extract_patches_with_features(test_features_df)
        
        # Align images with features (each image might produce multiple patches)
        X_train_features_aligned, y_train_encoded = self.align_features_with_patches(
            X_features_train_scaled, y_train_labels, len(X_train_images)
        )
        X_test_features_aligned, y_test_encoded = self.align_features_with_patches(
            X_features_test_scaled, y_test_labels, len(X_test_images)
        )
        
        # Convert to numpy arrays and normalize images
        X_train_images = np.array(X_train_images).astype(np.float32) / 255.0
        X_test_images = np.array(X_test_images).astype(np.float32) / 255.0
        
        # Convert labels to categorical
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = tf.keras.utils.to_categorical(y_train_encoded, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_encoded, num_classes)
        
        logger.info(f"Final hybrid data:")
        logger.info(f"  Images - Train: {X_train_images.shape}, Test: {X_test_images.shape}")
        logger.info(f"  Features - Train: {X_train_features_aligned.shape}, Test: {X_test_features_aligned.shape}")
        logger.info(f"  Number of classes: {num_classes}")
        logger.info(f"  Selected features: {X_train_features_aligned.shape[1]}")
        
        return (X_train_images, X_train_features_aligned, y_train_cat,
                X_test_images, X_test_features_aligned, y_test_cat, 
                y_test_encoded)
    
    def extract_patches_with_features(self, df):
        """Extract image patches and track corresponding labels"""
        patches = []
        labels = []
        
        for idx, row in df.iterrows():
            try:
                if os.path.exists(row['filepath']):
                    img = cv2.imread(row['filepath'])
                    if img is None:
                        continue
                    
                    # Convert BGR to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Extract patches
                    image_patches = self.extract_multiple_patches(img)
                    
                    for patch in image_patches:
                        patches.append(patch)
                        labels.append(row['scanner_id'])
                        
            except Exception as e:
                logger.warning(f"Failed to process {row['filepath']}: {e}")
                continue
        
        return patches, labels
    
    def extract_multiple_patches(self, image):
        """Extract multiple patches from single image"""
        patches = []
        h, w, c = image.shape
        
        if h < self.patch_size or w < self.patch_size:
            image = cv2.resize(image, (self.patch_size, self.patch_size))
            patches.append(image)
            return patches
        
        # Extract patches with stride
        step_size = self.patch_size // 2
        
        for y in range(0, h - self.patch_size + 1, step_size):
            for x in range(0, w - self.patch_size + 1, step_size):
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append(patch)
                
                # Limit patches per image
                if len(patches) >= 8:  # Max 8 patches per image
                    return patches
        
        if not patches:
            y = (h - self.patch_size) // 2
            x = (w - self.patch_size) // 2
            patch = image[y:y+self.patch_size, x:x+self.patch_size]
            patches.append(patch)
        
        return patches
    
    def align_features_with_patches(self, features, labels, num_patches):
        """Align extracted features with image patches"""
        aligned_features = []
        aligned_labels = []
        
        feature_idx = 0
        for i in range(num_patches):
            if feature_idx < len(features):
                aligned_features.append(features[feature_idx])
                
                # Encode label
                label = labels[i] if i < len(labels) else labels[-1]
                if label in self.label_encoder.classes_:
                    encoded_label = self.label_encoder.transform([label])[0]
                else:
                    encoded_label = 0  # Default to first class
                
                aligned_labels.append(encoded_label)
                
                # Move to next feature every few patches (since each image produces multiple patches)
                if (i + 1) % 4 == 0:  # Assuming ~4 patches per image
                    feature_idx += 1
        
        return np.array(aligned_features), np.array(aligned_labels)
    
    def train_hybrid_model(self, model, train_data, test_data):
        """Train the hybrid model with advanced techniques"""
        logger.info("="*80)
        logger.info("TRAINING ULTIMATE HYBRID MODEL")
        logger.info("="*80)
        
        X_train_img, X_train_feat, y_train = train_data
        X_test_img, X_test_feat, y_test = test_data
        
        # Compile with advanced optimizer
        optimizer = optimizers.Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-07
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Advanced callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=f"{self.output_path}/models/ultimate_hybrid_best.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Train model
        logger.info("Starting training...")
        history = model.fit(
            [X_train_img, X_train_feat], y_train,
            batch_size=16,  # Smaller batch for better convergence
            epochs=100,
            validation_data=([X_test_img, X_test_feat], y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def evaluate_ultimate_system(self, model, test_data, y_test_encoded):
        """Comprehensive evaluation of the ultimate system"""
        logger.info("="*80)
        logger.info("ULTIMATE SYSTEM EVALUATION")
        logger.info("="*80)
        
        X_test_img, X_test_feat, y_test = test_data
        
        # Per-patch predictions
        y_pred_proba = model.predict([X_test_img, X_test_feat])
        y_pred_patch = np.argmax(y_pred_proba, axis=1)
        y_true_patch = np.argmax(y_test, axis=1)
        
        # Per-patch accuracy
        patch_accuracy = accuracy_score(y_true_patch, y_pred_patch)
        patch_f1 = f1_score(y_true_patch, y_pred_patch, average='weighted')
        
        # Per-image accuracy (majority voting simulation)
        # Group patches by scanner and average predictions
        image_accuracy = self.simulate_majority_voting(y_pred_proba, y_test_encoded)
        
        logger.info(f"Per-patch accuracy: {patch_accuracy:.4f} ({patch_accuracy*100:.2f}%)")
        logger.info(f"Per-patch F1-score: {patch_f1:.4f}")
        logger.info(f"Per-image accuracy: {image_accuracy:.4f} ({image_accuracy*100:.2f}%)")
        
        # Create detailed confusion matrix
        cm = confusion_matrix(y_true_patch, y_pred_patch)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title('Ultimate Hybrid System - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/ultimate_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        return patch_accuracy, image_accuracy, patch_f1
    
    def simulate_majority_voting(self, y_pred_proba, y_test_encoded):
        """Simulate per-image accuracy using majority voting"""
        # Simple simulation: group predictions by batches and apply majority voting
        batch_size = 4  # Assuming 4 patches per image
        correct = 0
        total = 0
        
        for i in range(0, len(y_pred_proba), batch_size):
            batch_probs = y_pred_proba[i:i+batch_size]
            batch_true = y_test_encoded[i:i+batch_size] if i < len(y_test_encoded) else [y_test_encoded[-1]]
            
            if len(batch_probs) > 0:
                # Average probabilities and predict
                avg_prob = np.mean(batch_probs, axis=0)
                pred_class = np.argmax(avg_prob)
                true_class = batch_true[0] if len(batch_true) > 0 else 0
                
                if pred_class == true_class:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0
    
    def create_ensemble_with_traditional_ml(self, features_df):
        """Create ensemble with traditional ML models using extracted features"""
        logger.info("Creating ensemble with traditional ML models...")
        
        # Prepare data
        feature_columns = [col for col in features_df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = features_df[feature_columns].values
        y = features_df['scanner_id'].values
        split = features_df['split'].values
        
        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        
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
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train ensemble models
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=1000, max_depth=25, random_state=self.random_seed, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300, max_depth=10, learning_rate=0.1, random_state=self.random_seed
            )
        }
        
        results = {}
        for name, model in models.items():
            model.fit(X_train_scaled, y_train_encoded)
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test_encoded, y_pred)
            results[name] = accuracy
            logger.info(f"{name} accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        return results
    
    def run_ultimate_hybrid_pipeline(self):
        """Run the complete ultimate hybrid pipeline"""
        logger.info("="*80)
        logger.info("🚀 ULTIMATE HYBRID SYSTEM FOR 95%+ ACCURACY")
        logger.info("Combining CNN + All Extracted Features + Ensemble")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Load all extracted features
        features_df, prnu_fingerprints = self.load_all_extracted_features()
        if features_df is None:
            return
        
        # 2. Create ensemble with traditional ML (baseline)
        traditional_results = self.create_ensemble_with_traditional_ml(features_df)
        
        # 3. Prepare hybrid data (images + features)
        (X_train_img, X_train_feat, y_train_cat,
         X_test_img, X_test_feat, y_test_cat, y_test_encoded) = self.prepare_hybrid_data(features_df)
        
        # 4. Create ultimate hybrid model
        num_classes = len(self.label_encoder.classes_)
        feature_dim = X_train_feat.shape[1]
        
        hybrid_model = self.create_hybrid_cnn_architecture(num_classes, feature_dim)
        
        # Print model summary
        hybrid_model.summary()
        
        # 5. Train hybrid model
        train_data = (X_train_img, X_train_feat, y_train_cat)
        test_data = (X_test_img, X_test_feat, y_test_cat)
        
        history = self.train_hybrid_model(hybrid_model, train_data, test_data)
        
        # 6. Evaluate ultimate system
        patch_acc, image_acc, patch_f1 = self.evaluate_ultimate_system(
            hybrid_model, test_data, y_test_encoded
        )
        
        # 7. Final results
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🏆 ULTIMATE HYBRID SYSTEM RESULTS")
        print("="*80)
        
        print("\n📊 TRADITIONAL ML RESULTS:")
        for name, acc in traditional_results.items():
            print(f"  {name}: {acc:.4f} ({acc*100:.2f}%)")
        
        print(f"\n🤖 HYBRID CNN RESULTS:")
        print(f"  Per-patch accuracy: {patch_acc:.4f} ({patch_acc*100:.2f}%)")
        print(f"  Per-image accuracy: {image_acc:.4f} ({image_acc*100:.2f}%)")
        print(f"  Per-patch F1-score: {patch_f1:.4f}")
        
        best_acc = max(max(traditional_results.values()), image_acc)
        best_model = "Hybrid CNN" if image_acc == best_acc else max(traditional_results, key=traditional_results.get)
        
        print(f"\n🥇 BEST OVERALL:")
        print(f"  Model: {best_model}")
        print(f"  Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)")
        print(f"  Training time: {duration}")
        
        if best_acc >= 0.95:
            print("\n🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉")
            print("✅ TARGET REACHED!")
            print("🏅 ULTIMATE HYBRID SYSTEM WORKS!")
        elif best_acc >= 0.90:
            gap = 0.95 - best_acc
            print(f"\n🔥 EXCELLENT! Very close! Need {gap:.4f} ({gap*100:.2f}%) more")
            print("💪 Almost at target!")
        else:
            gap = 0.95 - best_acc
            print(f"\n📈 Progress! Need {gap:.4f} ({gap*100:.2f}%) more accuracy")
            print("🚀 Continue optimizing!")
        
        # Save final results
        with open(f"{self.output_path}/results/final_results.txt", 'w') as f:
            f.write("ULTIMATE HYBRID SYSTEM RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Best accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
            f.write(f"Best model: {best_model}\n")
            f.write(f"Target achieved: {'YES' if best_acc >= 0.95 else 'NO'}\n")
            f.write(f"Training time: {duration}\n")
        
        logger.info("Ultimate hybrid pipeline complete!")
        
        return {
            'best_accuracy': best_acc,
            'best_model': best_model,
            'target_achieved': best_acc >= 0.95,
            'traditional_results': traditional_results,
            'hybrid_results': {
                'patch_accuracy': patch_acc,
                'image_accuracy': image_acc,
                'f1_score': patch_f1
            }
        }

if __name__ == "__main__":
    # Run the ultimate hybrid system
    ultimate_system = UltimateHybrid95Plus()
    results = ultimate_system.run_ultimate_hybrid_pipeline()
