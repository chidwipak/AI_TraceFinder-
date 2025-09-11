#!/usr/bin/env python3
"""
CPU-OPTIMIZED CNN + MLP CLASSIFIER FOR 97%+ ACCURACY
Uses existing preprocessed data and extracted features for maximum efficiency

Features:
- Uses existing preprocessed images and extracted features
- Advanced CNN architecture with attention mechanisms
- Multi-scale feature extraction
- Ensemble CNN + MLP
- CPU optimization for speed
- Advanced data augmentation
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Deep Learning imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, DenseNet121
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Traditional ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

# Feature extraction
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage.filters import gabor
from scipy import ndimage
from scipy.fft import fft2
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
import pywt

import pickle
import tifffile
import imageio.v2 as imageio

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cpu_optimized_cnn_mlp_97_plus.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class CPUOptimizedCNNMLP97Plus:
    def __init__(self, preprocessed_path="preprocessed", features_path="features", 
                 prnu_path="prnu_fingerprints", output_path="cpu_optimized_cnn_mlp_97_results"):
        self.preprocessed_path = preprocessed_path
        self.features_path = features_path
        self.prnu_path = prnu_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Model parameters
        self.img_size = (256, 256)
        self.batch_size = 16  # Optimized for CPU
        self.epochs = 150
        self.learning_rate = 0.0001
        self.num_classes = 11
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.cnn_model = None
        self.mlp_model = None
        self.ensemble_model = None
        
        # Load existing features and PRNU fingerprints
        self.load_existing_data()
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/models",
            f"{self.output_path}/checkpoints",
            f"{self.output_path}/results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/features",
            f"{self.output_path}/augmented_data"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_existing_data(self):
        """Load existing preprocessed data, features, and PRNU fingerprints"""
        logger.info("="*80)
        logger.info("🚀 LOADING EXISTING PREPROCESSED DATA AND FEATURES")
        logger.info("="*80)
        
        try:
            # Load existing features
            self.X_features = np.load(f"{self.features_path}/feature_vectors/X_features.npy")
            self.y_labels = np.load(f"{self.features_path}/feature_vectors/y_labels.npy", allow_pickle=True)
            self.feature_names = pd.read_csv(f"{self.features_path}/feature_vectors/feature_names.csv")
            
            logger.info(f"✅ Loaded {len(self.X_features)} samples with {self.X_features.shape[1]} features")
            logger.info(f"✅ Feature names: {len(self.feature_names)} features")
            
            # Load PRNU fingerprints
            prnu_file = f"{self.prnu_path}/fingerprints/all_prnu_fingerprints.pkl"
            if os.path.exists(prnu_file):
                with open(prnu_file, 'rb') as f:
                    self.prnu_fingerprints = pickle.load(f)
                logger.info(f"✅ Loaded PRNU fingerprints for {len(self.prnu_fingerprints)} scanners")
            else:
                logger.warning("⚠️ PRNU fingerprints not found")
                self.prnu_fingerprints = {}
            
            # Load metadata
            self.train_metadata = pd.read_csv(f"{self.preprocessed_path}/metadata/metadata_train.csv")
            self.test_metadata = pd.read_csv(f"{self.preprocessed_path}/metadata/metadata_test.csv")
            
            logger.info(f"✅ Loaded {len(self.train_metadata)} training samples")
            logger.info(f"✅ Loaded {len(self.test_metadata)} test samples")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading existing data: {e}")
            return False
    
    def advanced_data_augmentation(self):
        """Create advanced data augmentation pipeline"""
        logger.info("🔧 CREATING ADVANCED DATA AUGMENTATION PIPELINE")
        
        # Training augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.8, 1.2],
            contrast_range=[0.8, 1.2],
            saturation_range=[0.8, 1.2],
            zoom_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest',
            rescale=1./255
        )
        
        # Validation augmentation (minimal)
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        return train_datagen, val_datagen
    
    def create_attention_module(self, x, filters, name):
        """Create attention module for feature enhancement"""
        # Channel attention
        channel_attention = layers.GlobalAveragePooling2D()(x)
        channel_attention = layers.Dense(filters // 8, activation='relu')(channel_attention)
        channel_attention = layers.Dense(filters, activation='sigmoid')(channel_attention)
        channel_attention = layers.Reshape((1, 1, filters))(channel_attention)
        
        # Spatial attention
        spatial_attention = layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')(x)
        
        # Apply attention
        x = layers.Multiply()([x, channel_attention])
        x = layers.Multiply()([x, spatial_attention])
        
        return x
    
    def create_advanced_cnn(self, input_shape, num_classes):
        """Create advanced CNN with attention mechanisms"""
        logger.info("🏗️ CREATING ADVANCED CNN WITH ATTENTION")
        
        # Input layer
        inputs = layers.Input(shape=input_shape)
        
        # Initial convolution
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
        
        # Residual block 1
        residual = x
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
        # Attention module 1
        x = self.create_attention_module(x, 64, 'attention_1')
        
        # Residual block 2
        residual = layers.Conv2D(128, (1, 1), strides=(2, 2))(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
        # Attention module 2
        x = self.create_attention_module(x, 128, 'attention_2')
        
        # Residual block 3
        residual = layers.Conv2D(256, (1, 1), strides=(2, 2))(x)
        x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
        # Attention module 3
        x = self.create_attention_module(x, 256, 'attention_3')
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def create_transfer_learning_model(self, input_shape, num_classes):
        """Create transfer learning model with EfficientNet"""
        logger.info("🏗️ CREATING TRANSFER LEARNING MODEL")
        
        # Base model
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze initial layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        # Add custom head
        inputs = layers.Input(shape=input_shape)
        x = base_model(inputs, training=False)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Add attention
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def load_images_from_metadata(self, metadata_df, split_name):
        """Load images from metadata using existing preprocessed data"""
        logger.info(f"Loading {split_name} images from preprocessed data...")
        
        images = []
        labels = []
        
        for idx, row in metadata_df.iterrows():
            if idx % 100 == 0:
                logger.info(f"Processing {split_name} image {idx}/{len(metadata_df)}")
            
            # Construct image path
            image_filename = os.path.basename(row['filepath']).replace('.tif', '.png')
            image_path = os.path.join(self.preprocessed_path, split_name, row['scanner_id'], 
                                    str(row['dpi']), image_filename)
            
            if os.path.exists(image_path):
                try:
                    # Load image
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img)
                    
                    images.append(img_array)
                    labels.append(row['scanner_id'])
                    
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} {split_name} images")
        return np.array(images), np.array(labels)
    
    def create_mlp_model(self, input_dim, num_classes):
        """Create MLP model for feature-based classification"""
        logger.info("🏗️ CREATING MLP MODEL")
        
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(input_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        return model
    
    def create_ensemble_model(self, cnn_model, mlp_model, num_classes):
        """Create ensemble model combining CNN and MLP"""
        logger.info("🏗️ CREATING ENSEMBLE MODEL")
        
        # CNN input
        cnn_input = layers.Input(shape=(256, 256, 3), name='cnn_input')
        cnn_features = cnn_model(cnn_input)
        
        # MLP input
        mlp_input = layers.Input(shape=(self.X_features.shape[1],), name='mlp_input')
        mlp_features = mlp_model(mlp_input)
        
        # Combine features
        combined = layers.Concatenate()([cnn_features, mlp_features])
        
        # Final classification layers
        x = layers.Dense(256, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        # Create ensemble model
        ensemble_model = models.Model(
            inputs=[cnn_input, mlp_input],
            outputs=outputs
        )
        
        return ensemble_model
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val):
        """Train CNN model"""
        logger.info("🎯 TRAINING CNN MODEL")
        
        # Create model
        self.cnn_model = self.create_advanced_cnn((256, 256, 3), self.num_classes)
        
        # Compile model
        self.cnn_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/cnn_best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.cnn_model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def train_mlp_model(self, X_train, y_train, X_val, y_val):
        """Train MLP model"""
        logger.info("🎯 TRAINING MLP MODEL")
        
        # Create model
        self.mlp_model = self.create_mlp_model(X_train.shape[1], self.num_classes)
        
        # Compile model
        self.mlp_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/mlp_best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.mlp_model.fit(
            X_train, y_train,
            batch_size=32,
            epochs=200,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def train_ensemble_model(self, X_train, y_train, X_val, y_val, features_train, features_val):
        """Train ensemble model"""
        logger.info("🎯 TRAINING ENSEMBLE MODEL")
        
        # Create ensemble model
        self.ensemble_model = self.create_ensemble_model(
            self.cnn_model, self.mlp_model, self.num_classes
        )
        
        # Compile model
        self.ensemble_model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks_list = [
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/ensemble_best_model.h5",
                monitor='val_accuracy',
                save_best_only=True,
                mode='max'
            ),
            callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=30,
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=15,
                min_lr=1e-8
            )
        ]
        
        # Train model
        history = self.ensemble_model.fit(
            [X_train, features_train], y_train,
            batch_size=self.batch_size,
            epochs=100,
            validation_data=([X_val, features_val], y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        return history
    
    def evaluate_model(self, model, X_test, y_test, features_test=None, model_name="Model"):
        """Evaluate model performance"""
        logger.info(f"📊 EVALUATING {model_name}")
        
        if features_test is not None:
            # Ensemble model
            y_pred = model.predict([X_test, features_test])
        else:
            # Single model
            y_pred = model.predict(X_test)
        
        # Convert predictions
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_test_classes = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test_classes, y_pred_classes)
        precision = precision_score(y_test_classes, y_pred_classes, average='weighted')
        recall = recall_score(y_test_classes, y_pred_classes, average='weighted')
        f1 = f1_score(y_test_classes, y_pred_classes, average='weighted')
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test_classes, y_pred_classes)
        
        # Save results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred_classes,
            'true_labels': y_test_classes
        }
        
        return results
    
    def save_models(self):
        """Save trained models"""
        logger.info("💾 SAVING TRAINED MODELS")
        
        # Save CNN model
        if self.cnn_model:
            self.cnn_model.save(f"{self.output_path}/models/cnn_model.h5")
            logger.info("✅ CNN model saved")
        
        # Save MLP model
        if self.mlp_model:
            self.mlp_model.save(f"{self.output_path}/models/mlp_model.h5")
            logger.info("✅ MLP model saved")
        
        # Save ensemble model
        if self.ensemble_model:
            self.ensemble_model.save(f"{self.output_path}/models/ensemble_model.h5")
            logger.info("✅ Ensemble model saved")
        
        # Save label encoder
        with open(f"{self.output_path}/models/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info("✅ Label encoder saved")
        
        # Save scaler
        with open(f"{self.output_path}/models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("✅ Scaler saved")
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("🚀 STARTING CPU-OPTIMIZED CNN + MLP PIPELINE FOR 97%+ ACCURACY")
        
        try:
            # 1. Load existing data
            if not self.load_existing_data():
                return None
            
            # 2. Load images from metadata
            X_train, y_train_raw = self.load_images_from_metadata(self.train_metadata, 'train')
            X_test, y_test_raw = self.load_images_from_metadata(self.test_metadata, 'test')
            
            # 3. Encode labels
            all_labels = np.concatenate([y_train_raw, y_test_raw])
            self.label_encoder.fit(all_labels)
            
            y_train = to_categorical(self.label_encoder.transform(y_train_raw), self.num_classes)
            y_test = to_categorical(self.label_encoder.transform(y_test_raw), self.num_classes)
            
            # 4. Use existing features
            features_train = self.X_features[:len(X_train)]
            features_test = self.X_features[len(X_train):len(X_train)+len(X_test)]
            
            # 5. Scale features
            features_train_scaled = self.scaler.fit_transform(features_train)
            features_test_scaled = self.scaler.transform(features_test)
            
            # 6. Split training data for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_seed, stratify=y_train_raw
            )
            
            features_train_split, features_val_split, _, _ = train_test_split(
                features_train_scaled, y_train, test_size=0.2, random_state=self.random_seed, stratify=y_train_raw
            )
            
            logger.info(f"Training set: {X_train_split.shape}")
            logger.info(f"Validation set: {X_val_split.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Number of classes: {self.num_classes}")
            logger.info(f"Features shape: {features_train_scaled.shape}")
            
            # 7. Train CNN model
            cnn_history = self.train_cnn_model(X_train_split, y_train_split, X_val_split, y_val_split)
            
            # 8. Train MLP model
            mlp_history = self.train_mlp_model(features_train_split, y_train_split, features_val_split, y_val_split)
            
            # 9. Train ensemble model
            ensemble_history = self.train_ensemble_model(
                X_train_split, y_train_split, X_val_split, y_val_split,
                features_train_split, features_val_split
            )
            
            # 10. Evaluate models
            cnn_results = self.evaluate_model(self.cnn_model, X_test, y_test, model_name="CNN")
            mlp_results = self.evaluate_model(self.mlp_model, features_test_scaled, y_test, model_name="MLP")
            ensemble_results = self.evaluate_model(
                self.ensemble_model, X_test, y_test, features_test_scaled, model_name="Ensemble"
            )
            
            # 11. Save models
            self.save_models()
            
            # 12. Save results
            results = {
                'cnn_results': cnn_results,
                'mlp_results': mlp_results,
                'ensemble_results': ensemble_results,
                'label_encoder': self.label_encoder,
                'class_names': self.label_encoder.classes_
            }
            
            with open(f"{self.output_path}/results/final_results.pkl", 'wb') as f:
                pickle.dump(results, f)
            
            # 13. Print final results
            logger.info("="*80)
            logger.info("🏁 FINAL RESULTS")
            logger.info("="*80)
            logger.info(f"CNN Accuracy: {cnn_results['accuracy']:.4f} ({cnn_results['accuracy']*100:.2f}%)")
            logger.info(f"MLP Accuracy: {mlp_results['accuracy']:.4f} ({mlp_results['accuracy']*100:.2f}%)")
            logger.info(f"Ensemble Accuracy: {ensemble_results['accuracy']:.4f} ({ensemble_results['accuracy']*100:.2f}%)")
            
            if ensemble_results['accuracy'] >= 0.97:
                logger.info("🎉🎉🎉 SUCCESS! 97%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            else:
                gap = 0.97 - ensemble_results['accuracy']
                logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 97%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            return None

if __name__ == "__main__":
    logger.info("🚀 STARTING CPU-OPTIMIZED CNN + MLP PIPELINE...")
    
    try:
        # Create and run pipeline
        pipeline = CPUOptimizedCNNMLP97Plus()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\n🎯 CPU-optimized CNN + MLP pipeline completed!")
            print(f"Best accuracy: {results['ensemble_results']['accuracy']:.4f}")
            
            if results['ensemble_results']['accuracy'] >= 0.97:
                print("🎉🎉🎉 97%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            else:
                print("📈 Keep pushing towards 97%+ accuracy!")
        else:
            print("❌ Pipeline failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)
