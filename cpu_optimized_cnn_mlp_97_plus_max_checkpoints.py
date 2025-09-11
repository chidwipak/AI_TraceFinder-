#!/usr/bin/env python3
"""
CPU-OPTIMIZED CNN + MLP CLASSIFIER WITH MAXIMUM CHECKPOINTS
Enhanced version with comprehensive checkpointing and error handling
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
# import cv2

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

import pickle
import tifffile
import imageio.v2 as imageio

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cpu_optimized_cnn_mlp_97_plus_max_checkpoints.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

class MaxCheckpointsCNNMLP97Plus:
    def __init__(self, preprocessed_path="preprocessed", features_path="features", 
                 prnu_path="prnu_fingerprints", output_path="cpu_optimized_cnn_mlp_97_results_max_checkpoints"):
        self.preprocessed_path = preprocessed_path
        self.features_path = features_path
        self.prnu_path = prnu_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Model parameters
        self.img_size = (256, 256)
        self.batch_size = 16
        self.epochs = 200
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
            f"{self.output_path}/checkpoints/cnn",
            f"{self.output_path}/checkpoints/mlp", 
            f"{self.output_path}/checkpoints/ensemble",
            f"{self.output_path}/results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/features",
            f"{self.output_path}/augmented_data",
            f"{self.output_path}/logs"
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
    
    def load_images_in_batches(self, metadata_df, split_name, batch_size=200):
        """Load images in batches to avoid memory issues"""
        logger.info(f"Loading {split_name} images in batches of {batch_size}...")
        
        all_images = []
        all_labels = []
        failed_count = 0
        
        # Process in batches
        for batch_start in range(0, len(metadata_df), batch_size):
            batch_end = min(batch_start + batch_size, len(metadata_df))
            batch_metadata = metadata_df.iloc[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}/{(len(metadata_df)-1)//batch_size + 1} ({batch_start}-{batch_end})")
            
            batch_images = []
            batch_labels = []
            
            for idx, row in batch_metadata.iterrows():
                # Construct image path
                image_filename = os.path.basename(row['filepath']).replace('.tif', '.png')
                image_path = os.path.join(self.preprocessed_path, split_name, row['scanner_id'], 
                                        str(row['dpi']), image_filename)
                
                if os.path.exists(image_path):
                    try:
                        # Load image
                        img = Image.open(image_path).convert('RGB')
                        img = img.resize(self.img_size)
                        img_array = np.array(img, dtype=np.float32)
                        
                        # Normalize to [0, 1] range
                        img_array = img_array / 255.0
                        
                        batch_images.append(img_array)
                        batch_labels.append(row['scanner_id'])
                        
                    except Exception as e:
                        logger.warning(f"Failed to load {image_path}: {e}")
                        failed_count += 1
                        continue
                else:
                    failed_count += 1
            
            # Add batch to main lists
            if batch_images:
                all_images.extend(batch_images)
                all_labels.extend(batch_labels)
            
            # Clear batch from memory
            del batch_images, batch_labels
        
        logger.info(f"Loaded {len(all_images)} {split_name} images (failed: {failed_count})")
        
        if len(all_images) == 0:
            logger.error(f"No images loaded for {split_name}!")
            return None, None
            
        return np.array(all_images), np.array(all_labels)
    
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
        
        # Residual block 2
        residual = layers.Conv2D(128, (1, 1), strides=(2, 2))(x)
        x = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
        # Residual block 3
        residual = layers.Conv2D(256, (1, 1), strides=(2, 2))(x)
        x = layers.Conv2D(256, (3, 3), strides=(2, 2), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
        
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
    
    def create_comprehensive_callbacks(self, model_name, monitor='val_accuracy'):
        """Create comprehensive callbacks with maximum checkpointing"""
        logger.info(f"🔧 CREATING COMPREHENSIVE CALLBACKS FOR {model_name}")
        
        callbacks_list = [
            # Best model checkpoint
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/{model_name.lower()}/{model_name.lower()}_best_model.h5",
                monitor=monitor,
                save_best_only=True,
                mode='max',
                verbose=1,
                save_weights_only=False
            ),
            
            # Every epoch checkpoint
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/{model_name.lower()}/{model_name.lower()}_epoch_{{epoch:03d}}_val_acc_{{val_accuracy:.4f}}.h5",
                monitor=monitor,
                save_best_only=False,
                mode='max',
                verbose=1,
                save_weights_only=False,
                period=5  # Save every 5 epochs
            ),
            
            # Weights only checkpoint (smaller files)
            callbacks.ModelCheckpoint(
                f"{self.output_path}/checkpoints/{model_name.lower()}/{model_name.lower()}_weights_epoch_{{epoch:03d}}.h5",
                monitor=monitor,
                save_best_only=False,
                mode='max',
                verbose=1,
                save_weights_only=True,
                period=10  # Save every 10 epochs
            ),
            
            # Early stopping
            callbacks.EarlyStopping(
                monitor=monitor,
                patience=30,
                restore_best_weights=True,
                verbose=1,
                mode='max'
            ),
            
            # Learning rate reduction
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1,
                mode='min'
            ),
            
            # CSV logger
            callbacks.CSVLogger(
                f"{self.output_path}/logs/{model_name.lower()}_training_log.csv",
                append=True
            ),
            
            # TensorBoard (if available)
            callbacks.TensorBoard(
                log_dir=f"{self.output_path}/logs/{model_name.lower()}_tensorboard",
                histogram_freq=1,
                write_graph=True,
                write_images=True
            )
        ]
        
        return callbacks_list
    
    def train_cnn_model(self, X_train, y_train, X_val, y_val):
        """Train CNN model with comprehensive checkpointing"""
        logger.info("🎯 TRAINING CNN MODEL WITH MAXIMUM CHECKPOINTS")
        
        try:
            # Create model
            self.cnn_model = self.create_advanced_cnn((256, 256, 3), self.num_classes)
            
            # Compile model
            self.cnn_model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Print model summary
            logger.info("CNN Model Summary:")
            self.cnn_model.summary()
            
            # Create comprehensive callbacks
            callbacks_list = self.create_comprehensive_callbacks("CNN")
            
            # Train model
            logger.info("Starting CNN training...")
            history = self.cnn_model.fit(
                X_train, y_train,
                batch_size=self.batch_size,
                epochs=self.epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            logger.info("✅ CNN training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"❌ CNN training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def train_mlp_model(self, X_train, y_train, X_val, y_val):
        """Train MLP model with comprehensive checkpointing"""
        logger.info("🎯 TRAINING MLP MODEL WITH MAXIMUM CHECKPOINTS")
        
        try:
            # Create model
            self.mlp_model = self.create_mlp_model(X_train.shape[1], self.num_classes)
            
            # Compile model
            self.mlp_model.compile(
                optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Print model summary
            logger.info("MLP Model Summary:")
            self.mlp_model.summary()
            
            # Create comprehensive callbacks
            callbacks_list = self.create_comprehensive_callbacks("MLP")
            
            # Train model
            logger.info("Starting MLP training...")
            history = self.mlp_model.fit(
                X_train, y_train,
                batch_size=32,
                epochs=200,
                validation_data=(X_val, y_val),
                callbacks=callbacks_list,
                verbose=1
            )
            
            logger.info("✅ MLP training completed successfully")
            return history
            
        except Exception as e:
            logger.error(f"❌ MLP training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name="Model"):
        """Evaluate model performance"""
        logger.info(f"📊 EVALUATING {model_name}")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test, verbose=1)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_test_classes = np.argmax(y_test, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test_classes, y_pred_classes)
            precision = precision_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
            recall = recall_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
            f1 = f1_score(y_test_classes, y_pred_classes, average='weighted', zero_division=0)
            
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
            
        except Exception as e:
            logger.error(f"❌ {model_name} evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
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
        
        # Save label encoder
        with open(f"{self.output_path}/models/label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        logger.info("✅ Label encoder saved")
        
        # Save scaler
        with open(f"{self.output_path}/models/scaler.pkl", 'wb') as f:
            pickle.dump(self.scaler, f)
        logger.info("✅ Scaler saved")
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline with maximum checkpointing"""
        logger.info("🚀 STARTING CPU-OPTIMIZED CNN + MLP PIPELINE WITH MAXIMUM CHECKPOINTS")
        
        try:
            # 1. Load existing data
            if not self.load_existing_data():
                return None
            
            # 2. Load images in batches
            logger.info("Loading training images in batches...")
            X_train, y_train_raw = self.load_images_in_batches(self.train_metadata, 'train', batch_size=200)
            if X_train is None:
                logger.error("Failed to load training images")
                return None
            
            logger.info("Loading test images in batches...")
            X_test, y_test_raw = self.load_images_in_batches(self.test_metadata, 'test', batch_size=200)
            if X_test is None:
                logger.error("Failed to load test images")
                return None
            
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
            
            # 9. Evaluate models
            cnn_results = self.evaluate_model(self.cnn_model, X_test, y_test, model_name="CNN")
            mlp_results = self.evaluate_model(self.mlp_model, features_test_scaled, y_test, model_name="MLP")
            
            # 10. Save models
            self.save_models()
            
            # 11. Save results
            results = {
                'cnn_results': cnn_results,
                'mlp_results': mlp_results,
                'label_encoder': self.label_encoder,
                'class_names': self.label_encoder.classes_
            }
            
            with open(f"{self.output_path}/results/final_results.pkl", 'wb') as f:
                pickle.dump(results, f)
            
            # 12. Print final results
            logger.info("="*80)
            logger.info("🏁 FINAL RESULTS")
            logger.info("="*80)
            
            if cnn_results:
                logger.info(f"CNN Accuracy: {cnn_results['accuracy']:.4f} ({cnn_results['accuracy']*100:.2f}%)")
            else:
                logger.error("CNN evaluation failed")
            
            if mlp_results:
                logger.info(f"MLP Accuracy: {mlp_results['accuracy']:.4f} ({mlp_results['accuracy']*100:.2f}%)")
            else:
                logger.error("MLP evaluation failed")
            
            if cnn_results and cnn_results['accuracy'] >= 0.97:
                logger.info("🎉🎉🎉 SUCCESS! 97%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            elif cnn_results:
                gap = 0.97 - cnn_results['accuracy']
                logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 97%")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    logger.info("🚀 STARTING CPU-OPTIMIZED CNN + MLP PIPELINE WITH MAXIMUM CHECKPOINTS...")
    
    try:
        # Create and run pipeline
        pipeline = MaxCheckpointsCNNMLP97Plus()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\n🎯 CPU-optimized CNN + MLP pipeline with maximum checkpoints completed!")
            if results['cnn_results']:
                print(f"CNN accuracy: {results['cnn_results']['accuracy']:.4f}")
            if results['mlp_results']:
                print(f"MLP accuracy: {results['mlp_results']['accuracy']:.4f}")
        else:
            print("❌ Pipeline failed!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
