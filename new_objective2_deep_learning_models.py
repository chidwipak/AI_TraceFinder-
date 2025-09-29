#!/usr/bin/env python3
"""
New Objective 2: Comprehensive Tampered/Original Detection - Deep Learning Models
Trains CNN models with attention mechanisms and mask-aware features for 90%+ accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Deep Learning libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3, ResNet50V2, VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import GlorotUniform

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Image processing
from PIL import Image
import cv2

# ML libraries for comparison
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ComprehensiveObjective2DeepLearning:
    def __init__(self, preprocessed_path="new_objective2_preprocessed", 
                 features_path="new_objective2_features", 
                 output_path="new_objective2_deep_learning_results"):
        self.preprocessed_path = preprocessed_path
        self.features_path = features_path
        self.output_path = output_path
        
        # Create output directories
        self.create_output_directories()
        
        # Model parameters
        self.img_size = (224, 224)
        self.batch_size = 16
        self.epochs = 100
        self.learning_rate = 0.0001
        self.patience = 15
        
        # Data containers
        self.train_df = None
        self.test_df = None
        self.X_train_img = None
        self.X_test_img = None
        self.y_train = None
        self.y_test = None
        self.X_train_features = None
        self.X_test_features = None
        
        # Models and results
        self.models = {}
        self.results = {}
        
        # Configure TensorFlow for CPU optimization
        self.configure_tensorflow()
    
    def configure_tensorflow(self):
        """Configure TensorFlow for optimal CPU performance"""
        # Set memory growth to avoid allocating all GPU memory at once
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
        
        # Configure CPU threads
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(4)
        
        logger.info("TensorFlow configured for optimal performance")
    
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path,
            f"{self.output_path}/models",
            f"{self.output_path}/results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/reports",
            f"{self.output_path}/checkpoints"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_and_prepare_data(self):
        """Load and prepare image data for deep learning"""
        logger.info("="*80)
        logger.info("LOADING AND PREPARING DEEP LEARNING DATA")
        logger.info("="*80)
        
        # Load train and test splits
        self.train_df = pd.read_csv(f"{self.preprocessed_path}/metadata/train_split.csv")
        self.test_df = pd.read_csv(f"{self.preprocessed_path}/metadata/test_split.csv")
        
        logger.info(f"Train samples: {len(self.train_df)}")
        logger.info(f"Test samples: {len(self.test_df)}")
        logger.info(f"Train class distribution: {self.train_df['label'].value_counts().to_dict()}")
        logger.info(f"Test class distribution: {self.test_df['label'].value_counts().to_dict()}")
        
        # Load images and prepare arrays
        self.X_train_img, self.y_train = self.load_image_data(self.train_df)
        self.X_test_img, self.y_test = self.load_image_data(self.test_df)
        
        # Load features for hybrid models
        self.load_feature_data()
        
        logger.info(f"Image data shapes - Train: {self.X_train_img.shape}, Test: {self.X_test_img.shape}")
        logger.info(f"Feature data shapes - Train: {self.X_train_features.shape}, Test: {self.X_test_features.shape}")
        logger.info("Data preparation completed!")
    
    def load_image_data(self, df):
        """Load and preprocess image data"""
        images = []
        labels = []
        
        for idx, row in df.iterrows():
            try:
                # Load image
                image_path = row['image_path']
                if not os.path.exists(image_path):
                    logger.warning(f"Image not found: {image_path}")
                    continue
                
                # Load and preprocess image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Resize image
                image = cv2.resize(image, self.img_size)
                
                # Normalize to [0, 1]
                image = image.astype(np.float32) / 255.0
                
                images.append(image)
                labels.append(row['label'])
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue
        
        return np.array(images), np.array(labels)
    
    def load_feature_data(self):
        """Load extracted features for hybrid models"""
        features_df = pd.read_csv(f"{self.features_path}/feature_vectors/comprehensive_features.csv")
        
        # Separate features and labels
        feature_columns = [col for col in features_df.columns 
                          if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]
        
        X_features = features_df[feature_columns].values
        y_features = features_df['label'].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_features = imputer.fit_transform(X_features)
        
        # Scale features
        scaler = StandardScaler()
        X_features = scaler.fit_transform(X_features)
        
        # Split into train and test
        train_indices = []
        test_indices = []
        
        for idx, row in features_df.iterrows():
            if row['image_path'] in self.train_df['image_path'].values:
                train_indices.append(idx)
            elif row['image_path'] in self.test_df['image_path'].values:
                test_indices.append(idx)
        
        self.X_train_features = X_features[train_indices]
        self.X_test_features = X_features[test_indices]
        
        # Save scaler for later use
        import joblib
        joblib.dump(scaler, f"{self.output_path}/models/feature_scaler.pkl")
    
    def create_data_augmentation_pipeline(self):
        """Create data augmentation pipeline"""
        train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        ])
        
        return train_transform
    
    def create_attention_module(self, input_tensor, name_prefix="attn"):
        """Create attention module for CNN"""
        # Global Average Pooling
        gap = layers.GlobalAveragePooling2D(name=f"{name_prefix}_gap")(input_tensor)
        
        # Dense layers for attention weights
        attention = layers.Dense(input_tensor.shape[-1] // 4, activation='relu', 
                               name=f"{name_prefix}_dense1")(gap)
        attention = layers.Dropout(0.2, name=f"{name_prefix}_dropout1")(attention)
        attention = layers.Dense(input_tensor.shape[-1], activation='sigmoid', 
                               name=f"{name_prefix}_dense2")(attention)
        
        # Reshape for multiplication
        attention = layers.Reshape((1, 1, input_tensor.shape[-1]), 
                                 name=f"{name_prefix}_reshape")(attention)
        
        # Apply attention
        attended = layers.Multiply(name=f"{name_prefix}_multiply")([input_tensor, attention])
        
        return attended
    
    def create_cnn_model(self, model_name="CNN_Attention"):
        """Create CNN model with attention mechanism"""
        inputs = layers.Input(shape=(*self.img_size, 3), name='input')
        
        # Initial convolution block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         kernel_regularizer=l2(0.001))(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 1
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 2 with attention
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        # Apply attention
        x = self.create_attention_module(x, "block2")
        
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Block 3 with attention
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        
        # Apply attention
        x = self.create_attention_module(x, "block3")
        
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
        
        return model
    
    def create_transfer_learning_model(self, base_model_name="EfficientNetB3"):
        """Create transfer learning model"""
        # Load pre-trained model
        if base_model_name == "EfficientNetB3":
            base_model = EfficientNetB3(weights='imagenet', include_top=False, 
                                      input_shape=(*self.img_size, 3))
        elif base_model_name == "ResNet50V2":
            base_model = ResNet50V2(weights='imagenet', include_top=False, 
                                   input_shape=(*self.img_size, 3))
        elif base_model_name == "VGG16":
            base_model = VGG16(weights='imagenet', include_top=False, 
                              input_shape=(*self.img_size, 3))
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Add custom head
        inputs = layers.Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        
        # Global pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs, name=f"Transfer_{base_model_name}")
        
        return model
    
    def create_hybrid_model(self, model_name="Hybrid_CNN_MLP"):
        """Create hybrid model combining CNN and MLP"""
        # CNN branch
        img_inputs = layers.Input(shape=(*self.img_size, 3), name='image_input')
        
        # CNN feature extraction
        cnn_branch = self.create_cnn_model("CNN_Branch")
        cnn_features = cnn_branch.layers[-4].output  # Get features before final dense layers
        
        # MLP branch for traditional features
        feature_inputs = layers.Input(shape=(self.X_train_features.shape[1],), name='feature_input')
        mlp_branch = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(feature_inputs)
        mlp_branch = layers.BatchNormalization()(mlp_branch)
        mlp_branch = layers.Dropout(0.3)(mlp_branch)
        
        mlp_branch = layers.Dense(128, activation='relu', kernel_regularizer=l2(0.001))(mlp_branch)
        mlp_branch = layers.BatchNormalization()(mlp_branch)
        mlp_branch = layers.Dropout(0.2)(mlp_branch)
        
        # Combine branches
        combined = layers.concatenate([cnn_features, mlp_branch])
        
        # Final layers
        combined = layers.Dense(512, activation='relu', kernel_regularizer=l2(0.001))(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.5)(combined)
        
        combined = layers.Dense(256, activation='relu', kernel_regularizer=l2(0.001))(combined)
        combined = layers.BatchNormalization()(combined)
        combined = layers.Dropout(0.3)(combined)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(combined)
        
        model = models.Model(inputs=[img_inputs, feature_inputs], outputs=outputs, name=model_name)
        
        return model
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_f1',
                mode='max',
                patience=self.patience,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{self.output_path}/checkpoints/{model_name}_best.h5",
                monitor='val_f1',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train_model(self, model, model_name, X_train, y_train, X_val=None, y_val=None):
        """Train a single model"""
        logger.info(f"Training {model_name}...")
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', self.f1_metric]
        )
        
        # Create callbacks
        callbacks_list = self.create_callbacks(model_name)
        
        # Prepare validation data
        if X_val is None:
            X_val, X_train_split, y_val, y_train_split = train_test_split(
                X_train, y_train, test_size=0.8, random_state=42, stratify=y_train
            )
        else:
            X_train_split, y_train_split = X_train, y_train
        
        # Train model
        history = model.fit(
            X_train_split, y_train_split,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
        
        # Save model
        model.save(f"{self.output_path}/models/{model_name}.h5")
        
        return model, history
    
    def f1_metric(self, y_true, y_pred):
        """Custom F1 metric for TensorFlow"""
        def recall(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
            return recall

        def precision(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
            predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
            return precision

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        precision_val = precision(y_true, y_pred)
        recall_val = recall(y_true, y_pred)
        return 2 * ((precision_val * recall_val) / (precision_val + recall_val + tf.keras.backend.epsilon()))
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Evaluate a trained model"""
        # Make predictions
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba.flatten()
        }
    
    def train_all_models(self):
        """Train all deep learning models"""
        logger.info("="*80)
        logger.info("TRAINING ALL DEEP LEARNING MODELS")
        logger.info("="*80)
        
        # Split training data for validation
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            self.X_train_img, self.y_train, test_size=0.2, random_state=42, stratify=self.y_train
        )
        
        # 1. CNN with Attention
        cnn_model = self.create_cnn_model("CNN_Attention")
        cnn_model, cnn_history = self.train_model(
            cnn_model, "CNN_Attention", 
            X_train_split, y_train_split, X_val_split, y_val_split
        )
        self.results['CNN_Attention'] = self.evaluate_model(cnn_model, "CNN_Attention", self.X_test_img, self.y_test)
        
        # 2. Transfer Learning Models
        transfer_models = ["EfficientNetB3", "ResNet50V2", "VGG16"]
        for base_model in transfer_models:
            try:
                tl_model = self.create_transfer_learning_model(base_model)
                tl_model, tl_history = self.train_model(
                    tl_model, f"Transfer_{base_model}",
                    X_train_split, y_train_split, X_val_split, y_val_split
                )
                self.results[f"Transfer_{base_model}"] = self.evaluate_model(
                    tl_model, f"Transfer_{base_model}", self.X_test_img, self.y_test
                )
            except Exception as e:
                logger.error(f"Failed to train {base_model}: {e}")
                continue
        
        # 3. Hybrid CNN-MLP Model
        try:
            hybrid_model = self.create_hybrid_model("Hybrid_CNN_MLP")
            
            # Prepare hybrid data
            X_train_hybrid = [X_train_split, self.X_train_features[:len(X_train_split)]]
            X_val_hybrid = [X_val_split, self.X_train_features[len(X_train_split):len(X_train_split)+len(X_val_split)]]
            X_test_hybrid = [self.X_test_img, self.X_test_features]
            
            hybrid_model, hybrid_history = self.train_model(
                hybrid_model, "Hybrid_CNN_MLP",
                X_train_hybrid, y_train_split, X_val_hybrid, y_val_split
            )
            self.results['Hybrid_CNN_MLP'] = self.evaluate_model(
                hybrid_model, "Hybrid_CNN_MLP", X_test_hybrid, self.y_test
            )
        except Exception as e:
            logger.error(f"Failed to train Hybrid model: {e}")
        
        logger.info("All deep learning models trained!")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("="*80)
        logger.info("GENERATING DEEP LEARNING EVALUATION REPORT")
        logger.info("="*80)
        
        # Collect all results
        all_results = []
        for model_name, result in self.results.items():
            all_results.append({
                'model_name': model_name,
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'auc': result['auc']
            })
        
        # Convert to DataFrame and sort by F1 score
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Save results
        results_df.to_csv(f"{self.output_path}/results/deep_learning_results.csv", index=False)
        
        # Generate visualizations
        self._plot_results_comparison(results_df)
        self._plot_model_performance(results_df)
        
        # Generate text report
        self._generate_text_report(results_df)
        
        logger.info("Deep learning evaluation report generated!")
    
    def _plot_results_comparison(self, results_df):
        """Plot results comparison"""
        plt.figure(figsize=(15, 10))
        
        # Performance comparison
        plt.subplot(2, 2, 1)
        plt.barh(range(len(results_df)), results_df['f1_score'])
        plt.yticks(range(len(results_df)), results_df['model_name'])
        plt.xlabel('F1 Score')
        plt.title('Deep Learning Models - F1 Score Comparison')
        
        plt.subplot(2, 2, 2)
        plt.barh(range(len(results_df)), results_df['accuracy'])
        plt.yticks(range(len(results_df)), results_df['model_name'])
        plt.xlabel('Accuracy')
        plt.title('Deep Learning Models - Accuracy Comparison')
        
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['accuracy'], results_df['f1_score'], alpha=0.7, s=100)
        for i, row in results_df.iterrows():
            plt.annotate(row['model_name'], (row['accuracy'], row['f1_score']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        plt.xlabel('Accuracy')
        plt.ylabel('F1 Score')
        plt.title('Accuracy vs F1 Score')
        
        plt.subplot(2, 2, 4)
        plt.barh(range(len(results_df)), results_df['auc'])
        plt.yticks(range(len(results_df)), results_df['model_name'])
        plt.xlabel('AUC')
        plt.title('Deep Learning Models - AUC Comparison')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/deep_learning_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self, results_df):
        """Plot model performance metrics"""
        plt.figure(figsize=(15, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            plt.barh(range(len(results_df)), results_df[metric])
            plt.yticks(range(len(results_df)), results_df['model_name'])
            plt.xlabel(metric.replace('_', ' ').title())
            plt.title(f'Deep Learning Models - {metric.replace("_", " ").title()}')
        
        # Confusion matrix for best model
        plt.subplot(2, 3, 6)
        best_result = results_df.iloc[0]
        best_model_info = self.results[best_result['model_name']]
        
        cm = confusion_matrix(self.y_test, best_model_info['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_result["model_name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/model_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results_df):
        """Generate comprehensive text report"""
        report_path = f"{self.output_path}/reports/deep_learning_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE OBJECTIVE 2 DEEP LEARNING MODELS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Training samples: {len(self.X_train_img)}\n")
            f.write(f"Test samples: {len(self.X_test_img)}\n")
            f.write(f"Image size: {self.img_size}\n")
            f.write(f"Training class distribution: {np.bincount(self.y_train)}\n")
            f.write(f"Test class distribution: {np.bincount(self.y_test)}\n\n")
            
            # Model Results
            f.write("DEEP LEARNING MODEL RESULTS\n")
            f.write("-"*40 + "\n")
            for i, (_, row) in enumerate(results_df.iterrows()):
                f.write(f"{i+1:2d}. {row['model_name']}\n")
                f.write(f"    Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"    Precision: {row['precision']:.4f}\n")
                f.write(f"    Recall: {row['recall']:.4f}\n")
                f.write(f"    F1 Score: {row['f1_score']:.4f}\n")
                f.write(f"    AUC: {row['auc']:.4f}\n\n")
            
            # Best Model Analysis
            best_result = results_df.iloc[0]
            f.write("BEST MODEL ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Best Model: {best_result['model_name']}\n")
            f.write(f"Best F1 Score: {best_result['f1_score']:.4f}\n")
            f.write(f"Best Accuracy: {best_result['accuracy']:.4f}\n")
            f.write(f"Best AUC: {best_result['auc']:.4f}\n\n")
            
            # Target Analysis
            target_accuracy = 0.90
            models_above_target = results_df[results_df['accuracy'] >= target_accuracy]
            
            f.write(f"MODELS ACHIEVING {target_accuracy*100}%+ ACCURACY\n")
            f.write("-"*40 + "\n")
            if len(models_above_target) > 0:
                f.write(f"Found {len(models_above_target)} models achieving {target_accuracy*100}%+ accuracy:\n")
                for _, row in models_above_target.iterrows():
                    f.write(f"  {row['model_name']}: {row['accuracy']:.4f}\n")
            else:
                f.write(f"No models achieved {target_accuracy*100}%+ accuracy\n")
                f.write(f"Best accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)\n")
                f.write(f"Gap to target: {(target_accuracy - best_result['accuracy'])*100:.2f} percentage points\n")
            
            f.write("\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS FOR 90%+ ACCURACY\n")
            f.write("-"*40 + "\n")
            if len(models_above_target) == 0:
                f.write("To achieve 90%+ accuracy, consider:\n")
                f.write("1. Advanced ensemble methods combining best models\n")
                f.write("2. More sophisticated data augmentation\n")
                f.write("3. Fine-tuning pre-trained models\n")
                f.write("4. Advanced regularization techniques\n")
                f.write("5. Custom loss functions\n")
                f.write("6. Multi-scale feature extraction\n")
                f.write("7. Attention mechanisms with multiple heads\n")
                f.write("8. Cross-validation with more sophisticated strategies\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_deep_learning_pipeline(self):
        """Run complete deep learning pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE OBJECTIVE 2 DEEP LEARNING PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Train all models
        self.train_all_models()
        
        # Step 3: Generate evaluation report
        self.generate_evaluation_report()
        
        logger.info("✅ COMPREHENSIVE DEEP LEARNING PIPELINE COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        
        return self.results

def main():
    """Main function to run comprehensive deep learning pipeline"""
    deep_learning = ComprehensiveObjective2DeepLearning()
    results = deep_learning.run_complete_deep_learning_pipeline()
    
    print("\n" + "="*80)
    print("DEEP LEARNING PIPELINE SUMMARY")
    print("="*80)
    
    # Find best result
    if results:
        all_results = [(name, result['f1_score'], result['accuracy']) for name, result in results.items()]
        best_result = max(all_results, key=lambda x: x[1])  # Sort by F1 score
        
        print(f"Best Model: {best_result[0]}")
        print(f"Best F1 Score: {best_result[1]:.4f}")
        print(f"Best Accuracy: {best_result[2]:.4f}")
        
        target_accuracy = 0.90
        if best_result[2] >= target_accuracy:
            print(f"🎉 TARGET ACHIEVED: {target_accuracy*100}%+ accuracy reached!")
        else:
            print(f"📈 Progress: {best_result[2]*100:.2f}% accuracy (need {target_accuracy*100:.1f}%)")
            print(f"Gap to target: {(target_accuracy - best_result[2])*100:.2f} percentage points")
    
    print("\nNext step: Create ensemble models combining best ML and DL models!")

if __name__ == "__main__":
    main()
