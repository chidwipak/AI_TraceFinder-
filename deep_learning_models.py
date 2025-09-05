#!/usr/bin/env python3
"""
Deep Learning Models for AI_TraceFinder - Objective 1: Scanner Identification
Implements CNN with transfer learning for scanner identification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50V2, EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DeepLearningModels:
    def __init__(self, preprocessed_path="preprocessed", output_path="deep_learning_models"):
        self.preprocessed_path = preprocessed_path
        self.output_path = output_path
        self.random_seed = 42
        self.img_size = (256, 256)
        self.batch_size = 32
        self.epochs = 50
        self.learning_rate = 0.001
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.history = None
        self.model = None
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/trained_models",
            f"{self.output_path}/evaluation_results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/metadata",
            f"{self.output_path}/checkpoints"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_preprocessed_data(self):
        """Load preprocessed image data and metadata"""
        logger.info("Loading preprocessed data...")
        
        # Load metadata
        train_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_train.csv")
        test_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_test.csv")
        
        if not os.path.exists(train_metadata_path) or not os.path.exists(test_metadata_path):
            logger.error("Metadata files not found. Please run preprocessing first.")
            return None
        
        train_df = pd.read_csv(train_metadata_path)
        test_df = pd.read_csv(test_metadata_path)
        
        logger.info(f"Training samples: {len(train_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        return train_df, test_df
    
    def load_images_from_dataframe(self, df, split_name):
        """Load images from dataframe and prepare for training"""
        logger.info(f"Loading {split_name} images...")
        
        images = []
        labels = []
        filepaths = []
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                logger.info(f"Processing {split_name} image {idx}/{len(df)}")
            
            # Construct image path
            image_filename = os.path.basename(row['filepath']).replace('.tif', '.png')
            image_path = os.path.join(self.preprocessed_path, split_name, row['scanner_id'], 
                                    str(row['dpi']), image_filename)
            
            if os.path.exists(image_path):
                try:
                    # Load and preprocess image
                    img = Image.open(image_path).convert('RGB')
                    img = img.resize(self.img_size)
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    
                    images.append(img_array)
                    labels.append(row['scanner_id'])
                    filepaths.append(image_path)
                    
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} {split_name} images")
        return np.array(images), np.array(labels), filepaths
    
    def prepare_data(self):
        """Prepare training and test data"""
        logger.info("Preparing data for deep learning...")
        
        # Load metadata
        data = self.load_preprocessed_data()
        if data is None:
            return None
        
        train_df, test_df = data
        
        # Load images
        X_train, y_train_raw, train_paths = self.load_images_from_dataframe(train_df, "train")
        X_test, y_test_raw, test_paths = self.load_images_from_dataframe(test_df, "test")
        
        # Encode labels
        all_labels = np.concatenate([y_train_raw, y_test_raw])
        self.label_encoder.fit(all_labels)
        
        y_train = self.label_encoder.transform(y_train_raw)
        y_test = self.label_encoder.transform(y_test_raw)
        
        # Convert to categorical
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test, num_classes)
        
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Test data shape: {X_test.shape}")
        logger.info(f"Number of classes: {num_classes}")
        logger.info(f"Class names: {self.label_encoder.classes_}")
        
        return X_train, X_test, y_train_cat, y_test_cat, y_train, y_test, num_classes
    
    def create_data_augmentation(self):
        """Create data augmentation pipeline"""
        logger.info("Creating data augmentation pipeline...")
        
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.9, 1.1],
            horizontal_flip=True,
            vertical_flip=False,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        return datagen
    
    def build_resnet_model(self, num_classes):
        """Build ResNet50V2-based model with transfer learning"""
        logger.info("Building ResNet50V2 model...")
        
        # Load pre-trained ResNet50V2
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_efficientnet_model(self, num_classes):
        """Build EfficientNetB0-based model with transfer learning"""
        logger.info("Building EfficientNetB0 model...")
        
        # Load pre-trained EfficientNetB0
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.img_size[0], self.img_size[1], 3)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        # Create model
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_custom_cnn(self, num_classes):
        """Build custom CNN model"""
        logger.info("Building custom CNN model...")
        
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_callbacks(self, model_name):
        """Create training callbacks"""
        callbacks = [
            # Model checkpoint
            ModelCheckpoint(
                filepath=os.path.join(self.output_path, "checkpoints", f"{model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        return callbacks
    
    def train_model(self, model, X_train, y_train, X_test, y_test, model_name):
        """Train the model"""
        logger.info(f"Training {model_name}...")
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Create data augmentation
        datagen = self.create_data_augmentation()
        
        # Train model
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=self.batch_size),
            epochs=self.epochs,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        return history, model
    
    def evaluate_model(self, model, X_test, y_test, y_test_raw, model_name):
        """Evaluate the trained model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred_proba = model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        y_true = np.argmax(y_test, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report - handle case where not all classes are in test set
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_names = self.label_encoder.classes_[unique_classes]
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'classification_report': report,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_true': y_true
        }
        
        return results
    
    def plot_training_curves(self, history, model_name):
        """Plot training curves"""
        logger.info(f"Plotting training curves for {model_name}...")
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Accuracy plot
        axes[0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0].set_title(f'{model_name} - Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)
        
        # Loss plot
        axes[1].plot(history.history['loss'], label='Training Loss')
        axes[1].plot(history.history['val_loss'], label='Validation Loss')
        axes[1].set_title(f'{model_name} - Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "visualizations", f"{model_name.lower().replace(' ', '_')}_training_curves.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, model_name):
        """Plot confusion matrix"""
        logger.info(f"Plotting confusion matrix for {model_name}...")
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "visualizations", f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model, model_name):
        """Save the trained model"""
        logger.info(f"Saving {model_name}...")
        
        # Save model
        model_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}.h5")
        model.save(model_path)
        
        # Save label encoder
        encoder_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}_label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Label encoder saved to {encoder_path}")
    
    def generate_evaluation_report(self, results, model_name):
        """Generate evaluation report"""
        logger.info(f"Generating evaluation report for {model_name}...")
        
        print(f"\n{model_name} - Evaluation Results:")
        print("="*60)
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        
        # Save detailed report
        report_path = os.path.join(self.output_path, "metadata", f"{model_name.lower().replace(' ', '_')}_evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write(f"{model_name} - EVALUATION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"Precision: {results['precision']:.4f}\n")
            f.write(f"Recall: {results['recall']:.4f}\n")
            f.write(f"F1-Score: {results['f1_score']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            f.write("-" * 40 + "\n")
            for class_name in self.label_encoder.classes_:
                if class_name in results['classification_report']:
                    class_report = results['classification_report'][class_name]
                    f.write(f"{class_name}:\n")
                    f.write(f"  Precision: {class_report['precision']:.4f}\n")
                    f.write(f"  Recall: {class_report['recall']:.4f}\n")
                    f.write(f"  F1-Score: {class_report['f1-score']:.4f}\n")
                    f.write(f"  Support: {class_report['support']}\n\n")
        
        logger.info(f"Evaluation report saved to {report_path}")
    
    def run_full_pipeline(self, model_type='resnet'):
        """Run the complete deep learning pipeline"""
        logger.info("Starting deep learning pipeline...")
        
        # Prepare data
        data = self.prepare_data()
        if data is None:
            return
        
        X_train, X_test, y_train_cat, y_test_cat, y_train, y_test, num_classes = data
        
        # Build model based on type
        if model_type == 'resnet':
            model = self.build_resnet_model(num_classes)
            model_name = "ResNet50V2"
        elif model_type == 'efficientnet':
            model = self.build_efficientnet_model(num_classes)
            model_name = "EfficientNetB0"
        elif model_type == 'custom':
            model = self.build_custom_cnn(num_classes)
            model_name = "Custom CNN"
        else:
            logger.error(f"Unknown model type: {model_type}")
            return
        
        # Train model
        history, trained_model = self.train_model(model, X_train, y_train_cat, X_test, y_test_cat, model_name)
        
        # Evaluate model
        results = self.evaluate_model(trained_model, X_test, y_test_cat, y_test, model_name)
        
        # Generate visualizations and reports
        self.plot_training_curves(history, model_name)
        self.plot_confusion_matrix(results['confusion_matrix'], model_name)
        self.generate_evaluation_report(results, model_name)
        
        # Save model
        self.save_model(trained_model, model_name)
        
        logger.info("Deep learning pipeline complete!")

if __name__ == "__main__":
    # Train ResNet50V2 model
    dl_pipeline = DeepLearningModels()
    dl_pipeline.run_full_pipeline(model_type='resnet')
