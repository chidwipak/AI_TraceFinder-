#!/usr/bin/env python3
"""
Advanced Scanner Identification for AI_TraceFinder - Objective 1: Scanner Identification
Combines PRNU fingerprints, CNN-based fingerprinting, and multi-modal feature fusion
Based on research: "Beyond PRNU: Learning Robust Device-Specific Fingerprint" and "Noiseprint: CNN-based fingerprint"
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
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
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

class AdvancedScannerIdentification:
    def __init__(self, preprocessed_path="preprocessed", features_path="features", 
                 prnu_path="prnu_fingerprints", output_path="advanced_models"):
        self.preprocessed_path = preprocessed_path
        self.features_path = features_path
        self.prnu_path = prnu_path
        self.output_path = output_path
        self.random_seed = 42
        self.img_size = (256, 256)
        self.batch_size = 32
        self.epochs = 100
        self.learning_rate = 0.001
        
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
            f"{self.output_path}/checkpoints"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_prnu_fingerprints(self):
        """Load PRNU fingerprints for correlation analysis"""
        logger.info("Loading PRNU fingerprints...")
        
        pickle_filepath = os.path.join(self.prnu_path, "fingerprints", "all_prnu_fingerprints.pkl")
        
        try:
            with open(pickle_filepath, 'rb') as f:
                self.prnu_fingerprints = pickle.load(f)
            logger.info(f"Loaded PRNU fingerprints for {len(self.prnu_fingerprints)} scanners")
        except Exception as e:
            logger.error(f"Failed to load PRNU fingerprints: {e}")
            self.prnu_fingerprints = {}
    
    def load_feature_data(self):
        """Load extracted features data"""
        logger.info("Loading feature data...")
        
        csv_path = os.path.join(self.features_path, "feature_vectors", "scanner_features.csv")
        if not os.path.exists(csv_path):
            logger.error(f"Feature CSV not found: {csv_path}")
            return None
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded feature data: {df.shape}")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train and test
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X[train_mask]
        y_train = y_encoded[train_mask]
        X_test = X[test_mask]
        y_test = y_encoded[test_mask]
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def load_image_data(self):
        """Load image data for CNN-based fingerprinting"""
        logger.info("Loading image data...")
        
        # Load metadata
        train_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_train.csv")
        test_metadata_path = os.path.join(self.preprocessed_path, "metadata", "metadata_test.csv")
        
        if not os.path.exists(train_metadata_path) or not os.path.exists(test_metadata_path):
            logger.error("Metadata files not found. Please run preprocessing first.")
            return None
        
        train_df = pd.read_csv(train_metadata_path)
        test_df = pd.read_csv(test_metadata_path)
        
        # Load images
        X_train_img, y_train_img = self.load_images_from_dataframe(train_df, "train")
        X_test_img, y_test_img = self.load_images_from_dataframe(test_df, "test")
        
        return X_train_img, X_test_img, y_train_img, y_test_img
    
    def load_images_from_dataframe(self, df, split_name):
        """Load images from dataframe"""
        logger.info(f"Loading {split_name} images...")
        
        images = []
        labels = []
        
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
                    
                except Exception as e:
                    logger.warning(f"Failed to load {image_path}: {e}")
                    continue
        
        logger.info(f"Loaded {len(images)} {split_name} images")
        return np.array(images), np.array(labels)
    
    def build_noiseprint_cnn(self, num_classes):
        """Build CNN-based fingerprinting model (Noiseprint approach)"""
        logger.info("Building Noiseprint CNN model...")
        
        # Input for image
        img_input = layers.Input(shape=(self.img_size[0], self.img_size[1], 3), name='image_input')
        
        # ResNet50V2 base for feature extraction
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_tensor=img_input
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Feature extraction layers
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # Output layer
        img_output = layers.Dense(num_classes, activation='softmax', name='image_output')(x)
        
        # Create model
        model = models.Model(inputs=img_input, outputs=img_output)
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def build_hybrid_model(self, num_classes, feature_dim):
        """Build hybrid model combining CNN and feature-based approaches"""
        logger.info("Building hybrid model...")
        
        # Input for image
        img_input = layers.Input(shape=(self.img_size[0], self.img_size[1], 3), name='image_input')
        
        # Input for extracted features
        feature_input = layers.Input(shape=(feature_dim,), name='feature_input')
        
        # CNN branch (Noiseprint approach)
        base_model = ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_tensor=img_input
        )
        base_model.trainable = False
        
        cnn_branch = base_model.output
        cnn_branch = layers.GlobalAveragePooling2D()(cnn_branch)
        cnn_branch = layers.Dropout(0.5)(cnn_branch)
        cnn_branch = layers.Dense(512, activation='relu')(cnn_branch)
        cnn_branch = layers.Dropout(0.3)(cnn_branch)
        cnn_branch = layers.Dense(256, activation='relu')(cnn_branch)
        
        # Feature branch
        feature_branch = layers.Dense(512, activation='relu')(feature_input)
        feature_branch = layers.Dropout(0.3)(feature_branch)
        feature_branch = layers.Dense(256, activation='relu')(feature_branch)
        feature_branch = layers.Dropout(0.2)(feature_branch)
        
        # Concatenate both branches
        combined = layers.Concatenate()([cnn_branch, feature_branch])
        combined = layers.Dropout(0.4)(combined)
        combined = layers.Dense(512, activation='relu')(combined)
        combined = layers.Dropout(0.3)(combined)
        combined = layers.Dense(256, activation='relu')(combined)
        combined = layers.Dropout(0.2)(combined)
        
        # Output layer
        output = layers.Dense(num_classes, activation='softmax', name='output')(combined)
        
        # Create model
        model = models.Model(inputs=[img_input, feature_input], outputs=output)
        
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
            ModelCheckpoint(
                filepath=os.path.join(self.output_path, "checkpoints", f"{model_name}_best.h5"),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
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
        return callbacks
    
    def train_hybrid_model(self, model, X_train_img, X_train_feat, y_train_cat, 
                          X_test_img, X_test_feat, y_test_cat, model_name):
        """Train the hybrid model"""
        logger.info(f"Training {model_name}...")
        
        # Create callbacks
        callbacks = self.create_callbacks(model_name)
        
        # Train model without data augmentation for now
        history = model.fit(
            [X_train_img, X_train_feat],
            y_train_cat,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=([X_test_img, X_test_feat], y_test_cat),
            callbacks=callbacks,
            verbose=1
        )
        
        return history, model
    
    def train_ensemble_model(self, X_train, y_train, X_test, y_test):
        """Train ensemble model using extracted features"""
        logger.info("Training ensemble model...")
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        rf_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = rf_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Ensemble model accuracy: {accuracy:.4f}")
        
        return rf_model, accuracy
    
    def evaluate_model(self, model, X_test_img, X_test_feat, y_test_cat, y_test, model_name):
        """Evaluate the trained model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        if isinstance(model, RandomForestClassifier):
            X_test_scaled = self.scaler.transform(X_test_feat)
            y_pred_proba = model.predict_proba(X_test_scaled)
            y_pred = model.predict(X_test_scaled)
        else:
            y_pred_proba = model.predict([X_test_img, X_test_feat])
            y_pred = np.argmax(y_pred_proba, axis=1)
        
        y_true = np.argmax(y_test_cat, axis=1) if len(y_test_cat.shape) > 1 else y_test
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
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
        
        if isinstance(model, RandomForestClassifier):
            # Save Random Forest model
            model_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        else:
            # Save Keras model
            model_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}.h5")
            model.save(model_path)
        
        # Save label encoder and scaler
        encoder_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}_label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        if hasattr(self, 'scaler'):
            scaler_path = os.path.join(self.output_path, "trained_models", f"{model_name.lower().replace(' ', '_')}_scaler.pkl")
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
        
        logger.info(f"Model saved to {model_path}")
    
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
    
    def run_full_pipeline(self):
        """Run the complete advanced scanner identification pipeline"""
        logger.info("Starting advanced scanner identification pipeline...")
        
        # Load PRNU fingerprints
        self.load_prnu_fingerprints()
        
        # Load feature data
        feature_data = self.load_feature_data()
        if feature_data is None:
            return
        
        X_train_feat, X_test_feat, y_train_feat, y_test_feat, feature_columns = feature_data
        
        # Load image data
        image_data = self.load_image_data()
        if image_data is None:
            return
        
        X_train_img, X_test_img, y_train_img, y_test_img = image_data
        
        # Prepare labels for hybrid model
        num_classes = len(self.label_encoder.classes_)
        y_train_cat = tf.keras.utils.to_categorical(y_train_feat, num_classes)
        y_test_cat = tf.keras.utils.to_categorical(y_test_feat, num_classes)
        
        logger.info(f"Feature dimension: {len(feature_columns)}")
        logger.info(f"Image data shape: {X_train_img.shape}")
        
        # Train ensemble model (features only)
        logger.info("Training ensemble model...")
        ensemble_model, ensemble_acc = self.train_ensemble_model(X_train_feat, y_train_feat, X_test_feat, y_test_feat)
        
        # Train hybrid model (CNN + features)
        logger.info("Training hybrid model...")
        hybrid_model = self.build_hybrid_model(num_classes, len(feature_columns))
        history, trained_hybrid = self.train_hybrid_model(
            hybrid_model, X_train_img, X_train_feat, y_train_cat,
            X_test_img, X_test_feat, y_test_cat, "Hybrid Model"
        )
        
        # Evaluate models
        logger.info("Evaluating models...")
        
        # Evaluate ensemble model
        ensemble_results = self.evaluate_model(
            ensemble_model, None, X_test_feat, y_test_cat, y_test_feat, "Ensemble Model"
        )
        
        # Evaluate hybrid model
        hybrid_results = self.evaluate_model(
            trained_hybrid, X_test_img, X_test_feat, y_test_cat, y_test_feat, "Hybrid Model"
        )
        
        # Generate visualizations and reports
        self.plot_training_curves(history, "Hybrid Model")
        self.plot_confusion_matrix(hybrid_results['confusion_matrix'], "Hybrid Model")
        self.plot_confusion_matrix(ensemble_results['confusion_matrix'], "Ensemble Model")
        
        self.generate_evaluation_report(ensemble_results, "Ensemble Model")
        self.generate_evaluation_report(hybrid_results, "Hybrid Model")
        
        # Save models
        self.save_model(ensemble_model, "Ensemble Model")
        self.save_model(trained_hybrid, "Hybrid Model")
        
        # Print final results
        print("\n" + "="*80)
        print("FINAL RESULTS COMPARISON")
        print("="*80)
        print(f"Ensemble Model (Features Only):")
        print(f"  Accuracy: {ensemble_results['accuracy']:.4f}")
        print(f"  F1-Score: {ensemble_results['f1_score']:.4f}")
        print(f"\nHybrid Model (CNN + Features):")
        print(f"  Accuracy: {hybrid_results['accuracy']:.4f}")
        print(f"  F1-Score: {hybrid_results['f1_score']:.4f}")
        
        best_model = "Hybrid Model" if hybrid_results['accuracy'] > ensemble_results['accuracy'] else "Ensemble Model"
        best_accuracy = max(hybrid_results['accuracy'], ensemble_results['accuracy'])
        
        print(f"\nBest Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        if best_accuracy >= 0.95:
            print("🎉 TARGET ACHIEVED: 95%+ accuracy reached!")
        else:
            print(f"⚠️  Target not reached. Need {0.95 - best_accuracy:.4f} more accuracy.")
        
        logger.info("Advanced scanner identification pipeline complete!")

if __name__ == "__main__":
    # Run advanced scanner identification
    advanced_pipeline = AdvancedScannerIdentification()
    advanced_pipeline.run_full_pipeline()
