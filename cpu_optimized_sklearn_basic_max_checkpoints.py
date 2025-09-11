#!/usr/bin/env python3
"""
SKLEARN-BASED CLASSIFIER WITH MAXIMUM CHECKPOINTS (BASIC VERSION)
Uses only available packages with comprehensive checkpointing
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import pickle
import joblib

# Traditional ML imports (only available ones)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('cpu_optimized_sklearn_basic_max_checkpoints.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

class BasicMaxCheckpointsSklearnClassifier:
    def __init__(self, features_path="features", output_path="sklearn_basic_max_checkpoints_results"):
        self.features_path = features_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Load existing features
        self.load_existing_data()
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/models",
            f"{self.output_path}/checkpoints",
            f"{self.output_path}/results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/logs"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_existing_data(self):
        """Load existing features"""
        logger.info("="*80)
        logger.info("🚀 LOADING EXISTING FEATURES")
        logger.info("="*80)
        
        try:
            # Load existing features
            self.X_features = np.load(f"{self.features_path}/feature_vectors/X_features.npy")
            self.y_labels = np.load(f"{self.features_path}/feature_vectors/y_labels.npy", allow_pickle=True)
            self.feature_names = pd.read_csv(f"{self.features_path}/feature_vectors/feature_names.csv")
            
            logger.info(f"✅ Loaded {len(self.X_features)} samples with {self.X_features.shape[1]} features")
            logger.info(f"✅ Feature names: {len(self.feature_names)} features")
            logger.info(f"✅ Unique classes: {len(np.unique(self.y_labels))}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error loading existing data: {e}")
            return False
    
    def create_models(self):
        """Create all models with optimized hyperparameters"""
        logger.info("🏗️ CREATING MODELS WITH OPTIMIZED HYPERPARAMETERS")
        
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=10,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=self.random_seed
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=25,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'LogisticRegression': LogisticRegression(
                C=10.0,
                max_iter=2000,
                random_state=self.random_seed,
                n_jobs=-1
            ),
            'SVM': SVC(
                C=10.0,
                kernel='rbf',
                gamma='scale',
                random_state=self.random_seed,
                probability=True
            ),
            'MLP': MLPClassifier(
                hidden_layer_sizes=(1024, 512, 256),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                learning_rate='adaptive',
                max_iter=1000,
                random_state=self.random_seed
            ),
            'Ridge': RidgeClassifier(
                alpha=0.1,
                random_state=self.random_seed
            )
        }
        
        return models
    
    def save_checkpoint(self, model_name, model, accuracy, epoch=None):
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if epoch is not None:
            filename = f"{self.output_path}/checkpoints/{model_name}_epoch_{epoch:03d}_acc_{accuracy:.4f}_{timestamp}.pkl"
        else:
            filename = f"{self.output_path}/checkpoints/{model_name}_final_acc_{accuracy:.4f}_{timestamp}.pkl"
        
        try:
            joblib.dump(model, filename)
            logger.info(f"✅ Checkpoint saved: {filename}")
            return filename
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint: {e}")
            return None
    
    def train_model_with_checkpoints(self, model_name, model, X_train, y_train, X_val, y_val):
        """Train model with comprehensive checkpointing"""
        logger.info(f"🎯 TRAINING {model_name} WITH MAXIMUM CHECKPOINTS")
        
        try:
            # Initial training
            logger.info(f"Starting {model_name} training...")
            start_time = time.time()
            
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            
            # Evaluate on validation set
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            logger.info(f"✅ {model_name} training completed in {training_time:.2f}s")
            logger.info(f"✅ {model_name} validation accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            
            # Save checkpoint
            checkpoint_file = self.save_checkpoint(model_name, model, accuracy)
            
            # Cross-validation for more robust evaluation
            logger.info(f"Running cross-validation for {model_name}...")
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
            logger.info(f"✅ {model_name} CV accuracy: {cv_mean:.4f} ± {cv_std:.4f}")
            
            # Save results
            results = {
                'model_name': model_name,
                'validation_accuracy': accuracy,
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'training_time': training_time,
                'checkpoint_file': checkpoint_file,
                'model': model
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {model_name} training failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate model on test set"""
        logger.info(f"📊 EVALUATING {model_name} ON TEST SET")
        
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            logger.info(f"{model_name} Test Results:")
            logger.info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  Precision: {precision:.4f}")
            logger.info(f"  Recall: {recall:.4f}")
            logger.info(f"  F1-Score: {f1:.4f}")
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Classification report
            report = classification_report(y_test, y_pred, target_names=self.label_encoder.classes_)
            
            # Save detailed results
            results = {
                'model_name': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': cm,
                'classification_report': report,
                'predictions': y_pred,
                'true_labels': y_test,
                'prediction_probabilities': y_pred_proba
            }
            
            return results
            
        except Exception as e:
            logger.error(f"❌ {model_name} evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def run_complete_pipeline(self):
        """Run the complete training pipeline with maximum checkpointing"""
        logger.info("🚀 STARTING SKLEARN BASIC PIPELINE WITH MAXIMUM CHECKPOINTS")
        
        try:
            # 1. Load existing data
            if not self.load_existing_data():
                return None
            
            # 2. Prepare data
            logger.info("Preparing data...")
            X = self.X_features
            y = self.y_labels
            
            # Encode labels
            y_encoded = self.label_encoder.fit_transform(y)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_encoded, test_size=0.2, random_state=self.random_seed, stratify=y_encoded
            )
            
            # Further split training for validation
            X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
                X_train, y_train, test_size=0.2, random_state=self.random_seed, stratify=y_train
            )
            
            logger.info(f"Training set: {X_train_split.shape}")
            logger.info(f"Validation set: {X_val_split.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
            
            # 3. Create models
            models = self.create_models()
            
            # 4. Train all models with checkpoints
            all_results = {}
            
            for model_name, model in models.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"TRAINING {model_name}")
                logger.info(f"{'='*60}")
                
                # Train with checkpoints
                train_results = self.train_model_with_checkpoints(
                    model_name, model, X_train_split, y_train_split, X_val_split, y_val_split
                )
                
                if train_results:
                    # Evaluate on test set
                    test_results = self.evaluate_model(model, X_test, y_test, model_name)
                    
                    if test_results:
                        all_results[model_name] = {
                            'train_results': train_results,
                            'test_results': test_results
                        }
                        
                        # Save final checkpoint
                        final_checkpoint = self.save_checkpoint(
                            f"{model_name}_final", model, test_results['accuracy']
                        )
                        all_results[model_name]['final_checkpoint'] = final_checkpoint
            
            # 5. Save all results
            with open(f"{self.output_path}/results/all_results.pkl", 'wb') as f:
                pickle.dump(all_results, f)
            
            # Save label encoder and scaler
            with open(f"{self.output_path}/models/label_encoder.pkl", 'wb') as f:
                pickle.dump(self.label_encoder, f)
            
            with open(f"{self.output_path}/models/scaler.pkl", 'wb') as f:
                pickle.dump(self.scaler, f)
            
            # 6. Print final results
            logger.info("\n" + "="*80)
            logger.info("🏁 FINAL RESULTS SUMMARY")
            logger.info("="*80)
            
            best_accuracy = 0
            best_model = None
            
            for model_name, results in all_results.items():
                if results['test_results']:
                    accuracy = results['test_results']['accuracy']
                    logger.info(f"{model_name}: {accuracy:.4f} ({accuracy*100:.2f}%)")
                    
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
            
            logger.info(f"\n🏆 BEST MODEL: {best_model} with {best_accuracy:.4f} ({best_accuracy*100:.2f}%) accuracy")
            
            if best_accuracy >= 0.97:
                logger.info("🎉🎉🎉 SUCCESS! 97%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            else:
                gap = 0.97 - best_accuracy
                logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 97%")
            
            return all_results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

if __name__ == "__main__":
    logger.info("🚀 STARTING SKLEARN BASIC PIPELINE WITH MAXIMUM CHECKPOINTS...")
    
    try:
        # Create and run pipeline
        pipeline = BasicMaxCheckpointsSklearnClassifier()
        results = pipeline.run_complete_pipeline()
        
        if results:
            print(f"\n🎯 Sklearn basic pipeline with maximum checkpoints completed!")
            best_accuracy = 0
            best_model = None
            for model_name, result in results.items():
                if result['test_results']:
                    accuracy = result['test_results']['accuracy']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model = model_name
            print(f"Best model: {best_model} with {best_accuracy:.4f} accuracy")
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
