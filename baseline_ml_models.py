#!/usr/bin/env python3
"""
Baseline Machine Learning Models for AI_TraceFinder - Objective 1: Scanner Identification
Trains and evaluates multiple ML models for scanner identification
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.pipeline import Pipeline
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BaselineMLModels:
    def __init__(self, features_path="features", output_path="baseline_models"):
        self.features_path = features_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize models and results storage
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/trained_models",
            f"{self.output_path}/evaluation_results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/metadata"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_feature_data(self):
        """Load the extracted feature data"""
        logger.info("Loading feature data...")
        
        # Load CSV data
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
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into train and test
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X_imputed[train_mask]
        y_train = y_encoded[train_mask]
        X_test = X_imputed[test_mask]
        y_test = y_encoded[test_mask]
        
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train, X_test, y_train, y_test, feature_columns
    
    def train_logistic_regression(self, X_train, y_train):
        """Train Logistic Regression model"""
        logger.info("Training Logistic Regression...")
        
        # Create pipeline with scaling
        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                random_state=self.random_seed,
                max_iter=1000,
                multi_class='multinomial',
                solver='lbfgs'
            ))
        ])
        
        # Train model
        lr_pipeline.fit(X_train, y_train)
        
        return lr_pipeline
    
    def train_svm(self, X_train, y_train):
        """Train Support Vector Machine model"""
        logger.info("Training Support Vector Machine...")
        
        # Create pipeline with scaling
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(
                random_state=self.random_seed,
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True
            ))
        ])
        
        # Train model
        svm_pipeline.fit(X_train, y_train)
        
        return svm_pipeline
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest model"""
        logger.info("Training Random Forest...")
        
        # Create pipeline with scaling
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_seed,
                n_jobs=-1
            ))
        ])
        
        # Train model
        rf_pipeline.fit(X_train, y_train)
        
        return rf_pipeline
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a single model"""
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            roc_auc = None
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        return results
    
    def cross_validate_model(self, model, X_train, y_train, model_name):
        """Perform cross-validation"""
        logger.info(f"Cross-validating {model_name}...")
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        
        # Cross-validation scores
        cv_accuracy = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
        cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
        
        cv_results = {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'cv_accuracy_scores': cv_accuracy,
            'cv_f1_scores': cv_f1
        }
        
        return cv_results
    
    def train_all_models(self, X_train, X_test, y_train, y_test):
        """Train all baseline models"""
        logger.info("Training all baseline models...")
        
        # Train models
        self.models['Logistic Regression'] = self.train_logistic_regression(X_train, y_train)
        self.models['SVM'] = self.train_svm(X_train, y_train)
        self.models['Random Forest'] = self.train_random_forest(X_train, y_train)
        
        # Evaluate each model
        for model_name, model in self.models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Test set evaluation
            test_results = self.evaluate_model(model, X_test, y_test, model_name)
            
            # Cross-validation
            cv_results = self.cross_validate_model(model, X_train, y_train, model_name)
            
            # Combine results
            self.results[model_name] = {**test_results, **cv_results}
        
        logger.info("All models trained and evaluated!")
    
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("Generating evaluation report...")
        
        print("\n" + "="*80)
        print("BASELINE ML MODELS EVALUATION REPORT")
        print("="*80)
        
        # Model comparison table
        print("\nModel Performance Comparison:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            accuracy = results['accuracy']
            precision = results['precision']
            recall = results['recall']
            f1 = results['f1_score']
            roc_auc = results['roc_auc'] if results['roc_auc'] is not None else 'N/A'
            
            print(f"{model_name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {roc_auc:<10}")
        
        # Cross-validation results
        print("\nCross-Validation Results:")
        print("-" * 80)
        print(f"{'Model':<20} {'CV Accuracy':<15} {'CV F1-Score':<15}")
        print("-" * 80)
        
        for model_name, results in self.results.items():
            cv_acc = results['cv_accuracy_mean']
            cv_acc_std = results['cv_accuracy_std']
            cv_f1 = results['cv_f1_mean']
            cv_f1_std = results['cv_f1_std']
            
            print(f"{model_name:<20} {cv_acc:.4f}±{cv_acc_std:.4f} {cv_f1:.4f}±{cv_f1_std:.4f}")
        
        # Best model identification
        best_model = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        print(f"\nBest Model (by F1-Score): {best_model}")
        print(f"F1-Score: {self.results[best_model]['f1_score']:.4f}")
        print(f"Accuracy: {self.results[best_model]['accuracy']:.4f}")
        
        # Save detailed report
        self.save_detailed_report()
    
    def save_detailed_report(self):
        """Save detailed evaluation report to file"""
        report_path = os.path.join(self.output_path, "metadata", "baseline_evaluation_report.txt")
        
        with open(report_path, 'w') as f:
            f.write("BASELINE ML MODELS EVALUATION REPORT\n")
            f.write("="*80 + "\n\n")
            
            # Model comparison
            f.write("Model Performance Comparison:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, results in self.results.items():
                accuracy = results['accuracy']
                precision = results['precision']
                recall = results['recall']
                f1 = results['f1_score']
                roc_auc = results['roc_auc'] if results['roc_auc'] is not None else 'N/A'
                
                f.write(f"{model_name:<20} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f} {roc_auc:<10}\n")
            
            # Cross-validation results
            f.write("\nCross-Validation Results:\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Model':<20} {'CV Accuracy':<15} {'CV F1-Score':<15}\n")
            f.write("-" * 80 + "\n")
            
            for model_name, results in self.results.items():
                cv_acc = results['cv_accuracy_mean']
                cv_acc_std = results['cv_accuracy_std']
                cv_f1 = results['cv_f1_mean']
                cv_f1_std = results['cv_f1_std']
                
                f.write(f"{model_name:<20} {cv_acc:.4f}±{cv_acc_std:.4f} {cv_f1:.4f}±{cv_f1_std:.4f}\n")
            
            # Detailed results for each model
            for model_name, results in self.results.items():
                f.write(f"\n{model_name} - Detailed Results:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Test Accuracy: {results['accuracy']:.4f}\n")
                f.write(f"Test Precision: {results['precision']:.4f}\n")
                f.write(f"Test Recall: {results['recall']:.4f}\n")
                f.write(f"Test F1-Score: {results['f1_score']:.4f}\n")
                if results['roc_auc'] is not None:
                    f.write(f"Test ROC-AUC: {results['roc_auc']:.4f}\n")
                f.write(f"CV Accuracy: {results['cv_accuracy_mean']:.4f} ± {results['cv_accuracy_std']:.4f}\n")
                f.write(f"CV F1-Score: {results['cv_f1_mean']:.4f} ± {results['cv_f1_std']:.4f}\n")
        
        logger.info(f"Detailed report saved to {report_path}")
    
    def visualize_confusion_matrices(self, y_test):
        """Visualize confusion matrices for all models"""
        logger.info("Creating confusion matrix visualizations...")
        
        n_models = len(self.models)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            
            # Create heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       ax=axes[i])
            
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            
            # Rotate x-axis labels for better readability
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "visualizations", "confusion_matrices.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Confusion matrices saved")
    
    def visualize_feature_importance(self, feature_columns):
        """Visualize feature importance for Random Forest"""
        logger.info("Creating feature importance visualization...")
        
        if 'Random Forest' in self.models:
            # Get feature importance from Random Forest
            rf_model = self.models['Random Forest']
            feature_importance = rf_model.named_steps['classifier'].feature_importances_
            
            # Create DataFrame for easier handling
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': feature_importance
            }).sort_values('importance', ascending=False)
            
            # Plot top 20 features
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(20)
            
            sns.barplot(data=top_features, x='importance', y='feature')
            plt.title('Random Forest - Top 20 Feature Importance')
            plt.xlabel('Feature Importance')
            plt.ylabel('Feature Name')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_path, "visualizations", "feature_importance.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save feature importance data
            importance_df.to_csv(os.path.join(self.output_path, "metadata", "feature_importance.csv"), 
                               index=False)
            
            logger.info("Feature importance visualization saved")
    
    def save_models(self):
        """Save trained models"""
        logger.info("Saving trained models...")
        
        for model_name, model in self.models.items():
            # Create safe filename
            safe_name = model_name.replace(' ', '_').lower()
            model_path = os.path.join(self.output_path, "trained_models", f"{safe_name}.pkl")
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Saved {model_name} to {model_path}")
        
        # Save label encoder
        encoder_path = os.path.join(self.output_path, "trained_models", "label_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        logger.info(f"Saved label encoder to {encoder_path}")
    
    def save_results(self):
        """Save evaluation results"""
        logger.info("Saving evaluation results...")
        
        # Convert results to DataFrame for easier handling
        results_data = []
        for model_name, results in self.results.items():
            row = {
                'model': model_name,
                'accuracy': results['accuracy'],
                'precision': results['precision'],
                'recall': results['recall'],
                'f1_score': results['f1_score'],
                'roc_auc': results['roc_auc'],
                'cv_accuracy_mean': results['cv_accuracy_mean'],
                'cv_accuracy_std': results['cv_accuracy_std'],
                'cv_f1_mean': results['cv_f1_mean'],
                'cv_f1_std': results['cv_f1_std']
            }
            results_data.append(row)
        
        results_df = pd.DataFrame(results_data)
        results_path = os.path.join(self.output_path, "evaluation_results", "model_results.csv")
        results_df.to_csv(results_path, index=False)
        
        logger.info(f"Results saved to {results_path}")
    
    def run_full_pipeline(self):
        """Run the complete baseline ML pipeline"""
        logger.info("Starting baseline ML pipeline...")
        
        # Load data
        data = self.load_feature_data()
        if data is None:
            return
        
        X_train, X_test, y_train, y_test, feature_columns = data
        
        # Train and evaluate models
        self.train_all_models(X_train, X_test, y_train, y_test)
        
        # Generate reports and visualizations
        self.generate_evaluation_report()
        self.visualize_confusion_matrices(y_test)
        self.visualize_feature_importance(feature_columns)
        
        # Save models and results
        self.save_models()
        self.save_results()
        
        logger.info("Baseline ML pipeline complete!")

if __name__ == "__main__":
    ml_pipeline = BaselineMLModels()
    ml_pipeline.run_full_pipeline()
