#!/usr/bin/env python3
"""
High Accuracy Scanner Identification System
Target: 95%+ accuracy by combining all extracted features and PRNU fingerprints
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

class HighAccuracyScannerID:
    def __init__(self, features_path="features", output_path="high_accuracy_models"):
        self.features_path = features_path
        self.output_path = output_path
        self.random_seed = 42
        
        # Create output directories
        os.makedirs(f"{self.output_path}/trained_models", exist_ok=True)
        os.makedirs(f"{self.output_path}/evaluation_results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_path}/metadata", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.best_models = {}
        
    def load_features_and_optimize(self):
        """Load features and optimize preprocessing"""
        logger.info("Loading and optimizing features...")
        
        # Load feature data
        features_csv = os.path.join(self.features_path, "feature_vectors", "scanner_features.csv")
        if not os.path.exists(features_csv):
            logger.error(f"Features file not found: {features_csv}")
            return None
        
        df = pd.read_csv(features_csv)
        logger.info(f"Loaded feature data: {df.shape}")
        
        # Separate features and labels
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Handle NaN values - use advanced imputation
        logger.info("Handling missing values...")
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        
        # Remove features with very low variance
        from sklearn.feature_selection import VarianceThreshold
        variance_selector = VarianceThreshold(threshold=0.001)
        X = variance_selector.fit_transform(X)
        
        # Select top features using mutual information
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        # Use more features for better performance
        k_best = min(100, X.shape[1])  # Use up to 100 best features
        selector = SelectKBest(mutual_info_classif, k=k_best)
        
        # Fit selector on training data only
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train_temp = X[train_mask]
        y_train_temp = y[train_mask]
        
        # Encode labels for feature selection
        y_encoded_temp = self.label_encoder.fit_transform(y_train_temp)
        
        # Select features
        X_train_selected = selector.fit_transform(X_train_temp, y_encoded_temp)
        X_test_selected = selector.transform(X[test_mask])
        
        # Get final data
        X_train = X_train_selected
        y_train = y_encoded_temp
        X_test = X_test_selected
        y_test = self.label_encoder.transform(y[test_mask])
        
        logger.info(f"Selected {X_train.shape[1]} features")
        logger.info(f"Training set: {X_train.shape}")
        logger.info(f"Test set: {X_test.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        # Store the preprocessing pipeline
        self.imputer = imputer
        self.variance_selector = variance_selector  
        self.feature_selector = selector
        
        return X_train, X_test, y_train, y_test
        
    def optimize_hyperparameters(self, X_train, y_train):
        """Optimize hyperparameters for each model"""
        logger.info("Optimizing hyperparameters...")
        
        # Scale data for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        optimized_models = {}
        
        # 1. Random Forest with extensive hyperparameter tuning
        logger.info("Optimizing Random Forest...")
        rf_param_grid = {
            'n_estimators': [500, 800, 1000],
            'max_depth': [15, 20, 25, None],
            'min_samples_split': [2, 3, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True],
            'class_weight': ['balanced', None]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_seed, n_jobs=-1),
            rf_param_grid,
            cv=3,  # Reduced CV for speed
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        rf_grid.fit(X_train, y_train)
        optimized_models['RandomForest'] = rf_grid.best_estimator_
        logger.info(f"Best RF params: {rf_grid.best_params_}")
        logger.info(f"Best RF score: {rf_grid.best_score_:.4f}")
        
        # 2. Gradient Boosting
        logger.info("Optimizing Gradient Boosting...")
        gb_param_grid = {
            'n_estimators': [200, 300, 500],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.05, 0.1, 0.15],
            'subsample': [0.8, 0.9, 1.0],
            'max_features': ['sqrt', 'log2']
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_seed),
            gb_param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        gb_grid.fit(X_train, y_train)
        optimized_models['GradientBoosting'] = gb_grid.best_estimator_
        logger.info(f"Best GB params: {gb_grid.best_params_}")
        logger.info(f"Best GB score: {gb_grid.best_score_:.4f}")
        
        # 3. SVM
        logger.info("Optimizing SVM...")
        svm_param_grid = {
            'C': [1, 10, 100, 1000],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly'],
            'class_weight': ['balanced', None]
        }
        
        svm_grid = GridSearchCV(
            SVC(random_state=self.random_seed, probability=True),
            svm_param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        svm_grid.fit(X_train_scaled, y_train)
        optimized_models['SVM'] = svm_grid.best_estimator_
        logger.info(f"Best SVM params: {svm_grid.best_params_}")
        logger.info(f"Best SVM score: {svm_grid.best_score_:.4f}")
        
        # 4. Logistic Regression
        logger.info("Optimizing Logistic Regression...")
        lr_param_grid = {
            'C': [0.1, 1, 10, 100],
            'solver': ['liblinear', 'lbfgs'],
            'max_iter': [1000, 2000],
            'class_weight': ['balanced', None]
        }
        
        lr_grid = GridSearchCV(
            LogisticRegression(random_state=self.random_seed),
            lr_param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        lr_grid.fit(X_train_scaled, y_train)
        optimized_models['LogisticRegression'] = lr_grid.best_estimator_
        logger.info(f"Best LR params: {lr_grid.best_params_}")
        logger.info(f"Best LR score: {lr_grid.best_score_:.4f}")
        
        return optimized_models
    
    def create_ensemble_model(self, optimized_models, X_train, y_train):
        """Create optimized ensemble model"""
        logger.info("Creating ensemble model...")
        
        # Scale data for SVM and LR
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Create ensemble with optimized models
        ensemble_models = []
        
        # Add Random Forest and Gradient Boosting (don't need scaling)
        ensemble_models.append(('rf', optimized_models['RandomForest']))
        ensemble_models.append(('gb', optimized_models['GradientBoosting']))
        
        # Create scaled versions for SVM and LR
        svm_scaled = optimized_models['SVM']
        lr_scaled = optimized_models['LogisticRegression']
        
        # Create voting classifier
        voting_clf = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use probabilities for better ensemble
            n_jobs=-1
        )
        
        voting_clf.fit(X_train, y_train)
        
        return voting_clf
    
    def evaluate_models(self, models, X_train, X_test, y_train, y_test):
        """Evaluate all models comprehensively"""
        logger.info("Evaluating all models...")
        
        results = {}
        
        # Scale data for models that need it
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Use scaled data for SVM and LR
            if name in ['SVM', 'LogisticRegression']:
                X_train_eval = X_train_scaled
                X_test_eval = X_test_scaled
            else:
                X_train_eval = X_train
                X_test_eval = X_test
            
            # Train model if not already trained
            if hasattr(model, 'predict'):
                # Model is already trained
                pass
            else:
                model.fit(X_train_eval, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_eval)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation on training set
            cv_scores = cross_val_score(
                model, X_train_eval, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            results[name] = {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'confusion_matrix': cm,
                'y_pred': y_pred,
                'model': model
            }
            
            logger.info(f"{name} - Test Accuracy: {accuracy:.4f}, CV F1: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
        
        return results
    
    def plot_results(self, results):
        """Plot comprehensive results"""
        logger.info("Creating visualizations...")
        
        # 1. Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        accuracies = [results[m]['test_accuracy'] for m in models]
        f1_scores = [results[m]['test_f1'] for m in models]
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]
        
        # Test accuracy
        axes[0, 0].bar(models, accuracies, color='skyblue')
        axes[0, 0].set_title('Test Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Test F1-score
        axes[0, 1].bar(models, f1_scores, color='lightgreen')
        axes[0, 1].set_title('Test F1-Score')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Cross-validation F1-score
        axes[1, 0].bar(models, cv_means, yerr=cv_stds, capsize=5, color='orange')
        axes[1, 0].set_title('Cross-Validation F1-Score')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Model comparison
        x_pos = np.arange(len(models))
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, f1_scores, width, label='Test F1-Score', alpha=0.8)
        axes[1, 1].set_title('Model Comparison')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Best model confusion matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_cm = results[best_model_name]['confusion_matrix']
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(best_cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_,
                   yticklabels=self.label_encoder.classes_)
        plt.title(f'{best_model_name} - Confusion Matrix (Best Model)')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/best_model_confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models_and_results(self, results):
        """Save all models and results"""
        logger.info("Saving models and results...")
        
        # Save models
        for name, result in results.items():
            model_path = f"{self.output_path}/trained_models/{name.lower()}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save preprocessing pipeline
        pipeline_path = f"{self.output_path}/trained_models/preprocessing_pipeline.pkl"
        pipeline = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'variance_selector': self.variance_selector,
            'feature_selector': self.feature_selector
        }
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Save results summary
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Test_Accuracy': result['test_accuracy'],
                'Test_Precision': result['test_precision'],
                'Test_Recall': result['test_recall'],
                'Test_F1': result['test_f1'],
                'CV_Mean_F1': result['cv_mean'],
                'CV_Std_F1': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f"{self.output_path}/evaluation_results/model_results.csv", index=False)
        
        # Generate detailed report
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_accuracy = results[best_model_name]['test_accuracy']
        
        report_path = f"{self.output_path}/metadata/high_accuracy_report.txt"
        with open(report_path, 'w') as f:
            f.write("HIGH ACCURACY SCANNER IDENTIFICATION REPORT\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Best Model: {best_model_name}\n")
            f.write(f"Best Accuracy: {best_accuracy:.4f}\n\n")
            
            f.write("All Model Results:\n")
            f.write("-" * 40 + "\n")
            for name, result in results.items():
                f.write(f"{name}:\n")
                f.write(f"  Test Accuracy: {result['test_accuracy']:.4f}\n")
                f.write(f"  Test F1-Score: {result['test_f1']:.4f}\n")
                f.write(f"  CV F1-Score: {result['cv_mean']:.4f}±{result['cv_std']:.4f}\n\n")
        
        return best_model_name, best_accuracy
    
    def run_high_accuracy_pipeline(self):
        """Run the complete high accuracy pipeline"""
        logger.info("Starting high accuracy scanner identification pipeline...")
        
        # 1. Load and optimize features
        data = self.load_features_and_optimize()
        if data is None:
            logger.error("Failed to load features")
            return
        
        X_train, X_test, y_train, y_test = data
        
        # 2. Optimize hyperparameters
        optimized_models = self.optimize_hyperparameters(X_train, y_train)
        
        # 3. Create ensemble model
        ensemble_model = self.create_ensemble_model(optimized_models, X_train, y_train)
        optimized_models['Ensemble'] = ensemble_model
        
        # 4. Evaluate all models
        results = self.evaluate_models(optimized_models, X_train, X_test, y_train, y_test)
        
        # 5. Create visualizations
        self.plot_results(results)
        
        # 6. Save models and results
        best_model_name, best_accuracy = self.save_models_and_results(results)
        
        # 7. Print final results
        print("\n" + "="*80)
        print("HIGH ACCURACY SCANNER IDENTIFICATION RESULTS")
        print("="*80)
        
        for name, result in results.items():
            print(f"{name}:")
            print(f"  Test Accuracy: {result['test_accuracy']:.4f}")
            print(f"  Test F1-Score: {result['test_f1']:.4f}")
            print(f"  CV F1-Score: {result['cv_mean']:.4f}±{result['cv_std']:.4f}")
            print()
        
        print(f"Best Model: {best_model_name}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        if best_accuracy >= 0.95:
            print("🎉 TARGET ACHIEVED: 95%+ accuracy reached!")
        else:
            print(f"⚠️  Target not reached. Need {0.95 - best_accuracy:.4f} more accuracy.")
            print("\nRecommendations for further improvement:")
            print("1. Collect more high-quality training data")
            print("2. Engineer additional domain-specific features")
            print("3. Try advanced ensemble methods (stacking)")
            print("4. Use deep learning with more sophisticated architectures")
        
        logger.info("High accuracy pipeline complete!")

if __name__ == "__main__":
    # Run high accuracy scanner identification
    high_accuracy_pipeline = HighAccuracyScannerID()
    high_accuracy_pipeline.run_high_accuracy_pipeline()
