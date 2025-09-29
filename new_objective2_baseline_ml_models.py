#!/usr/bin/env python3
"""
New Objective 2: Comprehensive Tampered/Original Detection - Baseline ML Models
Trains traditional ML models with class balancing and comprehensive evaluation
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

# ML libraries
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Class balancing
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Advanced ML libraries
import xgboost as xgb
import lightgbm as lgb

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveObjective2BaselineML:
    def __init__(self, features_path="new_objective2_features", output_path="new_objective2_baseline_ml_results"):
        self.features_path = features_path
        self.output_path = output_path
        
        # Create output directories
        self.create_output_directories()
        
        # Data containers
        self.features_df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        
        # Models and results
        self.models = {}
        self.results = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path,
            f"{self.output_path}/models",
            f"{self.output_path}/results",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/reports"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_features_and_prepare_data(self):
        """Load features and prepare data for training"""
        logger.info("="*80)
        logger.info("LOADING FEATURES AND PREPARING DATA")
        logger.info("="*80)
        
        # Load features
        features_file = f"{self.features_path}/feature_vectors/comprehensive_features.csv"
        if not os.path.exists(features_file):
            raise FileNotFoundError(f"Features file not found: {features_file}")
        
        self.features_df = pd.read_csv(features_file)
        logger.info(f"Loaded features: {self.features_df.shape}")
        
        # Separate features and labels
        feature_columns = [col for col in self.features_df.columns 
                          if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]
        
        X = self.features_df[feature_columns].values
        y = self.features_df['label'].values
        
        # Handle missing values
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        # Split into train and test (using the same split as preprocessing)
        train_df = pd.read_csv("new_objective2_preprocessed/metadata/train_split.csv")
        test_df = pd.read_csv("new_objective2_preprocessed/metadata/test_split.csv")
        
        # Get train and test indices
        train_indices = []
        test_indices = []
        
        for idx, row in self.features_df.iterrows():
            if row['image_path'] in train_df['image_path'].values:
                train_indices.append(idx)
            elif row['image_path'] in test_df['image_path'].values:
                test_indices.append(idx)
        
        self.X_train = X[train_indices]
        self.X_test = X[test_indices]
        self.y_train = y[train_indices]
        self.y_test = y[test_indices]
        
        logger.info(f"Train set: {self.X_train.shape}, Labels: {np.bincount(self.y_train)}")
        logger.info(f"Test set: {self.X_test.shape}, Labels: {np.bincount(self.y_test)}")
        
        # Scale features
        self.scaler = RobustScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        logger.info("Data preparation completed!")
    
    def create_baseline_models(self):
        """Create baseline ML models"""
        logger.info("="*80)
        logger.info("CREATING BASELINE ML MODELS")
        logger.info("="*80)
        
        # Load class weights
        with open("new_objective2_preprocessed/metadata/class_weights.json", 'r') as f:
            class_weights = json.load(f)
        
        # Convert to sklearn format
        class_weight_dict = {int(k): v for k, v in class_weights.items()}
        
        # Define models with hyperparameters
        self.models = {
            'Logistic_Regression': LogisticRegression(
                random_state=42, 
                class_weight='balanced',
                max_iter=1000,
                C=1.0
            ),
            
            'Logistic_Regression_Weighted': LogisticRegression(
                random_state=42,
                class_weight=class_weight_dict,
                max_iter=1000,
                C=1.0
            ),
            
            'SVM_Linear': SVC(
                kernel='linear',
                random_state=42,
                class_weight='balanced',
                probability=True
            ),
            
            'SVM_RBF': SVC(
                kernel='rbf',
                random_state=42,
                class_weight='balanced',
                probability=True,
                gamma='scale'
            ),
            
            'Random_Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            
            'Random_Forest_Weighted': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight=class_weight_dict,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            
            'Gradient_Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            
            'XGBoost': xgb.XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=len(self.y_train[self.y_train==0]) / len(self.y_train[self.y_train==1])
            ),
            
            'LightGBM': lgb.LGBMClassifier(
                random_state=42,
                class_weight='balanced',
                verbose=-1,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6
            ),
            
            'KNN': KNeighborsClassifier(
                n_neighbors=5,
                weights='distance'
            ),
            
            'Naive_Bayes': GaussianNB(),
            
            'Decision_Tree': DecisionTreeClassifier(
                random_state=42,
                class_weight='balanced',
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        
        logger.info(f"Created {len(self.models)} baseline models")
    
    def apply_class_balancing_techniques(self):
        """Apply various class balancing techniques"""
        logger.info("="*80)
        logger.info("APPLYING CLASS BALANCING TECHNIQUES")
        logger.info("="*80)
        
        # Define balancing techniques
        balancing_techniques = {
            'SMOTE': SMOTE(random_state=42, k_neighbors=3),
            'ADASYN': ADASYN(random_state=42, n_neighbors=3),
            'BorderlineSMOTE': BorderlineSMOTE(random_state=42, k_neighbors=3),
            'SMOTETomek': SMOTETomek(random_state=42),
            'SMOTEENN': SMOTEENN(random_state=42)
        }
        
        # Apply each technique and train models
        for technique_name, technique in balancing_techniques.items():
            logger.info(f"Applying {technique_name}...")
            
            try:
                # Apply balancing
                X_balanced, y_balanced = technique.fit_resample(self.X_train, self.y_train)
                
                logger.info(f"  Original: {len(self.y_train)} samples")
                logger.info(f"  Balanced: {len(y_balanced)} samples")
                logger.info(f"  Original distribution: {np.bincount(self.y_train)}")
                logger.info(f"  Balanced distribution: {np.bincount(y_balanced)}")
                
                # Train models with balanced data
                technique_models = {}
                for model_name, model in self.models.items():
                    try:
                        # Clone model
                        from sklearn.base import clone
                        balanced_model = clone(model)
                        
                        # Train model
                        balanced_model.fit(X_balanced, y_balanced)
                        
                        # Evaluate
                        y_pred = balanced_model.predict(self.X_test)
                        y_pred_proba = balanced_model.predict_proba(self.X_test)[:, 1] if hasattr(balanced_model, 'predict_proba') else None
                        
                        # Calculate metrics
                        accuracy = accuracy_score(self.y_test, y_pred)
                        precision = precision_score(self.y_test, y_pred)
                        recall = recall_score(self.y_test, y_pred)
                        f1 = f1_score(self.y_test, y_pred)
                        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                        
                        technique_models[f"{model_name}_{technique_name}"] = {
                            'model': balanced_model,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'auc': auc,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba
                        }
                        
                        logger.info(f"    {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                        
                    except Exception as e:
                        logger.warning(f"    {model_name} failed with {technique_name}: {e}")
                        continue
                
                self.results[technique_name] = technique_models
                
            except Exception as e:
                logger.error(f"Failed to apply {technique_name}: {e}")
                continue
        
        logger.info(f"Applied {len(balancing_techniques)} balancing techniques")
    
    def train_ensemble_models(self):
        """Train ensemble models"""
        logger.info("="*80)
        logger.info("TRAINING ENSEMBLE MODELS")
        logger.info("="*80)
        
        # Get best models from each technique
        best_models = {}
        for technique, models in self.results.items():
            best_model_name = max(models.keys(), key=lambda x: models[x]['f1_score'])
            best_models[best_model_name] = models[best_model_name]['model']
            logger.info(f"Best model from {technique}: {best_model_name}")
        
        # Create ensemble models
        ensemble_models = {
            'Voting_Hard': VotingClassifier(
                estimators=list(best_models.items())[:5],  # Top 5 models
                voting='hard'
            ),
            
            'Voting_Soft': VotingClassifier(
                estimators=list(best_models.items())[:5],  # Top 5 models
                voting='soft'
            ),
            
            'Stacking_LR': StackingClassifier(
                estimators=list(best_models.items())[:5],  # Top 5 models
                final_estimator=LogisticRegression(random_state=42),
                cv=5
            ),
            
            'Stacking_RF': StackingClassifier(
                estimators=list(best_models.items())[:5],  # Top 5 models
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                cv=5
            )
        }
        
        # Train ensemble models
        for ensemble_name, ensemble_model in ensemble_models.items():
            try:
                logger.info(f"Training {ensemble_name}...")
                
                # Train ensemble
                ensemble_model.fit(self.X_train, self.y_train)
                
                # Evaluate
                y_pred = ensemble_model.predict(self.X_test)
                y_pred_proba = ensemble_model.predict_proba(self.X_test)[:, 1] if hasattr(ensemble_model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                self.results['Ensemble'] = self.results.get('Ensemble', {})
                self.results['Ensemble'][ensemble_name] = {
                    'model': ensemble_model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                logger.info(f"  {ensemble_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to train {ensemble_name}: {e}")
                continue
        
        logger.info("Ensemble models training completed!")
    
    def perform_hyperparameter_tuning(self):
        """Perform hyperparameter tuning for best models"""
        logger.info("="*80)
        logger.info("PERFORMING HYPERPARAMETER TUNING")
        logger.info("="*80)
        
        # Find best performing models
        all_results = []
        for technique, models in self.results.items():
            for model_name, result in models.items():
                all_results.append((technique, model_name, result['f1_score']))
        
        # Sort by F1 score
        all_results.sort(key=lambda x: x[2], reverse=True)
        
        # Select top 3 models for tuning
        top_models = all_results[:3]
        logger.info(f"Top 3 models for tuning: {[f'{t}_{m}' for t, m, _ in top_models]}")
        
        # Hyperparameter grids
        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            
            'XGBoost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 6, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            
            'LogisticRegression': {
                'C': [0.1, 1.0, 10.0, 100.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            }
        }
        
        # Perform tuning for each top model
        for technique, model_name, f1_score in top_models:
            if any(param_name in model_name for param_name in param_grids.keys()):
                logger.info(f"Tuning {model_name}...")
                
                # Determine parameter grid
                param_grid = None
                base_model = None
                
                if 'RandomForest' in model_name:
                    param_grid = param_grids['RandomForest']
                    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')
                elif 'XGBoost' in model_name:
                    param_grid = param_grids['XGBoost']
                    base_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
                elif 'LogisticRegression' in model_name:
                    param_grid = param_grids['LogisticRegression']
                    base_model = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)
                
                if param_grid and base_model:
                    try:
                        # Use RandomizedSearchCV for efficiency
                        random_search = RandomizedSearchCV(
                            base_model,
                            param_grid,
                            n_iter=20,
                            cv=5,
                            scoring='f1',
                            random_state=42,
                            n_jobs=-1
                        )
                        
                        random_search.fit(self.X_train, self.y_train)
                        
                        # Evaluate tuned model
                        y_pred = random_search.predict(self.X_test)
                        y_pred_proba = random_search.predict_proba(self.X_test)[:, 1] if hasattr(random_search, 'predict_proba') else None
                        
                        accuracy = accuracy_score(self.y_test, y_pred)
                        precision = precision_score(self.y_test, y_pred)
                        recall = recall_score(self.y_test, y_pred)
                        f1 = f1_score(self.y_test, y_pred)
                        auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                        
                        self.results['Tuned'] = self.results.get('Tuned', {})
                        self.results['Tuned'][f"{model_name}_Tuned"] = {
                            'model': random_search.best_estimator_,
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'auc': auc,
                            'y_pred': y_pred,
                            'y_pred_proba': y_pred_proba,
                            'best_params': random_search.best_params_
                        }
                        
                        logger.info(f"  Tuned {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                        logger.info(f"  Best params: {random_search.best_params_}")
                        
                    except Exception as e:
                        logger.error(f"Failed to tune {model_name}: {e}")
                        continue
        
        logger.info("Hyperparameter tuning completed!")
    
    def generate_comprehensive_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        logger.info("="*80)
        logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        logger.info("="*80)
        
        # Collect all results
        all_results = []
        for technique, models in self.results.items():
            for model_name, result in models.items():
                all_results.append({
                    'technique': technique,
                    'model_name': model_name,
                    'accuracy': result['accuracy'],
                    'precision': result['precision'],
                    'recall': result['recall'],
                    'f1_score': result['f1_score'],
                    'auc': result['auc']
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Sort by F1 score
        results_df = results_df.sort_values('f1_score', ascending=False)
        
        # Save results
        results_df.to_csv(f"{self.output_path}/results/all_results.csv", index=False)
        
        # Generate visualizations
        self._plot_results_comparison(results_df)
        self._plot_technique_comparison(results_df)
        self._plot_model_performance(results_df)
        
        # Generate text report
        self._generate_text_report(results_df)
        
        # Save best model
        best_result = results_df.iloc[0]
        best_model_info = self.results[best_result['technique']][best_result['model_name']]
        
        import joblib
        joblib.dump(best_model_info['model'], f"{self.output_path}/models/best_model.pkl")
        joblib.dump(self.scaler, f"{self.output_path}/models/scaler.pkl")
        
        logger.info(f"Best model: {best_result['model_name']} with F1={best_result['f1_score']:.4f}")
        logger.info("Comprehensive evaluation report generated!")
    
    def _plot_results_comparison(self, results_df):
        """Plot results comparison"""
        plt.figure(figsize=(15, 10))
        
        # Top 10 models comparison
        top_10 = results_df.head(10)
        
        plt.subplot(2, 2, 1)
        plt.barh(range(len(top_10)), top_10['f1_score'])
        plt.yticks(range(len(top_10)), [f"{row['technique']}_{row['model_name']}" for _, row in top_10.iterrows()])
        plt.xlabel('F1 Score')
        plt.title('Top 10 Models by F1 Score')
        
        plt.subplot(2, 2, 2)
        plt.barh(range(len(top_10)), top_10['accuracy'])
        plt.yticks(range(len(top_10)), [f"{row['technique']}_{row['model_name']}" for _, row in top_10.iterrows()])
        plt.xlabel('Accuracy')
        plt.title('Top 10 Models by Accuracy')
        
        plt.subplot(2, 2, 3)
        plt.scatter(results_df['accuracy'], results_df['f1_score'], alpha=0.6)
        plt.xlabel('Accuracy')
        plt.ylabel('F1 Score')
        plt.title('Accuracy vs F1 Score')
        
        plt.subplot(2, 2, 4)
        plt.scatter(results_df['auc'], results_df['f1_score'], alpha=0.6)
        plt.xlabel('AUC')
        plt.ylabel('F1 Score')
        plt.title('AUC vs F1 Score')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/results_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_technique_comparison(self, results_df):
        """Plot technique comparison"""
        plt.figure(figsize=(15, 8))
        
        # Group by technique
        technique_stats = results_df.groupby('technique').agg({
            'f1_score': ['mean', 'max', 'std'],
            'accuracy': ['mean', 'max', 'std'],
            'auc': ['mean', 'max', 'std']
        }).round(4)
        
        # Plot F1 score comparison
        plt.subplot(2, 3, 1)
        technique_stats[('f1_score', 'mean')].plot(kind='bar')
        plt.title('Average F1 Score by Technique')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 2)
        technique_stats[('f1_score', 'max')].plot(kind='bar')
        plt.title('Best F1 Score by Technique')
        plt.ylabel('F1 Score')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 3)
        technique_stats[('accuracy', 'mean')].plot(kind='bar')
        plt.title('Average Accuracy by Technique')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 4)
        technique_stats[('accuracy', 'max')].plot(kind='bar')
        plt.title('Best Accuracy by Technique')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 5)
        technique_stats[('auc', 'mean')].plot(kind='bar')
        plt.title('Average AUC by Technique')
        plt.ylabel('AUC')
        plt.xticks(rotation=45)
        
        plt.subplot(2, 3, 6)
        technique_stats[('auc', 'max')].plot(kind='bar')
        plt.title('Best AUC by Technique')
        plt.ylabel('AUC')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/technique_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self, results_df):
        """Plot model performance metrics"""
        plt.figure(figsize=(15, 10))
        
        # Top 15 models for detailed comparison
        top_15 = results_df.head(15)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc']
        
        for i, metric in enumerate(metrics):
            plt.subplot(2, 3, i+1)
            plt.barh(range(len(top_15)), top_15[metric])
            plt.yticks(range(len(top_15)), [f"{row['technique']}_{row['model_name']}" for _, row in top_15.iterrows()])
            plt.xlabel(metric.replace('_', ' ').title())
            plt.title(f'Top 15 Models by {metric.replace("_", " ").title()}')
        
        # Confusion matrix for best model
        plt.subplot(2, 3, 6)
        best_result = results_df.iloc[0]
        best_model_info = self.results[best_result['technique']][best_result['model_name']]
        
        cm = confusion_matrix(self.y_test, best_model_info['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_result["model_name"]}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/model_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results_df):
        """Generate comprehensive text report"""
        report_path = f"{self.output_path}/reports/baseline_ml_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE OBJECTIVE 2 BASELINE ML MODELS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Features: {self.X_train.shape[1]}\n")
            f.write(f"Training class distribution: {np.bincount(self.y_train)}\n")
            f.write(f"Test class distribution: {np.bincount(self.y_test)}\n\n")
            
            # Best Results
            f.write("TOP 10 RESULTS\n")
            f.write("-"*40 + "\n")
            for i, (_, row) in enumerate(results_df.head(10).iterrows()):
                f.write(f"{i+1:2d}. {row['technique']}_{row['model_name']}\n")
                f.write(f"    Accuracy: {row['accuracy']:.4f}\n")
                f.write(f"    Precision: {row['precision']:.4f}\n")
                f.write(f"    Recall: {row['recall']:.4f}\n")
                f.write(f"    F1 Score: {row['f1_score']:.4f}\n")
                f.write(f"    AUC: {row['auc']:.4f}\n\n")
            
            # Technique Summary
            f.write("TECHNIQUE SUMMARY\n")
            f.write("-"*40 + "\n")
            technique_summary = results_df.groupby('technique').agg({
                'f1_score': ['count', 'mean', 'max', 'std'],
                'accuracy': ['mean', 'max'],
                'auc': ['mean', 'max']
            }).round(4)
            
            for technique in technique_summary.index:
                f.write(f"{technique}:\n")
                f.write(f"  Models: {technique_summary.loc[technique, ('f1_score', 'count')]}\n")
                f.write(f"  Average F1: {technique_summary.loc[technique, ('f1_score', 'mean')]:.4f}\n")
                f.write(f"  Best F1: {technique_summary.loc[technique, ('f1_score', 'max')]:.4f}\n")
                f.write(f"  Average Accuracy: {technique_summary.loc[technique, ('accuracy', 'mean')]:.4f}\n")
                f.write(f"  Best Accuracy: {technique_summary.loc[technique, ('accuracy', 'max')]:.4f}\n\n")
            
            # Performance Analysis
            f.write("PERFORMANCE ANALYSIS\n")
            f.write("-"*40 + "\n")
            best_result = results_df.iloc[0]
            f.write(f"Best Model: {best_result['technique']}_{best_result['model_name']}\n")
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
                    f.write(f"  {row['technique']}_{row['model_name']}: {row['accuracy']:.4f}\n")
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
                f.write("1. Deep learning models with CNN architectures\n")
                f.write("2. Advanced ensemble methods\n")
                f.write("3. More sophisticated feature engineering\n")
                f.write("4. Data augmentation techniques\n")
                f.write("5. Transfer learning approaches\n")
                f.write("6. Hyperparameter optimization with larger search spaces\n")
                f.write("7. Cross-validation with more folds\n")
                f.write("8. Feature selection and dimensionality reduction\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_baseline_ml_pipeline(self):
        """Run complete baseline ML pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE OBJECTIVE 2 BASELINE ML PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load and prepare data
        self.load_features_and_prepare_data()
        
        # Step 2: Create baseline models
        self.create_baseline_models()
        
        # Step 3: Apply class balancing techniques
        self.apply_class_balancing_techniques()
        
        # Step 4: Train ensemble models
        self.train_ensemble_models()
        
        # Step 5: Perform hyperparameter tuning
        self.perform_hyperparameter_tuning()
        
        # Step 6: Generate comprehensive evaluation report
        self.generate_comprehensive_evaluation_report()
        
        logger.info("✅ COMPREHENSIVE BASELINE ML PIPELINE COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        
        return self.results

def main():
    """Main function to run comprehensive baseline ML pipeline"""
    baseline_ml = ComprehensiveObjective2BaselineML()
    results = baseline_ml.run_complete_baseline_ml_pipeline()
    
    print("\n" + "="*80)
    print("BASELINE ML PIPELINE SUMMARY")
    print("="*80)
    
    # Find best result
    all_results = []
    for technique, models in results.items():
        for model_name, result in models.items():
            all_results.append((technique, model_name, result['f1_score'], result['accuracy']))
    
    if all_results:
        best_result = max(all_results, key=lambda x: x[2])  # Sort by F1 score
        print(f"Best Model: {best_result[1]} ({best_result[0]})")
        print(f"Best F1 Score: {best_result[2]:.4f}")
        print(f"Best Accuracy: {best_result[3]:.4f}")
        
        target_accuracy = 0.90
        if best_result[3] >= target_accuracy:
            print(f"🎉 TARGET ACHIEVED: {target_accuracy*100}%+ accuracy reached!")
        else:
            print(f"📈 Progress: {best_result[3]*100:.2f}% accuracy (need {target_accuracy*100:.1f}%)")
            print(f"Gap to target: {(target_accuracy - best_result[3])*100:.2f} percentage points")
    
    print("\nNext step: Run deep learning models to push towards 90%+ accuracy!")

if __name__ == "__main__":
    main()
