#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Baseline ML Models
Implements baseline machine learning models for tampered image detection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score, roc_auc_score, roc_curve
)
from sklearn.feature_selection import SelectKBest, f_classif, RFE, SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class BaselineMLModels:
    """Baseline machine learning models for tampered image detection"""
    
    def __init__(self, features_path='objective2_forensic_features.csv'):
        self.features_path = features_path
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.models = {}
        self.results = {}
        self.feature_importance = {}
        
    def load_and_prepare_data(self, test_size=0.2, random_state=42):
        """Load features and prepare train-test split"""
        print("=== Loading and Preparing Data ===")
        
        # Load features
        if not os.path.exists(self.features_path):
            raise FileNotFoundError(f"Features file not found: {self.features_path}")
        
        df = pd.read_csv(self.features_path)
        print(f"Loaded features: {df.shape}")
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['image_path', 'label', 'tamper_type', 'category']]
        X = df[feature_cols]
        y = df['label']
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label distribution:")
        print(f"  - Original: {sum(y == 0)}")
        print(f"  - Tampered: {sum(y == 1)}")
        
        # Handle infinite and NaN values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Train set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def scale_features(self, scaler_type='standard'):
        """Scale features using different scalers"""
        print(f"\n=== Scaling Features ({scaler_type}) ===")
        
        if scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif scaler_type == 'robust':
            self.scaler = RobustScaler()
        elif scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Scaler type must be 'standard', 'robust', or 'minmax'")
        
        # Fit scaler on training data
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Features scaled using {scaler_type} scaler")
        print(f"Train scaled shape: {self.X_train_scaled.shape}")
        print(f"Test scaled shape: {self.X_test_scaled.shape}")
        
        return self.X_train_scaled, self.X_test_scaled
    
    def feature_selection(self, method='selectkbest', k=20):
        """Perform feature selection"""
        print(f"\n=== Feature Selection ({method}, k={k}) ===")
        
        if method == 'selectkbest':
            selector = SelectKBest(score_func=f_classif, k=k)
            self.X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = selector.transform(self.X_test_scaled)
            self.selected_features = selector.get_support(indices=True)
            
        elif method == 'rfe':
            # Use Random Forest for RFE
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(rf, n_features_to_select=k)
            self.X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = selector.transform(self.X_test_scaled)
            self.selected_features = selector.get_support(indices=True)
            
        elif method == 'from_model':
            # Use Random Forest for feature selection
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = SelectFromModel(rf, max_features=k)
            self.X_train_selected = selector.fit_transform(self.X_train_scaled, self.y_train)
            self.X_test_selected = selector.transform(self.X_test_scaled)
            self.selected_features = selector.get_support(indices=True)
        
        print(f"Selected {len(self.selected_features)} features")
        print(f"Selected feature indices: {self.selected_features}")
        
        return self.X_train_selected, self.X_test_selected
    
    def initialize_models(self):
        """Initialize baseline models"""
        print("\n=== Initializing Baseline Models ===")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM (Linear)': SVC(kernel='linear', random_state=42, probability=True),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} baseline models")
        return self.models
    
    def train_models(self, use_selected_features=True):
        """Train all baseline models"""
        print("\n=== Training Baseline Models ===")
        
        if use_selected_features and hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
            X_test = self.X_test_selected
        else:
            X_train = self.X_train_scaled
            X_test = self.X_test_scaled
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"Training {name}...")
            
            try:
                # Train model
                model.fit(X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                
                # ROC AUC (if probabilities available)
                roc_auc = None
                if y_pred_proba is not None:
                    roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                
                # Cross-validation score
                cv_scores = cross_val_score(model, X_train, self.y_train, cv=5, scoring='accuracy')
                
                # Store results
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"  - Accuracy: {accuracy:.4f}")
                print(f"  - F1-Score: {f1:.4f}")
                print(f"  - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"  - Error training {name}: {e}")
                self.results[name] = {'error': str(e)}
        
        return self.results
    
    def hyperparameter_tuning(self, model_name, param_grid, cv=5):
        """Perform hyperparameter tuning for a specific model"""
        print(f"\n=== Hyperparameter Tuning for {model_name} ===")
        
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        # Use selected features if available
        if hasattr(self, 'X_train_selected'):
            X_train = self.X_train_selected
        else:
            X_train = self.X_train_scaled
        
        # Grid search
        grid_search = GridSearchCV(
            self.models[model_name], 
            param_grid, 
            cv=cv, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, self.y_train)
        
        # Update model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_, grid_search.best_params_
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n=== Model Evaluation ===")
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                results_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1_score'],
                    'ROC-AUC': result['roc_auc'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Summary:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        results_df.to_csv('objective2_baseline_results.csv', index=False)
        print(f"\nResults saved to: objective2_baseline_results.csv")
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations of model results"""
        print("\n=== Creating Result Visualizations ===")
        
        # Create output directory
        os.makedirs('objective2_baseline_results', exist_ok=True)
        
        # Prepare data for visualization
        results_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                results_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'Precision': result['precision'],
                    'Recall': result['recall'],
                    'F1-Score': result['f1_score'],
                    'ROC-AUC': result['roc_auc'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        results_df = pd.DataFrame(results_data)
        
        # 1. Model comparison bar chart
        plt.figure(figsize=(15, 10))
        
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(results_df))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Baseline Model Performance Comparison')
        plt.xticks(x + width*2, results_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_baseline_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-validation scores
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['Model'], results_df['CV Mean'], yerr=results_df['CV Std'], 
                capsize=5, alpha=0.7, color='skyblue')
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Cross-Validation Performance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_baseline_results/cv_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrices for top 3 models
        top_models = results_df.head(3)['Model'].tolist()
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i, model_name in enumerate(top_models):
            if model_name in self.results and 'error' not in self.results[model_name]:
                cm = confusion_matrix(self.y_test, self.results[model_name]['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
                axes[i].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["accuracy"]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('objective2_baseline_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. ROC curves for models with probability predictions
        plt.figure(figsize=(10, 8))
        for name, result in self.results.items():
            if 'error' not in result and result['y_pred_proba'] is not None:
                fpr, tpr, _ = roc_curve(self.y_test, result['y_pred_proba'])
                auc = result['roc_auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Baseline Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_baseline_results/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to: objective2_baseline_results/")
    
    def analyze_feature_importance(self):
        """Analyze feature importance for tree-based models"""
        print("\n=== Analyzing Feature Importance ===")
        
        tree_models = ['Random Forest', 'Extra Trees', 'Gradient Boosting', 'XGBoost', 'LightGBM']
        
        for model_name in tree_models:
            if model_name in self.results and 'error' not in self.results[model_name]:
                model = self.results[model_name]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    
                    # Get feature names
                    if hasattr(self, 'selected_features'):
                        feature_names = [f'Feature_{i}' for i in self.selected_features]
                    else:
                        feature_names = [f'Feature_{i}' for i in range(len(importances))]
                    
                    # Store feature importance
                    self.feature_importance[model_name] = {
                        'importances': importances,
                        'feature_names': feature_names
                    }
                    
                    print(f"\n{model_name} - Top 10 Important Features:")
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    print(importance_df.head(10).to_string(index=False, float_format='%.4f'))
        
        return self.feature_importance
    
    def save_models(self, output_dir='objective2_baseline_models'):
        """Save trained models"""
        print(f"\n=== Saving Models to {output_dir} ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, result in self.results.items():
            if 'error' not in result:
                model_path = os.path.join(output_dir, f'{name.replace(" ", "_").lower()}.joblib')
                joblib.dump(result['model'], model_path)
                print(f"Saved {name} to {model_path}")
        
        # Save scaler
        if self.scaler is not None:
            scaler_path = os.path.join(output_dir, 'scaler.joblib')
            joblib.dump(self.scaler, scaler_path)
            print(f"Saved scaler to {scaler_path}")
        
        # Save feature selection info
        if hasattr(self, 'selected_features'):
            # Convert numpy arrays to lists for JSON serialization
            feature_importance_serializable = {}
            for model_name, importance_data in self.feature_importance.items():
                feature_importance_serializable[model_name] = {
                    'importances': importance_data['importances'].tolist(),
                    'feature_names': importance_data['feature_names']
                }
            
            feature_info = {
                'selected_features': self.selected_features.tolist(),
                'feature_importance': feature_importance_serializable
            }
            feature_path = os.path.join(output_dir, 'feature_info.json')
            with open(feature_path, 'w') as f:
                json.dump(feature_info, f, indent=2)
            print(f"Saved feature info to {feature_path}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== Generating Evaluation Report ===")
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for name, result in self.results.items():
            if 'error' not in result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
        
        # Generate report
        report = {
            'dataset_info': {
                'total_samples': len(self.X_train) + len(self.X_test),
                'train_samples': len(self.X_train),
                'test_samples': len(self.X_test),
                'features': self.X_train.shape[1],
                'selected_features': len(self.selected_features) if hasattr(self, 'selected_features') else self.X_train.shape[1]
            },
            'best_model': {
                'name': best_model,
                'accuracy': best_accuracy,
                'results': {
                    'accuracy': self.results[best_model]['accuracy'],
                    'precision': self.results[best_model]['precision'],
                    'recall': self.results[best_model]['recall'],
                    'f1_score': self.results[best_model]['f1_score'],
                    'roc_auc': self.results[best_model]['roc_auc'],
                    'cv_mean': self.results[best_model]['cv_mean'],
                    'cv_std': self.results[best_model]['cv_std']
                } if best_model else None
            },
            'model_performance': {
                name: {
                    'accuracy': result['accuracy'],
                    'f1_score': result['f1_score'],
                    'roc_auc': result['roc_auc']
                } for name, result in self.results.items() if 'error' not in result
            },
            'recommendations': [
                f"Best performing model: {best_model} with {best_accuracy:.4f} accuracy",
                "Consider ensemble methods for improved performance",
                "Feature engineering and selection can further improve results",
                "Deep learning models may provide better performance for this task"
            ]
        }
        
        # Save report
        with open('objective2_baseline_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: objective2_baseline_report.json")
        print(f"\nBest Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return report

def main():
    """Main function to run baseline ML models"""
    # Initialize baseline models
    baseline = BaselineMLModels()
    
    # Load and prepare data
    baseline.load_and_prepare_data()
    
    # Scale features
    baseline.scale_features(scaler_type='standard')
    
    # Feature selection
    baseline.feature_selection(method='selectkbest', k=30)
    
    # Initialize models
    baseline.initialize_models()
    
    # Train models
    baseline.train_models(use_selected_features=True)
    
    # Evaluate models
    results_df = baseline.evaluate_models()
    
    # Create visualizations
    baseline.visualize_results()
    
    # Analyze feature importance
    baseline.analyze_feature_importance()
    
    # Save models
    baseline.save_models()
    
    # Generate report
    baseline.generate_report()
    
    print("\n=== Baseline ML Models Complete ===")
    print("Next steps:")
    print("1. Review baseline model results")
    print("2. Proceed with deep learning models")
    print("3. Implement ensemble methods")

if __name__ == "__main__":
    main()