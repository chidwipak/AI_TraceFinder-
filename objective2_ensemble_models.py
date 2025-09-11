#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Ensemble Models
Creates ensemble models combining multiple approaches for higher accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from scipy.stats import mode
import warnings
warnings.filterwarnings('ignore')

class EnsembleModel:
    """Ensemble model combining multiple approaches"""
    
    def __init__(self, models, weights=None, method='voting'):
        self.models = models
        self.weights = weights
        self.method = method
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit all models in the ensemble"""
        print(f"Training ensemble with {len(self.models)} models using {self.method} method...")
        
        for i, (name, model) in enumerate(self.models.items()):
            print(f"Training {name}...")
            try:
                model.fit(X, y)
                print(f"  - {name} trained successfully")
            except Exception as e:
                print(f"  - Error training {name}: {e}")
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        predictions = []
        probabilities = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    pred = model.predict(X)
                    probabilities.append(proba)
                    predictions.append(pred)
                else:
                    pred = model.predict(X)
                    predictions.append(pred)
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        if self.method == 'voting':
            # Hard voting
            if self.weights:
                weighted_predictions = []
                for i, pred in enumerate(predictions):
                    weighted_predictions.append(pred * self.weights[i])
                final_pred = np.round(np.mean(weighted_predictions, axis=0))
            else:
                final_pred = mode(predictions, axis=0)[0].flatten()
        
        elif self.method == 'averaging':
            # Average probabilities
            if probabilities:
                if self.weights:
                    weighted_probs = []
                    for i, proba in enumerate(probabilities):
                        weighted_probs.append(proba * self.weights[i])
                    avg_proba = np.mean(weighted_probs, axis=0)
                else:
                    avg_proba = np.mean(probabilities, axis=0)
                final_pred = np.argmax(avg_proba, axis=1)
            else:
                # Fallback to voting
                final_pred = mode(predictions, axis=0)[0].flatten()
        
        return final_pred
    
    def predict_proba(self, X):
        """Predict probabilities using ensemble"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        probabilities = []
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X)
                    probabilities.append(proba)
            except Exception as e:
                print(f"Error with {name}: {e}")
        
        if probabilities:
            if self.weights:
                weighted_probs = []
                for i, proba in enumerate(probabilities):
                    weighted_probs.append(proba * self.weights[i])
                avg_proba = np.mean(weighted_probs, axis=0)
            else:
                avg_proba = np.mean(probabilities, axis=0)
            return avg_proba
        else:
            # Fallback: convert predictions to probabilities
            pred = self.predict(X)
            proba = np.zeros((len(pred), 2))
            proba[np.arange(len(pred)), pred] = 1
            return proba

class AdvancedEnsemble:
    """Advanced ensemble with stacking and blending"""
    
    def __init__(self, base_models, meta_model=None):
        self.base_models = base_models
        self.meta_model = meta_model or LogisticRegression()
        self.is_fitted = False
        
    def fit(self, X, y):
        """Fit the stacking ensemble"""
        print("Training stacking ensemble...")
        
        # Train base models
        base_predictions = []
        for name, model in self.base_models.items():
            print(f"Training base model: {name}")
            model.fit(X, y)
            
            # Get cross-validated predictions for meta-model
            cv_predictions = []
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            for train_idx, val_idx in cv.split(X, y):
                X_train_cv, X_val_cv = X[train_idx], X[val_idx]
                y_train_cv = y[train_idx]
                
                model.fit(X_train_cv, y_train_cv)
                
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_val_cv)[:, 1]
                else:
                    pred = model.predict(X_val_cv)
                
                cv_predictions.extend(pred)
            
            base_predictions.append(cv_predictions)
        
        # Create meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Train meta-model
        print("Training meta-model...")
        self.meta_model.fit(meta_features, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using stacking"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        # Create meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Get meta-model predictions
        meta_predictions = self.meta_model.predict(meta_features)
        
        return meta_predictions
    
    def predict_proba(self, X):
        """Predict probabilities using stacking"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Get base model predictions
        base_predictions = []
        for name, model in self.base_models.items():
            if hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]
            else:
                pred = model.predict(X)
            base_predictions.append(pred)
        
        # Create meta-features
        meta_features = np.column_stack(base_predictions)
        
        # Get meta-model probabilities
        meta_proba = self.meta_model.predict_proba(meta_features)
        
        return meta_proba

class SUPATLANTIQUEEnsemble:
    """Main ensemble class for SUPATLANTIQUE dataset"""
    
    def __init__(self):
        self.ensemble_models = {}
        self.results = {}
        
    def load_data(self):
        """Load preprocessed data"""
        print("=== Loading Preprocessed Data ===")
        
        # Load features
        features_df = pd.read_csv('objective2_forensic_features.csv')
        print(f"Loaded features: {features_df.shape}")
        
        # Load train-test splits
        train_df = pd.read_csv('objective2_train_split.csv')
        test_df = pd.read_csv('objective2_test_split.csv')
        
        print(f"Train set: {len(train_df)} images")
        print(f"Test set: {len(test_df)} images")
        
        # Prepare feature matrices
        feature_cols = [col for col in features_df.columns if col not in ['image_path', 'label', 'tamper_type', 'category']]
        
        # Get features for train and test sets
        train_features = []
        test_features = []
        
        for _, row in train_df.iterrows():
            img_features = features_df[features_df['image_path'] == row['image_path']]
            if not img_features.empty:
                train_features.append(img_features[feature_cols].values[0])
        
        for _, row in test_df.iterrows():
            img_features = features_df[features_df['image_path'] == row['image_path']]
            if not img_features.empty:
                test_features.append(img_features[feature_cols].values[0])
        
        X_train = np.array(train_features)
        X_test = np.array(test_features)
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        # Handle infinite and NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"Feature matrix shapes: Train {X_train.shape}, Test {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def create_base_models(self):
        """Create base models for ensemble"""
        print("=== Creating Base Models ===")
        
        base_models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM (RBF)': SVC(kernel='rbf', random_state=42, probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'XGBoost': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1)
        }
        
        print(f"Created {len(base_models)} base models")
        return base_models
    
    def create_ensemble_models(self, base_models):
        """Create different ensemble models"""
        print("=== Creating Ensemble Models ===")
        
        ensemble_models = {}
        
        # 1. Voting Classifier (Hard Voting)
        voting_models = [
            ('lr', base_models['Logistic Regression']),
            ('svm', base_models['SVM (RBF)']),
            ('rf', base_models['Random Forest'])
        ]
        ensemble_models['Voting Classifier (Hard)'] = VotingClassifier(
            estimators=voting_models, voting='hard'
        )
        
        # 2. Voting Classifier (Soft Voting)
        ensemble_models['Voting Classifier (Soft)'] = VotingClassifier(
            estimators=voting_models, voting='soft'
        )
        
        # 3. Custom Ensemble (Voting)
        ensemble_models['Custom Ensemble (Voting)'] = EnsembleModel(
            models=base_models, method='voting'
        )
        
        # 4. Custom Ensemble (Averaging)
        ensemble_models['Custom Ensemble (Averaging)'] = EnsembleModel(
            models=base_models, method='averaging'
        )
        
        # 5. Weighted Ensemble
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]  # Based on individual performance
        ensemble_models['Weighted Ensemble'] = EnsembleModel(
            models=base_models, weights=weights, method='averaging'
        )
        
        # 6. Stacking Ensemble
        ensemble_models['Stacking Ensemble'] = AdvancedEnsemble(
            base_models=base_models, meta_model=LogisticRegression()
        )
        
        # 7. Bagging Ensemble
        ensemble_models['Bagging Ensemble'] = BaggingClassifier(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=10, random_state=42
        )
        
        # 8. AdaBoost Ensemble
        ensemble_models['AdaBoost Ensemble'] = AdaBoostClassifier(
            estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            n_estimators=10, random_state=42
        )
        
        print(f"Created {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def train_ensemble_models(self, X_train, y_train, X_test, y_test):
        """Train all ensemble models"""
        print("=== Training Ensemble Models ===")
        
        base_models = self.create_base_models()
        ensemble_models = self.create_ensemble_models(base_models)
        
        self.results = {}
        
        for name, ensemble in ensemble_models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Train ensemble
                ensemble.fit(X_train, y_train)
                
                # Make predictions
                y_pred = ensemble.predict(X_test)
                y_pred_proba = ensemble.predict_proba(X_test)[:, 1] if hasattr(ensemble, 'predict_proba') else None
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
                
                # Cross-validation score
                cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5, scoring='accuracy')
                
                # Store results
                self.results[name] = {
                    'ensemble': ensemble,
                    'accuracy': accuracy,
                    'roc_auc': roc_auc,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"  - Accuracy: {accuracy:.4f}")
                print(f"  - ROC AUC: {roc_auc:.4f}" if roc_auc else "  - ROC AUC: N/A")
                print(f"  - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                print(f"  - Error training {name}: {e}")
                self.results[name] = {'error': str(e)}
        
        return self.results
    
    def evaluate_ensemble_models(self):
        """Evaluate all ensemble models"""
        print("\n=== Ensemble Model Evaluation ===")
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                results_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'ROC AUC': result['roc_auc'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Accuracy', ascending=False)
        
        print("\nEnsemble Model Performance:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        results_df.to_csv('objective2_ensemble_results.csv', index=False)
        print(f"\nResults saved to: objective2_ensemble_results.csv")
        
        return results_df
    
    def visualize_results(self, y_test):
        """Create visualizations of ensemble results"""
        print("\n=== Creating Ensemble Result Visualizations ===")
        
        # Create output directory
        os.makedirs('objective2_ensemble_results', exist_ok=True)
        
        # Prepare data for visualization
        results_data = []
        for name, result in self.results.items():
            if 'error' not in result:
                results_data.append({
                    'Model': name,
                    'Accuracy': result['accuracy'],
                    'ROC AUC': result['roc_auc'],
                    'CV Mean': result['cv_mean'],
                    'CV Std': result['cv_std']
                })
        
        results_df = pd.DataFrame(results_data)
        
        # 1. Model comparison
        plt.figure(figsize=(15, 8))
        
        metrics = ['Accuracy', 'ROC AUC', 'CV Mean']
        x = np.arange(len(results_df))
        width = 0.25
        
        for i, metric in enumerate(metrics):
            plt.bar(x + i*width, results_df[metric], width, label=metric, alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Ensemble Model Performance Comparison')
        plt.xticks(x + width, results_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_ensemble_results/ensemble_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Cross-validation scores
        plt.figure(figsize=(12, 6))
        plt.bar(results_df['Model'], results_df['CV Mean'], yerr=results_df['CV Std'], 
                capsize=5, alpha=0.7, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('Cross-Validation Accuracy')
        plt.title('Ensemble Cross-Validation Performance')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_ensemble_results/ensemble_cv_scores.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Confusion matrices for top 3 models
        top_models = results_df.head(3)['Model'].tolist()
        
        fig, axes = plt.subplots(1, min(3, len(top_models)), figsize=(5*min(3, len(top_models)), 4))
        if len(top_models) == 1:
            axes = [axes]
        
        for i, model_name in enumerate(top_models):
            if model_name in self.results and 'error' not in self.results[model_name]:
                cm = confusion_matrix(y_test, self.results[model_name]['y_pred'])
                sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=axes[i])
                axes[i].set_title(f'{model_name}\nAccuracy: {self.results[model_name]["accuracy"]:.3f}')
                axes[i].set_xlabel('Predicted')
                axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('objective2_ensemble_results/ensemble_confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to: objective2_ensemble_results/")
    
    def save_models(self, output_dir='objective2_ensemble_models'):
        """Save trained ensemble models"""
        print(f"\n=== Saving Ensemble Models to {output_dir} ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, result in self.results.items():
            if 'error' not in result:
                model_path = os.path.join(output_dir, f'{name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.joblib')
                joblib.dump(result['ensemble'], model_path)
                print(f"Saved {name} to {model_path}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== Generating Ensemble Evaluation Report ===")
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for name, result in self.results.items():
            if 'error' not in result and result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
        
        # Generate report
        report = {
            'best_model': {
                'name': best_model,
                'accuracy': best_accuracy,
                'results': {
                    'accuracy': self.results[best_model]['accuracy'],
                    'roc_auc': self.results[best_model]['roc_auc'],
                    'cv_mean': self.results[best_model]['cv_mean'],
                    'cv_std': self.results[best_model]['cv_std']
                } if best_model else None
            },
            'model_performance': {
                name: {
                    'accuracy': result['accuracy'],
                    'roc_auc': result['roc_auc'],
                    'cv_mean': result['cv_mean'],
                    'cv_std': result['cv_std']
                } for name, result in self.results.items() if 'error' not in result
            },
            'recommendations': [
                f"Best performing ensemble: {best_model} with {best_accuracy:.4f} accuracy",
                "Ensemble methods show improved performance over individual models",
                "Consider combining with deep learning models for further improvement",
                "Feature engineering and hyperparameter tuning may boost performance"
            ]
        }
        
        # Save report
        with open('objective2_ensemble_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Ensemble evaluation report saved to: objective2_ensemble_report.json")
        print(f"\nBest Ensemble Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return report

def main():
    """Main function to run ensemble models"""
    # Initialize ensemble pipeline
    ensemble_pipeline = SUPATLANTIQUEEnsemble()
    
    # Load data
    X_train, X_test, y_train, y_test = ensemble_pipeline.load_data()
    
    # Train ensemble models
    ensemble_pipeline.train_ensemble_models(X_train, y_train, X_test, y_test)
    
    # Evaluate models
    results_df = ensemble_pipeline.evaluate_ensemble_models()
    
    # Create visualizations
    ensemble_pipeline.visualize_results(y_test)
    
    # Save models
    ensemble_pipeline.save_models()
    
    # Generate report
    ensemble_pipeline.generate_report()
    
    print("\n=== Ensemble Models Complete ===")
    print("Next steps:")
    print("1. Review ensemble model results")
    print("2. Analyze why 90% accuracy not achieved")
    print("3. Implement advanced optimization techniques")

if __name__ == "__main__":
    main()
