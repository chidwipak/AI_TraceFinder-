#!/usr/bin/env python3
"""
New Objective 2: Simplified Ensemble Models (Without TensorFlow)
Creates ensemble models using only scikit-learn for better compatibility
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import joblib

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedObjective2Ensemble:
    def __init__(self, baseline_path="new_objective2_baseline_ml_results", 
                 features_path="new_objective2_features",
                 output_path="new_objective2_ensemble_results"):
        self.baseline_path = baseline_path
        self.features_path = features_path
        self.output_path = output_path
        
        # Create output directories
        self.create_output_directories()
        
        # Data containers
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Models and results
        self.ensemble_models = {}
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
    
    def load_baseline_models_and_data(self):
        """Load baseline ML models and prepare data"""
        logger.info("="*80)
        logger.info("LOADING BASELINE MODELS AND DATA")
        logger.info("="*80)
        
        # Load baseline results
        baseline_results_file = f"{self.baseline_path}/results/all_results.csv"
        if os.path.exists(baseline_results_file):
            baseline_results_df = pd.read_csv(baseline_results_file)
            logger.info(f"Loaded baseline results: {len(baseline_results_df)} models")
            
            # Get top 5 baseline models
            top_baseline_models = baseline_results_df.head(5)
            logger.info("Top 5 baseline models:")
            for _, row in top_baseline_models.iterrows():
                logger.info(f"  {row['model_name']}: F1={row['f1_score']:.4f}, Acc={row['accuracy']:.4f}")
        
        # Load features data
        features_df = pd.read_csv(f"{self.features_path}/feature_vectors/comprehensive_features.csv")
        
        # Separate features and labels
        feature_columns = [col for col in features_df.columns 
                          if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]
        
        X_features = features_df[feature_columns].values
        y_features = features_df['label'].values
        
        # Handle missing values and scale
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X_features = imputer.fit_transform(X_features)
        
        scaler = StandardScaler()
        X_features = scaler.fit_transform(X_features)
        
        # Save scaler for later use
        joblib.dump(scaler, f"{self.output_path}/models/feature_scaler.pkl")
        
        # Load train/test splits
        train_df = pd.read_csv("new_objective2_preprocessed/metadata/train_split.csv")
        test_df = pd.read_csv("new_objective2_preprocessed/metadata/test_split.csv")
        
        # Split features into train and test
        train_indices = []
        test_indices = []
        
        for idx, row in features_df.iterrows():
            if row['image_path'] in train_df['image_path'].values:
                train_indices.append(idx)
            elif row['image_path'] in test_df['image_path'].values:
                test_indices.append(idx)
        
        self.X_train = X_features[train_indices]
        self.X_test = X_features[test_indices]
        self.y_train = y_features[train_indices]
        self.y_test = y_features[test_indices]
        
        logger.info(f"Feature data - Train: {self.X_train.shape}, Test: {self.X_test.shape}")
        logger.info(f"Labels - Train: {np.bincount(self.y_train)}, Test: {np.bincount(self.y_test)}")
        
        logger.info("Baseline models and data loaded!")
    
    def create_ensemble_models(self):
        """Create ensemble models from baseline ML models"""
        logger.info("="*80)
        logger.info("CREATING ENSEMBLE MODELS")
        logger.info("="*80)
        
        # Load class weights
        with open("new_objective2_preprocessed/metadata/class_weights.json", 'r') as f:
            class_weights = json.load(f)
        class_weight_dict = {int(k): v for k, v in class_weights.items()}
        
        # Create individual models for ensemble
        models_for_ensemble = [
            ('lr_balanced', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000)),
            ('lr_weighted', LogisticRegression(random_state=42, class_weight=class_weight_dict, max_iter=1000)),
            ('svm_linear', SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True)),
            ('svm_rbf', SVC(kernel='rbf', random_state=42, class_weight='balanced', probability=True)),
            ('rf_balanced', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')),
            ('rf_weighted', RandomForestClassifier(n_estimators=100, random_state=42, class_weight=class_weight_dict))
        ]
        
        # Apply SMOTE for balancing
        from imblearn.over_sampling import SMOTE
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_train_balanced, y_train_balanced = smote.fit_resample(self.X_train, self.y_train)
        
        # Train individual models
        trained_models = []
        for name, model in models_for_ensemble:
            try:
                model.fit(X_train_balanced, y_train_balanced)
                trained_models.append((name, model))
                logger.info(f"Trained {name}")
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")
        
        # Create ensemble models
        if len(trained_models) >= 3:
            # Voting Classifier (Hard)
            voting_hard = VotingClassifier(
                estimators=trained_models[:5],  # Top 5 models
                voting='hard'
            )
            voting_hard.fit(X_train_balanced, y_train_balanced)
            self.ensemble_models['Voting_Hard'] = voting_hard
            
            # Voting Classifier (Soft)
            voting_soft = VotingClassifier(
                estimators=trained_models[:5],  # Top 5 models
                voting='soft'
            )
            voting_soft.fit(X_train_balanced, y_train_balanced)
            self.ensemble_models['Voting_Soft'] = voting_soft
            
            # Stacking Classifier
            stacking = StackingClassifier(
                estimators=trained_models[:5],  # Top 5 models
                final_estimator=LogisticRegression(random_state=42),
                cv=5
            )
            stacking.fit(X_train_balanced, y_train_balanced)
            self.ensemble_models['Stacking'] = stacking
            
            # Advanced Stacking with different meta-learners
            stacking_rf = StackingClassifier(
                estimators=trained_models[:5],
                final_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
                cv=5
            )
            stacking_rf.fit(X_train_balanced, y_train_balanced)
            self.ensemble_models['Stacking_RF'] = stacking_rf
            
            logger.info(f"Created {len(self.ensemble_models)} ensemble models")
    
    def evaluate_ensemble_models(self):
        """Evaluate all ensemble models"""
        logger.info("="*80)
        logger.info("EVALUATING ENSEMBLE MODELS")
        logger.info("="*80)
        
        for model_name, model in self.ensemble_models.items():
            try:
                logger.info(f"Evaluating {model_name}...")
                
                # Standard ensemble model
                y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
                y_pred = model.predict(self.X_test)
                
                if y_pred_proba is None:
                    y_pred_proba = y_pred.astype(float)
                
                # Calculate metrics
                accuracy = accuracy_score(self.y_test, y_pred)
                precision = precision_score(self.y_test, y_pred)
                recall = recall_score(self.y_test, y_pred)
                f1 = f1_score(self.y_test, y_pred)
                auc = roc_auc_score(self.y_test, y_pred_proba) if y_pred_proba is not None else 0
                
                self.results[model_name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                logger.info(f"  {model_name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, AUC={auc:.4f}")
                
            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        logger.info(f"Evaluated {len(self.results)} ensemble models")
    
    def generate_comprehensive_report(self):
        """Generate comprehensive ensemble evaluation report"""
        logger.info("="*80)
        logger.info("GENERATING COMPREHENSIVE ENSEMBLE REPORT")
        logger.info("="*80)
        
        if not self.results:
            logger.error("No ensemble results to report")
            return
        
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
        results_df.to_csv(f"{self.output_path}/results/ensemble_results.csv", index=False)
        
        # Generate visualizations
        self._plot_ensemble_comparison(results_df)
        self._plot_model_performance(results_df)
        
        # Generate text report
        self._generate_text_report(results_df)
        
        # Save best model
        best_result = results_df.iloc[0]
        best_model_info = self.results[best_result['model_name']]
        
        # Save best ensemble model
        joblib.dump(best_model_info['model'], f"{self.output_path}/models/best_ensemble_model.pkl")
        
        # Also save the meta-learner for hybrid approach
        if best_result['model_name'].startswith('Stacking'):
            meta_learner = best_model_info['model'].final_estimator_
            joblib.dump(meta_learner, f"{self.output_path}/models/best_ensemble_meta_learner.pkl")
        
        logger.info("Comprehensive ensemble report generated!")
    
    def _plot_ensemble_comparison(self, results_df):
        """Plot ensemble comparison"""
        plt.figure(figsize=(15, 10))
        
        # Performance comparison
        plt.subplot(2, 2, 1)
        plt.barh(range(len(results_df)), results_df['f1_score'])
        plt.yticks(range(len(results_df)), results_df['model_name'])
        plt.xlabel('F1 Score')
        plt.title('Ensemble Models - F1 Score Comparison')
        
        plt.subplot(2, 2, 2)
        plt.barh(range(len(results_df)), results_df['accuracy'])
        plt.yticks(range(len(results_df)), results_df['model_name'])
        plt.xlabel('Accuracy')
        plt.title('Ensemble Models - Accuracy Comparison')
        
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
        plt.title('Ensemble Models - AUC Comparison')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/ensemble_comparison.png", 
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
            plt.title(f'Ensemble Models - {metric.replace("_", " ").title()}')
        
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
        plt.savefig(f"{self.output_path}/visualizations/ensemble_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, results_df):
        """Generate comprehensive text report"""
        report_path = f"{self.output_path}/reports/ensemble_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("SIMPLIFIED OBJECTIVE 2 ENSEMBLE MODELS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Training samples: {len(self.X_train)}\n")
            f.write(f"Test samples: {len(self.X_test)}\n")
            f.write(f"Training class distribution: {np.bincount(self.y_train)}\n")
            f.write(f"Test class distribution: {np.bincount(self.y_test)}\n\n")
            
            # Ensemble Results
            f.write("ENSEMBLE MODEL RESULTS\n")
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
            f.write("BEST ENSEMBLE MODEL ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Best Model: {best_result['model_name']}\n")
            f.write(f"Best F1 Score: {best_result['f1_score']:.4f}\n")
            f.write(f"Best Accuracy: {best_result['accuracy']:.4f}\n")
            f.write(f"Best AUC: {best_result['auc']:.4f}\n\n")
            
            # Target Analysis
            target_accuracy = 0.90
            models_above_target = results_df[results_df['accuracy'] >= target_accuracy]
            
            f.write(f"ENSEMBLE MODELS ACHIEVING {target_accuracy*100}%+ ACCURACY\n")
            f.write("-"*40 + "\n")
            if len(models_above_target) > 0:
                f.write(f"🎉 SUCCESS! Found {len(models_above_target)} ensemble models achieving {target_accuracy*100}%+ accuracy:\n")
                for _, row in models_above_target.iterrows():
                    f.write(f"  ✅ {row['model_name']}: {row['accuracy']:.4f} ({row['accuracy']*100:.2f}%)\n")
                f.write(f"\n🎯 TARGET ACHIEVED: {target_accuracy*100}%+ accuracy reached with ensemble methods!\n")
            else:
                f.write(f"No ensemble models achieved {target_accuracy*100}%+ accuracy\n")
                f.write(f"Best accuracy: {best_result['accuracy']:.4f} ({best_result['accuracy']*100:.2f}%)\n")
                f.write(f"Gap to target: {(target_accuracy - best_result['accuracy'])*100:.2f} percentage points\n")
            
            f.write("\n")
            
            # Final Verdict
            f.write("FINAL VERDICT\n")
            f.write("-"*40 + "\n")
            if len(models_above_target) > 0:
                f.write("✅ OBJECTIVE ACHIEVED: Successfully achieved 90%+ accuracy using ensemble methods!\n")
            else:
                f.write("⚠️  OBJECTIVE NOT FULLY ACHIEVED: 90%+ accuracy not reached.\n")
                f.write("However, significant progress was made with ensemble methods.\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_ensemble_pipeline(self):
        """Run complete ensemble pipeline"""
        logger.info("🚀 STARTING SIMPLIFIED OBJECTIVE 2 ENSEMBLE PIPELINE")
        logger.info("="*80)
        
        # Step 1: Load baseline models and data
        self.load_baseline_models_and_data()
        
        # Step 2: Create ensemble models
        self.create_ensemble_models()
        
        # Step 3: Evaluate ensemble models
        self.evaluate_ensemble_models()
        
        # Step 4: Generate comprehensive report
        self.generate_comprehensive_report()
        
        logger.info("✅ SIMPLIFIED ENSEMBLE PIPELINE COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        
        return self.results

def main():
    """Main function to run simplified ensemble pipeline"""
    ensemble = SimplifiedObjective2Ensemble()
    results = ensemble.run_complete_ensemble_pipeline()
    
    print("\n" + "="*80)
    print("SIMPLIFIED ENSEMBLE PIPELINE SUMMARY")
    print("="*80)
    
    # Find best result
    if results:
        all_results = [(name, result['f1_score'], result['accuracy']) for name, result in results.items()]
        best_result = max(all_results, key=lambda x: x[1])  # Sort by F1 score
        
        print(f"Best Ensemble Model: {best_result[0]}")
        print(f"Best F1 Score: {best_result[1]:.4f}")
        print(f"Best Accuracy: {best_result[2]:.4f}")
        
        target_accuracy = 0.90
        if best_result[2] >= target_accuracy:
            print(f"🎉 TARGET ACHIEVED: {target_accuracy*100}%+ accuracy reached!")
            print("✅ Objective 2 successfully completed with 90%+ accuracy!")
        else:
            print(f"📈 Progress: {best_result[2]*100:.2f}% accuracy (need {target_accuracy*100:.1f}%)")
            print(f"Gap to target: {(target_accuracy - best_result[2])*100:.2f} percentage points")
    
    print("\nNext step: Integrate with Streamlit app!")

if __name__ == "__main__":
    main()
