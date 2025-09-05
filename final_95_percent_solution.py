#!/usr/bin/env python3
"""
FINAL 95%+ ACCURACY SOLUTION - IMPROVED VERSION
Building on our 91.03% GradientBoosting result to push over 95%
Enhanced with better monitoring, resource management, and cleanup
"""

import os
import sys
import psutil
import signal
import atexit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold, SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
import xgboost as xgb
import lightgbm as lgb
import pickle
import logging
import time
from datetime import datetime
import warnings
import gc
import joblib
from contextlib import contextmanager

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

@contextmanager
def safe_parallel_context(n_jobs=1):
    """Context manager for safe parallel processing with cleanup"""
    try:
        yield
    finally:
        # Force cleanup of any remaining workers
        try:
            gc.collect()
            cleanup_workers()
        except:
            pass

def cleanup_workers():
    """Clean up any orphaned joblib workers"""
    try:
        current_process = psutil.Process()
        children = current_process.children(recursive=True)
        for child in children:
            try:
                if 'LokyProcess' in child.name() or 'joblib' in ' '.join(child.cmdline()):
                    logger.info(f"Terminating worker process: {child.pid}")
                    child.terminate()
                    child.wait(timeout=3)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
                pass
    except Exception as e:
        logger.warning(f"Error during worker cleanup: {e}")

def check_system_resources():
    """Check system resources and warn if overloaded"""
    load_avg = os.getloadavg()[0]
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    logger.info(f"System Load: {load_avg:.2f} (CPUs: {cpu_count})")
    logger.info(f"Memory Usage: {memory.percent:.1f}% ({memory.available/1024**3:.1f}GB available)")
    
    if load_avg > cpu_count * 2:
        logger.warning(f"HIGH SYSTEM LOAD: {load_avg:.2f} (recommend < {cpu_count * 2})")
        return False
    
    if memory.percent > 85:
        logger.warning(f"HIGH MEMORY USAGE: {memory.percent:.1f}%")
        return False
    
    return True

class Final95PercentSolution:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "final_95_results_monitored"
        self.start_time = datetime.now()
        self.checkpoint_interval = 300  # Save progress every 5 minutes
        
        # Register cleanup on exit
        atexit.register(cleanup_workers)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        
        # Create output directories
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        os.makedirs(f"{self.output_path}/checkpoints", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        
        # Resource monitoring
        self.resource_check_interval = 60  # Check every minute
        self.last_resource_check = time.time()
    
    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully"""
        logger.info(f"Received signal {signum}, cleaning up...")
        cleanup_workers()
        sys.exit(0)
    
    def _monitor_resources(self):
        """Monitor system resources during execution"""
        if time.time() - self.last_resource_check > self.resource_check_interval:
            if not check_system_resources():
                logger.warning("Resource constraints detected, consider pausing...")
            self.last_resource_check = time.time()
    
    def _save_checkpoint(self, data, name):
        """Save checkpoint data"""
        checkpoint_path = f"{self.output_path}/checkpoints/{name}_{int(time.time())}.pkl"
        try:
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
        
    def load_and_prepare_ultimate_features(self):
        """Load all features and prepare for 95%+ accuracy"""
        logger.info("="*80)
        logger.info("LOADING ULTIMATE FEATURES FOR 95%+ ACCURACY")
        logger.info("="*80)
        
        # Load our comprehensive features
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Fix split issue by creating proper stratified split
        df['split'] = 'unknown'
        
        # Create stratified split
        from sklearn.model_selection import train_test_split
        train_idx, test_idx = train_test_split(
            range(len(df)), 
            test_size=0.15, 
            random_state=self.random_seed,
            stratify=df['scanner_id']
        )
        
        df.iloc[train_idx, df.columns.get_loc('split')] = 'train'
        df.iloc[test_idx, df.columns.get_loc('split')] = 'test'
        
        split_counts = df['split'].value_counts()
        logger.info(f"Stratified split: {split_counts}")
        
        return df
    
    def advanced_feature_engineering(self, df):
        """Advanced feature engineering for maximum accuracy"""
        logger.info("Performing advanced feature engineering...")
        
        # Separate features from metadata
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Advanced missing value handling
        logger.info("Handling missing values with KNN imputation...")
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        
        # Remove extremely low variance features
        logger.info("Removing low variance features...")
        variance_selector = VarianceThreshold(threshold=0.001)
        X = variance_selector.fit_transform(X)
        
        # Split data
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        # Advanced feature selection - use multiple methods
        logger.info("Advanced feature selection...")
        
        # 1. Select top features using mutual information
        selector_mi = SelectKBest(mutual_info_classif, k=min(120, X_train.shape[1]))
        X_train_mi = selector_mi.fit_transform(X_train, y_train_encoded)
        X_test_mi = selector_mi.transform(X_test)
        
        # 2. Use RandomForest feature importance
        rf_selector = RandomForestClassifier(n_estimators=200, random_state=self.random_seed, n_jobs=-1)
        rf_selector.fit(X_train_mi, y_train_encoded)
        
        # Get feature importances and select top features
        importances = rf_selector.feature_importances_
        top_indices = np.argsort(importances)[-100:]  # Top 100 features
        
        X_train_selected = X_train_mi[:, top_indices]
        X_test_selected = X_test_mi[:, top_indices]
        
        # Scale features
        logger.info("Scaling features with RobustScaler...")
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        logger.info(f"Final feature dimensions: {X_train_scaled.shape[1]}")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def create_ultimate_models(self):
        """Create ultimate models optimized for 95%+ accuracy"""
        logger.info("Creating ultimate models...")
        
        models = {}
        
        # 1. Super-tuned Gradient Boosting (our best so far at 91.03%)
        models['SuperGradientBoosting'] = GradientBoostingClassifier(
            n_estimators=1000,
            max_depth=15,
            learning_rate=0.05,
            subsample=0.9,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed
        )
        
        # 2. Advanced XGBoost
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=1000,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # 3. LightGBM
        models['LightGBM'] = lgb.LGBMClassifier(
            n_estimators=1000,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=-1
        )
        
        # 4. Extra Trees (highly randomized)
        models['ExtraTrees'] = ExtraTreesClassifier(
            n_estimators=1500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # 5. Advanced Random Forest
        models['SuperRandomForest'] = RandomForestClassifier(
            n_estimators=1500,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            oob_score=True,
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        return models
    
    def hyperparameter_optimization(self, models, X_train, y_train):
        """Optimize hyperparameters for best models"""
        logger.info("Optimizing hyperparameters...")
        
        optimized_models = {}
        
        # Optimize the best performing models
        # 1. Super-tune Gradient Boosting
        logger.info("Super-tuning Gradient Boosting...")
        gb_param_grid = {
            'n_estimators': [800, 1000, 1200],
            'max_depth': [12, 15, 18],
            'learning_rate': [0.03, 0.05, 0.07],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gb_grid = GridSearchCV(
            GradientBoostingClassifier(random_state=self.random_seed),
            gb_param_grid,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        gb_grid.fit(X_train, y_train)
        optimized_models['SuperGradientBoosting'] = gb_grid.best_estimator_
        logger.info(f"Best GB params: {gb_grid.best_params_}")
        logger.info(f"Best GB CV score: {gb_grid.best_score_:.4f}")
        
        # Use the other models as-is (already well-tuned)
        optimized_models['XGBoost'] = models['XGBoost']
        optimized_models['LightGBM'] = models['LightGBM']
        optimized_models['ExtraTrees'] = models['ExtraTrees']
        optimized_models['SuperRandomForest'] = models['SuperRandomForest']
        
        return optimized_models
    
    def create_ultimate_ensemble(self, models):
        """Create ultimate ensemble for maximum accuracy"""
        logger.info("Creating ultimate ensemble...")
        
        # Select best models for ensemble
        ensemble_models = [
            ('sgb', models['SuperGradientBoosting']),
            ('xgb', models['XGBoost']),
            ('lgb', models['LightGBM']),
            ('et', models['ExtraTrees']),
            ('srf', models['SuperRandomForest'])
        ]
        
        ensemble = VotingClassifier(
            estimators=ensemble_models,
            voting='soft',  # Use probabilities for better ensemble
            n_jobs=-1
        )
        
        return ensemble
    
    def safe_model_evaluation(self, name, model, X_train, X_test, y_train, y_test):
        """Safely evaluate a single model with monitoring and cleanup"""
        logger.info(f"="*60)
        logger.info(f"EVALUATING: {name}")
        logger.info(f"="*60)
        
        start_time = time.time()
        self._monitor_resources()
        
        try:
            # Set conservative n_jobs to avoid overwhelming the system
            if hasattr(model, 'n_jobs'):
                original_n_jobs = model.n_jobs
                # Limit to 4 cores maximum during high load
                load_avg = os.getloadavg()[0]
                cpu_count = psutil.cpu_count()
                if load_avg > cpu_count:
                    model.n_jobs = min(4, cpu_count // 4) 
                    logger.info(f"Reduced n_jobs to {model.n_jobs} due to high system load")
            
            with safe_parallel_context():
                logger.info(f"Training {name}...")
                
                # Train model with timeout safety
                model.fit(X_train, y_train)
                
                logger.info(f"Making predictions with {name}...")
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                
                # Calculate basic metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                logger.info(f"Running cross-validation for {name}...")
                # Reduced CV for faster execution during high load
                cv_splits = 5 if load_avg > cpu_count else 10
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_seed),
                    scoring='accuracy',
                    n_jobs=min(2, psutil.cpu_count() // 8)  # Very conservative
                )
                
                elapsed = time.time() - start_time
                
                result = {
                    'test_accuracy': accuracy,
                    'test_precision': precision,
                    'test_recall': recall,
                    'test_f1': f1,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'y_pred': y_pred,
                    'model': model,
                    'training_time': elapsed
                }
                
                # Save checkpoint for this model
                self._save_checkpoint(result, f"model_{name.lower()}")
                
                logger.info(f"✅ {name} COMPLETED:")
                logger.info(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                logger.info(f"  CV Accuracy: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
                logger.info(f"  Training Time: {elapsed:.1f}s")
                
                # Check if we hit 95%
                if accuracy >= 0.95:
                    logger.info(f"🎉 {name} ACHIEVED 95%+ ACCURACY! 🎉")
                
                # Force cleanup after each model
                gc.collect()
                
                return result
                
        except Exception as e:
            logger.error(f"❌ Error evaluating {name}: {e}")
            return {
                'test_accuracy': 0.0,
                'test_precision': 0.0,
                'test_recall': 0.0,
                'test_f1': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'y_pred': None,
                'model': None,
                'training_time': time.time() - start_time,
                'error': str(e)
            }
        finally:
            # Restore original n_jobs
            if hasattr(model, 'n_jobs') and 'original_n_jobs' in locals():
                model.n_jobs = original_n_jobs
    
    def comprehensive_evaluation(self, models, X_train, X_test, y_train, y_test):
        """Comprehensive evaluation targeting 95%+ with safe execution"""
        logger.info("="*80)
        logger.info("COMPREHENSIVE EVALUATION FOR 95%+ ACCURACY")
        logger.info("Enhanced with resource monitoring and cleanup")
        logger.info("="*80)
        
        # Check system resources before starting
        if not check_system_resources():
            logger.warning("System under heavy load - proceeding with caution...")
        
        results = {}
        
        # Evaluate models one by one to avoid resource conflicts
        for i, (name, model) in enumerate(models.items(), 1):
            logger.info(f"\n📊 PROGRESS: Model {i}/{len(models)}")
            
            # Monitor resources before each model
            self._monitor_resources()
            
            # Evaluate single model safely
            result = self.safe_model_evaluation(name, model, X_train, X_test, y_train, y_test)
            results[name] = result
            
            # Give system a brief rest between models
            time.sleep(2)
            
            # Check if we should continue based on resources
            if not check_system_resources():
                logger.warning("High resource usage detected - consider stopping here")
                response = input("Continue with remaining models? (y/n): ").lower()
                if response != 'y':
                    logger.info("Stopping evaluation due to resource constraints")
                    break
        
        logger.info("\n" + "="*80)
        logger.info("EVALUATION SUMMARY")
        logger.info("="*80)
        
        # Sort results by accuracy
        sorted_results = sorted(
            [(name, result) for name, result in results.items() if 'error' not in result],
            key=lambda x: x[1]['test_accuracy'],
            reverse=True
        )
        
        for name, result in sorted_results:
            acc = result['test_accuracy']
            logger.info(f"{name}: {acc:.4f} ({acc*100:.2f}%)")
        
        return results
    
    def create_final_visualizations(self, results):
        """Create comprehensive visualizations"""
        logger.info("Creating final visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(results.keys())
        accuracies = [results[m]['test_accuracy'] for m in models]
        f1_scores = [results[m]['test_f1'] for m in models]
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]
        
        # Test accuracy
        bars1 = axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.8)
        axes[0, 0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 0].set_title('Test Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Highlight bars above 95%
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            if acc >= 0.95:
                bar.set_color('gold')
                bar.set_alpha(1.0)
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation accuracy
        bars2 = axes[0, 1].bar(models, cv_means, yerr=cv_stds, capsize=5, color='lightgreen', alpha=0.8)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 1].set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Best model confusion matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
        best_y_pred = results[best_model_name]['y_pred']
        best_y_test = [i for i in range(len(best_y_pred))]  # Simplified for visualization
        
        # Feature importance (if available)
        if hasattr(results[best_model_name]['model'], 'feature_importances_'):
            importances = results[best_model_name]['model'].feature_importances_
            top_10_indices = np.argsort(importances)[-10:]
            
            axes[1, 0].barh(range(10), importances[top_10_indices], color='orange', alpha=0.8)
            axes[1, 0].set_title(f'{best_model_name} - Top 10 Feature Importances')
            axes[1, 0].set_xlabel('Importance')
            axes[1, 0].set_yticks(range(10))
            axes[1, 0].set_yticklabels([f'Feature {i}' for i in top_10_indices])
        
        # Summary statistics
        axes[1, 1].axis('off')
        best_acc = max(accuracies)
        summary_text = f"""
FINAL RESULTS SUMMARY

Best Model: {best_model_name}
Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)

Target (95%): {'✅ ACHIEVED!' if best_acc >= 0.95 else '❌ Not reached'}

Models above 90%:
"""
        for name, acc in zip(models, accuracies):
            if acc >= 0.90:
                summary_text += f"• {name}: {acc:.3f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/final_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_final_results(self, results):
        """Save final results and models"""
        logger.info("Saving final results...")
        
        # Save all models
        for name, result in results.items():
            model_path = f"{self.output_path}/models/{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Create results summary
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Test_Accuracy': result['test_accuracy'],
                'Test_Precision': result['test_precision'],
                'Test_Recall': result['test_recall'],
                'Test_F1': result['test_f1'],
                'CV_Mean': result['cv_mean'],
                'CV_Std': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        summary_df.to_csv(f"{self.output_path}/results/final_summary.csv", index=False)
        
        return summary_df
    
    def run_final_95_pipeline(self):
        """Run the final pipeline to achieve 95%+ accuracy"""
        logger.info("="*80)
        logger.info("🎯 FINAL PUSH FOR 95%+ ACCURACY")
        logger.info("Building on 91.03% GradientBoosting success!")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Load and prepare ultimate features
        df = self.load_and_prepare_ultimate_features()
        if df is None:
            return
        
        # 2. Advanced feature engineering
        X_train, X_test, y_train, y_test = self.advanced_feature_engineering(df)
        
        # 3. Create ultimate models
        models = self.create_ultimate_models()
        
        # 4. Hyperparameter optimization
        optimized_models = self.hyperparameter_optimization(models, X_train, y_train)
        
        # 5. Create ultimate ensemble
        ensemble = self.create_ultimate_ensemble(optimized_models)
        optimized_models['UltimateEnsemble'] = ensemble
        
        # 6. Comprehensive evaluation
        results = self.comprehensive_evaluation(optimized_models, X_train, X_test, y_train, y_test)
        
        # 7. Create visualizations
        self.create_final_visualizations(results)
        
        # 8. Save results
        summary_df = self.save_final_results(results)
        
        # 9. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        print("\n" + "="*80)
        print("🏆 FINAL RESULTS - TARGETING 95%+ ACCURACY")
        print("="*80)
        
        print(summary_df.to_string(index=False, float_format='%.4f'))
        
        best_model = summary_df.iloc[0]
        best_accuracy = best_model['Test_Accuracy']
        
        print(f"\n🥇 BEST MODEL: {best_model['Model']}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        print(f"⏱️  TOTAL TIME: {duration}")
        
        # Check if we achieved the goal
        models_above_95 = summary_df[summary_df['Test_Accuracy'] >= 0.95]
        models_above_90 = summary_df[summary_df['Test_Accuracy'] >= 0.90]
        
        if len(models_above_95) > 0:
            print("\n🎉🎉🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            print("✅ TARGET REACHED!")
            print(f"🏅 {len(models_above_95)} model(s) achieved 95%+:")
            for _, row in models_above_95.iterrows():
                print(f"   • {row['Model']}: {row['Test_Accuracy']:.4f}")
        elif len(models_above_90) > 0:
            gap = 0.95 - best_accuracy
            print(f"\n🔥 EXCELLENT PROGRESS! Need {gap:.4f} ({gap*100:.2f}%) more")
            print(f"💪 {len(models_above_90)} model(s) above 90%:")
            for _, row in models_above_90.iterrows():
                print(f"   • {row['Model']}: {row['Test_Accuracy']:.4f}")
        else:
            print("\n📈 Continue optimizing for higher accuracy")
        
        logger.info("Final 95% pipeline complete!")
        
        return {
            'best_accuracy': best_accuracy,
            'best_model': best_model['Model'],
            'target_achieved': best_accuracy >= 0.95,
            'models_above_95': len(models_above_95),
            'models_above_90': len(models_above_90),
            'summary': summary_df
        }

if __name__ == "__main__":
    import sys
    
    # Check system resources first
    logger.info("Checking system resources before starting...")
    if not check_system_resources():
        logger.warning("System is under high load. Consider waiting for load to reduce.")
        response = input("Continue anyway? (y/n): ").lower()
        if response != 'y':
            logger.info("Exiting due to high system load.")
            sys.exit(1)
    
    final_system = Final95PercentSolution()
    
    # Allow running just LightGBM for testing
    if len(sys.argv) > 1 and sys.argv[1] == "--test-lightgbm":
        logger.info("🧪 TESTING MODE: Running LightGBM only")
        
        # Load data
        df = final_system.load_and_prepare_ultimate_features()
        if df is None:
            sys.exit(1)
        
        X_train, X_test, y_train, y_test = final_system.advanced_feature_engineering(df)
        
        # Create just LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,  # Reduced for testing
            max_depth=8,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=2,  # Conservative
            verbose=-1
        )
        
        models = {'LightGBM_Test': lgb_model}
        results = final_system.comprehensive_evaluation(models, X_train, X_test, y_train, y_test)
        
        best_acc = results['LightGBM_Test']['test_accuracy']
        logger.info(f"🎯 LightGBM Test Result: {best_acc:.4f} ({best_acc*100:.2f}%)")
        
    else:
        # Run the full pipeline
        logger.info("🚀 Running full 95%+ accuracy pipeline...")
        results = final_system.run_final_95_pipeline()
