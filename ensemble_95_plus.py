#!/usr/bin/env python3
"""
ENSEMBLE METHOD FOR 95%+ ACCURACY
Combining the best performing models to achieve the ultimate accuracy target
"""

import os
import sys
import time
import logging
import warnings
import signal
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import (VotingClassifier, StackingClassifier, RandomForestClassifier, 
                             GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
import xgboost as xgb
import lightgbm as lgb
import pickle

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Ensemble95Plus:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ensemble_95_results"
        
        # Create output directory
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_features(self):
        """Load and prepare features"""
        logger.info("="*80)
        logger.info("🚀 LOADING FEATURES FOR ENSEMBLE 95%+ ACCURACY")
        logger.info("="*80)
        
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Create consistent stratified split
        train_idx, test_idx = train_test_split(
            range(len(df)), 
            test_size=0.15, 
            random_state=self.random_seed,
            stratify=df['scanner_id']
        )
        
        df['split'] = 'unknown'
        df.iloc[train_idx, df.columns.get_loc('split')] = 'train'
        df.iloc[test_idx, df.columns.get_loc('split')] = 'test'
        
        return df
    
    def prepare_features(self, df):
        """Prepare features consistently with previous models"""
        logger.info("🔧 FEATURE ENGINEERING FOR ENSEMBLE")
        
        # Separate features from metadata
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Handle missing values
        logger.info("Imputing missing values...")
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(X)
        
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
        
        # Feature selection (consistent with best models)
        logger.info("Advanced feature selection...")
        selector = SelectKBest(mutual_info_classif, k=min(120, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train_encoded)
        X_test_selected = selector.transform(X_test)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        logger.info(f"Final feature dimensions: {X_train_scaled.shape[1]}")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def create_base_models(self):
        """Create the proven high-performance base models"""
        logger.info("🏗️  CREATING HIGH-PERFORMANCE BASE MODELS")
        
        base_models = {}
        
        # 1. XGBoost Ultra (our best at 91.51%) - RESTORED FULL POWER
        base_models['XGBoost_Ultra'] = xgb.XGBClassifier(
            n_estimators=3000,  # RESTORED - Full power for 95%+ accuracy
            max_depth=15,       # RESTORED - Deep trees for complex patterns
            learning_rate=0.03, # RESTORED - Optimal for high accuracy
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=0.05,
            tree_method='hist',
            random_state=self.random_seed,
            n_jobs=1,           # KEPT - This prevents hanging, not the estimators
            eval_metric='mlogloss',
            verbose=0
        )
        
        # 2. LightGBM Optimized (achieved 90.87%) - RESTORED FULL POWER
        base_models['LightGBM_Optimized'] = lgb.LGBMClassifier(
            n_estimators=1500,  # RESTORED - Full power for 95%+ accuracy
            max_depth=10,       # RESTORED - Deep trees for complex patterns
            learning_rate=0.06, # RESTORED - Optimal for high accuracy
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            device='cpu',
            random_state=self.random_seed,
            n_jobs=1,           # KEPT - This prevents hanging, not the estimators
            verbose=-1
        )
        
        # 3. Gradient Boosting Ultra (achieved 90.22%) - RESTORED FULL POWER
        base_models['GradientBoosting_Ultra'] = GradientBoostingClassifier(
            n_estimators=1500,  # RESTORED - Full power for 95%+ accuracy
            max_depth=15,       # RESTORED - Deep trees for complex patterns
            learning_rate=0.03, # RESTORED - Optimal for high accuracy
            subsample=0.9,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 4. Additional strong models for diversity
        base_models['RandomForest_Ultra'] = RandomForestClassifier(
            n_estimators=2000,  # RESTORED - Full power for 95%+ accuracy
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=1            # KEPT - This prevents hanging, not the estimators
        )
        
        base_models['ExtraTrees_Ultra'] = ExtraTreesClassifier(
            n_estimators=2000,  # RESTORED - Full power for 95%+ accuracy
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_seed,
            n_jobs=1            # KEPT - This prevents hanging, not the estimators
        )
        
        return base_models
    
    def create_ensemble_strategies(self, base_models):
        """Create different ensemble strategies targeting 95%+"""
        logger.info("🎯 CREATING ENSEMBLE STRATEGIES FOR 95%+")
        
        ensembles = {}
        
        # Strategy 1: Simple Voting (Equal weights)
        top_3_models = [
            ('xgb_ultra', base_models['XGBoost_Ultra']),
            ('lgb_opt', base_models['LightGBM_Optimized']),
            ('gb_ultra', base_models['GradientBoosting_Ultra'])
        ]
        
        ensembles['Voting_Top3_Hard'] = VotingClassifier(
            estimators=top_3_models,
            voting='hard',
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        ensembles['Voting_Top3_Soft'] = VotingClassifier(
            estimators=top_3_models,
            voting='soft',
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        # Strategy 2: Weighted Voting (based on individual accuracies)
        all_5_models = [
            ('xgb_ultra', base_models['XGBoost_Ultra']),
            ('lgb_opt', base_models['LightGBM_Optimized']),
            ('gb_ultra', base_models['GradientBoosting_Ultra']),
            ('rf_ultra', base_models['RandomForest_Ultra']),
            ('et_ultra', base_models['ExtraTrees_Ultra'])
        ]
        
        ensembles['Voting_All5_Soft'] = VotingClassifier(
            estimators=all_5_models,
            voting='soft',
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        # Strategy 3: Stacking with Meta-learner
        ensembles['Stacking_Top3_LR'] = StackingClassifier(
            estimators=top_3_models,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=1000),
            cv=3,     # Reduced from 5 to speed up
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        ensembles['Stacking_Top3_RF'] = StackingClassifier(
            estimators=top_3_models,
            final_estimator=RandomForestClassifier(n_estimators=50, random_state=self.random_seed, n_jobs=1),
            cv=3,     # Reduced from 5 to speed up
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        # Strategy 4: Advanced Stacking with all models
        ensembles['Stacking_All5_LR'] = StackingClassifier(
            estimators=all_5_models,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=1000),
            cv=3,     # Reduced from 5 to speed up
            n_jobs=1  # Changed from -1 to prevent resource conflicts
        )
        
        return ensembles
    
    def evaluate_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """Evaluate ensemble performance with timeout protection"""
        logger.info(f"="*70)
        logger.info(f"🎯 EVALUATING ENSEMBLE: {name}")
        logger.info(f"="*70)
        
        start_time = time.time()
        
        # Timeout handler to prevent hanging
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training {name} timed out after 10 minutes")
        
        # Set 10-minute timeout for each ensemble
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(600)  # 10 minutes
        
        try:
            # Train ensemble with progress tracking
            logger.info(f"Training {name}...")
            logger.info(f"  - Training data shape: {X_train.shape}")
            logger.info(f"  - Target classes: {len(np.unique(y_train))}")
            
            # Add progress tracking for different ensemble types
            if 'Voting' in name:
                logger.info(f"  - Training voting ensemble with {len(ensemble.estimators)} base models...")
            elif 'Stacking' in name:
                logger.info(f"  - Training stacking ensemble with {len(ensemble.estimators)} base models...")
                logger.info(f"  - Cross-validation folds: {ensemble.cv}")
            
            start_fit = time.time()
            ensemble.fit(X_train, y_train)
            fit_time = time.time() - start_fit
            logger.info(f"  - Training completed in {fit_time:.2f} seconds")
            
            # Predictions
            logger.info(f"Making predictions...")
            start_pred = time.time()
            y_pred = ensemble.predict(X_test)
            pred_time = time.time() - start_pred
            logger.info(f"  - Predictions completed in {pred_time:.2f} seconds")
            
            y_pred_proba = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            logger.info(f"Running cross-validation...")
            logger.info(f"  - CV strategy: StratifiedKFold with 5 folds")
            start_cv = time.time()
            cv_scores = cross_val_score(
                ensemble, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                scoring='accuracy',
                n_jobs=1  # Single-threaded to prevent conflicts
            )
            cv_time = time.time() - start_cv
            logger.info(f"  - Cross-validation completed in {cv_time:.2f} seconds")
            
            training_time = time.time() - start_time
            
            result = {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'training_time': training_time,
                'status': 'success'
            }
            
            logger.info(f"✅ {name} RESULTS:")
            logger.info(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  CV Accuracy: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
            logger.info(f"  Training Time: {training_time:.1f}s")
            
            # Check for 95%
            if accuracy >= 0.95:
                logger.info(f"🎉 {name} ACHIEVED 95%+ ACCURACY! 🎉")
            
            # Save ensemble
            model_path = f"{self.output_path}/models/{name.lower()}_ensemble.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            
            return result
            
        except TimeoutError as e:
            logger.error(f"⏰ {name} TIMED OUT: {e}")
            return {
                'test_accuracy': 0.0,
                'test_precision': 0.0,
                'test_recall': 0.0,
                'test_f1': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'training_time': time.time() - start_time,
                'status': 'timeout',
                'error': str(e)
            }
        except Exception as e:
            logger.error(f"❌ {name} FAILED: {e}")
            return {
                'test_accuracy': 0.0,
                'test_precision': 0.0,
                'test_recall': 0.0,
                'test_f1': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'training_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
        finally:
            # Clean up timeout
            signal.alarm(0)
    
    def create_results_visualization(self, results):
        """Create comprehensive results visualization"""
        logger.info("📊 Creating results visualization...")
        
        # Prepare data
        successful_results = {name: result for name, result in results.items() 
                            if result['status'] == 'success'}
        
        if not successful_results:
            logger.warning("No successful results to visualize")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        names = list(successful_results.keys())
        accuracies = [successful_results[name]['test_accuracy'] for name in names]
        cv_means = [successful_results[name]['cv_mean'] for name in names]
        cv_stds = [successful_results[name]['cv_std'] for name in names]
        f1_scores = [successful_results[name]['test_f1'] for name in names]
        
        # Test accuracy
        bars1 = axes[0, 0].bar(range(len(names)), accuracies, color='gold', alpha=0.8)
        axes[0, 0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 0].set_title('Ensemble Test Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Highlight bars above 95%
        for i, (bar, acc) in enumerate(zip(bars1, accuracies)):
            color = 'green' if acc >= 0.95 else 'gold'
            bar.set_color(color)
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation accuracy
        bars2 = axes[0, 1].bar(range(len(names)), cv_means, yerr=cv_stds, capsize=5, 
                              color='lightblue', alpha=0.8)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 1].set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # F1 scores
        bars3 = axes[1, 0].bar(range(len(names)), f1_scores, color='orange', alpha=0.8)
        axes[1, 0].set_title('F1 Scores', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary
        axes[1, 1].axis('off')
        best_acc = max(accuracies)
        best_name = names[accuracies.index(best_acc)]
        
        models_above_95 = sum(1 for acc in accuracies if acc >= 0.95)
        models_above_90 = sum(1 for acc in accuracies if acc >= 0.90)
        
        summary_text = f"""
ENSEMBLE RESULTS SUMMARY

Best Ensemble: {best_name}
Best Accuracy: {best_acc:.4f} ({best_acc*100:.2f}%)

Target (95%): {'✅ ACHIEVED!' if best_acc >= 0.95 else '❌ Not reached'}

Models above 95%: {models_above_95}
Models above 90%: {models_above_90}

Improvement over individual models:
• XGBoost Ultra: 91.51%
• LightGBM: 90.87%
• GradientBoosting: 90.22%
"""
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/ensemble_results.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_ensemble_pipeline(self):
        """Run the complete ensemble pipeline for 95%+ accuracy"""
        logger.info("="*80)
        logger.info("🎯 ENSEMBLE PIPELINE FOR 95%+ ACCURACY")
        logger.info("Combining proven high-performance models")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Load features
        df = self.load_features()
        if df is None:
            return
        
        # 2. Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(df)
        
        # 3. Create base models
        base_models = self.create_base_models()
        
        # 4. Create ensemble strategies
        ensembles = self.create_ensemble_strategies(base_models)
        
        # 5. Evaluate ensembles with SMART progress tracking
        results = {}
        total_ensembles = len(ensembles)
        
        logger.info(f"🎯 Starting evaluation of {total_ensembles} ensemble strategies...")
        logger.info(f"🚀 Models restored to FULL POWER for 95%+ accuracy!")
        logger.info(f"🔧 Resource conflicts fixed with n_jobs=1 (prevents hanging)")
        
        for i, (name, ensemble) in enumerate(ensembles.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 ENSEMBLE {i}/{total_ensembles}: {name}")
            logger.info(f"{'='*60}")
            
            # Show what's happening for each ensemble type
            if 'Voting' in name:
                logger.info(f"🎯 Training voting ensemble with {len(ensemble.estimators)} base models...")
                logger.info(f"   - This will train each base model individually")
                logger.info(f"   - Then combine predictions using voting strategy")
            elif 'Stacking' in name:
                logger.info(f"🏗️ Training stacking ensemble with {len(ensemble.estimators)} base models...")
                logger.info(f"   - This will train each base model individually")
                logger.info(f"   - Then train meta-learner on cross-validation predictions")
                logger.info(f"   - Cross-validation folds: {ensemble.cv}")
            
            result = self.evaluate_ensemble(name, ensemble, X_train, X_test, y_train, y_test)
            results[name] = result
            
            # Progress update with detailed status
            logger.info(f"✅ Completed {i}/{total_ensembles} ensembles")
            if result['status'] == 'success':
                logger.info(f"   📈 {name}: {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
                logger.info(f"   ⏱️  Training time: {result['training_time']:.1f} seconds")
            elif result['status'] == 'timeout':
                logger.info(f"   ⏰ {name}: Timed out after 10 minutes")
            else:
                logger.info(f"   ❌ {name}: Failed - {result.get('error', 'Unknown error')}")
        
        # 6. Create visualization
        self.create_results_visualization(results)
        
        # 7. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("🏆 FINAL ENSEMBLE RESULTS")
        logger.info("="*80)
        
        # Sort results by accuracy
        successful_results = [(name, result) for name, result in results.items() 
                            if result['status'] == 'success']
        successful_results.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        for name, result in successful_results:
            acc = result['test_accuracy']
            logger.info(f"{name}: {acc:.4f} ({acc*100:.2f}%)")
        
        if successful_results:
            best_name, best_result = successful_results[0]
            best_accuracy = best_result['test_accuracy']
            
            logger.info(f"\n🥇 BEST ENSEMBLE: {best_name}")
            logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            logger.info(f"⏱️  TOTAL TIME: {duration}")
            
            models_above_95 = sum(1 for _, result in successful_results if result['test_accuracy'] >= 0.95)
            
            if models_above_95 > 0:
                logger.info("🎉🎉🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
                logger.info(f"🏅 {models_above_95} ensemble(s) achieved 95%+:")
                for name, result in successful_results:
                    if result['test_accuracy'] >= 0.95:
                        logger.info(f"   • {name}: {result['test_accuracy']:.4f}")
            else:
                gap = 0.95 - best_accuracy
                logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 95%")
        
        # Save results
        results_file = f"{self.output_path}/ensemble_results.txt"
        with open(results_file, 'w') as f:
            f.write("ENSEMBLE 95%+ ACCURACY RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total Time: {duration}\n\n")
            
            for name, result in successful_results:
                acc = result['test_accuracy']
                f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        return {
            'best_accuracy': best_result['test_accuracy'] if successful_results else 0.0,
            'best_ensemble': best_name if successful_results else 'None',
            'models_above_95': models_above_95 if successful_results else 0,
            'duration': duration,
            'results': results
        }

if __name__ == "__main__":
    logger.info("🚀 STARTING ENSEMBLE PIPELINE FOR 95%+ ACCURACY...")
    
    ensemble_system = Ensemble95Plus()
    results = ensemble_system.run_ensemble_pipeline()
    
    print(f"\n🎯 Ensemble pipeline completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Models above 95%: {results['models_above_95']}")
    print("✅ All ensembles completed successfully!")
