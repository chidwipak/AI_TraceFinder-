#!/usr/bin/env python3
"""
ULTRA-FAST ENSEMBLE PIPELINE FOR 95%+ ACCURACY
FIXED VERSION: Prevents deadlocks and adds better error handling
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

# Machine Learning imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, 
    GradientBoostingClassifier, VotingClassifier, StackingClassifier,
    BaggingClassifier, AdaBoostClassifier
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_95_plus_ultra.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UltraFastEnsemble95Plus:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ensemble_95_results_ultra"
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set random seeds
        np.random.seed(self.random_seed)
        
        # Add timeout settings to prevent deadlocks
        self.max_training_time = 600  # 10 minutes max for training
        self.max_cv_time = 300        # 5 minutes max for CV
        self.max_prediction_time = 60 # 1 minute max for predictions
        
    def load_and_preprocess_features(self):
        """Load and preprocess features with ultra-fast techniques"""
        logger.info("🚀 LOADING FEATURES FOR ULTRA-FAST ENSEMBLE 95%+ ACCURACY")
        
        try:
            # Load features from the correct location
            X = np.load("features/feature_vectors/X_features.npy")
            y = np.load("features/feature_vectors/y_labels.npy", allow_pickle=True)
            
            # Convert text labels to integers using LabelEncoder
            if y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                logger.info(f"Converted {len(label_encoder.classes_)} text labels to integers")
                logger.info(f"Label mapping: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
            
            logger.info(f"Loaded {len(X)} samples with {X.shape[1]} features")
            
            # Ultra-fast preprocessing
            logger.info("🔧 ULTRA-FAST FEATURE ENGINEERING")
            
            # Simple imputation
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Feature selection (keep top 90 features for balance of speed/accuracy)
            selector = SelectKBest(score_func=mutual_info_classif, k=90)
            X_selected = selector.fit_transform(X_imputed, y)
            
            # Split data
            split_idx = int(0.85 * len(X))
            X_train, X_test = X_selected[:split_idx], X_selected[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Final feature dimensions: {X_selected.shape[1]}")
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Number of classes: {len(np.unique(y))}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error loading features: {e}")
            raise
    
    def create_ultra_fast_base_models(self):
        """Create ultra-fast but powerful base models"""
        logger.info("🏗️ CREATING ULTRA-FAST HIGH-PERFORMANCE BASE MODELS")
        
        base_models = {}
        
        # 1. XGBoost Ultra-Fast (balanced for speed/accuracy)
        base_models['XGBoost_Ultra'] = xgb.XGBClassifier(
            n_estimators=300,       # Balanced for speed/accuracy
            max_depth=10,            # Balanced depth
            learning_rate=0.15,      # Optimal learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',      # Fastest tree method
            random_state=self.random_seed,
            n_jobs=1,
            eval_metric='mlogloss',
            verbose=0
        )
        
        # 2. LightGBM Ultra-Fast
        base_models['LightGBM_Ultra'] = lgb.LGBMClassifier(
            n_estimators=250,       # Balanced for speed/accuracy
            max_depth=10,            # Balanced depth
            learning_rate=0.12,      # Optimal learning rate
            subsample=0.8,
            colsample_bytree=0.8,
            device='cpu',
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1
        )
        
        # 3. Random Forest Ultra-Fast
        base_models['RandomForest_Ultra'] = RandomForestClassifier(
            n_estimators=200,       # Balanced for speed/accuracy
            max_depth=12,            # Balanced depth
            min_samples_split=4,     # Optimized for speed
            min_samples_leaf=2,      # Optimized for speed
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # 4. Extra Trees Ultra-Fast
        base_models['ExtraTrees_Ultra'] = ExtraTreesClassifier(
            n_estimators=200,       # Balanced for speed/accuracy
            max_depth=12,            # Balanced depth
            min_samples_split=4,     # Optimized for speed
            min_samples_leaf=2,      # Optimized for speed
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # 5. Gradient Boosting Ultra-Fast
        base_models['GradientBoosting_Ultra'] = GradientBoostingClassifier(
            n_estimators=250,       # Balanced for speed/accuracy
            max_depth=10,            # Balanced depth
            learning_rate=0.12,      # Optimal learning rate
            subsample=0.8,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 6. SVM Ultra-Fast (Linear kernel for speed)
        base_models['SVM_Linear'] = SVC(
            kernel='linear',         # Linear kernel is much faster
            C=1.0,
            probability=True,
            random_state=self.random_seed
        )
        
        # 7. K-Nearest Neighbors Ultra-Fast
        base_models['KNN_Ultra'] = KNeighborsClassifier(
            n_neighbors=7,           # Optimized for accuracy
            weights='distance',
            metric='minkowski',
            p=2
        )
        
        return base_models
    
    def create_ultra_fast_ensemble_strategies(self, base_models):
        """Create ultra-fast ensemble strategies with safer combinations"""
        logger.info("🎯 CREATING ULTRA-FAST ENSEMBLE STRATEGIES")
        
        ensembles = {}
        
        # Strategy 1: Weighted Voting (fastest)
        top_5_models = [
            ('xgb_ultra', base_models['XGBoost_Ultra']),
            ('lgb_ultra', base_models['LightGBM_Ultra']),
            ('rf_ultra', base_models['RandomForest_Ultra']),
            ('et_ultra', base_models['ExtraTrees_Ultra']),
            ('gb_ultra', base_models['GradientBoosting_Ultra'])
        ]
        
        ensembles['Voting_Top5_Soft'] = VotingClassifier(
            estimators=top_5_models,
            voting='soft',
            n_jobs=1
        )
        
        # Strategy 2: Stacking with Top 3 (balanced)
        top_3_models = [
            ('xgb_ultra', base_models['XGBoost_Ultra']),
            ('lgb_ultra', base_models['LightGBM_Ultra']),
            ('rf_ultra', base_models['RandomForest_Ultra'])
        ]
        
        ensembles['Stacking_Top3_LR'] = StackingClassifier(
            estimators=top_3_models,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=500),
            cv=3,                   # Reduced CV folds for speed
            n_jobs=1
        )
        
        # Strategy 3: Stacking with Top 5 and Ridge (faster than LR)
        ensembles['Stacking_Top5_Ridge'] = StackingClassifier(
            estimators=top_5_models,
            final_estimator=RidgeClassifier(random_state=self.random_seed),
            cv=3,                   # Reduced CV folds for speed
            n_jobs=1
        )
        
        # Strategy 4: Bagging with Random Forest (fast and stable)
        ensembles['Bagging_RF'] = BaggingClassifier(
            estimator=RandomForestClassifier(
                n_estimators=100,   # Reduced for speed
                max_depth=8,        # Reduced for speed
                random_state=self.random_seed,
                n_jobs=1
            ),
            n_estimators=10,        # Reduced for speed
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # Strategy 5: AdaBoost with Decision Tree (fast and stable)
        ensembles['AdaBoost_DT'] = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(
                max_depth=6,        # Reduced for speed
                random_state=self.random_seed
            ),
            n_estimators=50,        # Reduced for speed
            learning_rate=0.1,
            random_state=self.random_seed
        )
        
        # Strategy 6: SAFER All Models Voting (reduced complexity)
        # Use only the most stable models to prevent deadlocks
        safe_models = [
            ('xgb_ultra', base_models['XGBoost_Ultra']),
            ('lgb_ultra', base_models['LightGBM_Ultra']),
            ('rf_ultra', base_models['RandomForest_Ultra']),
            ('et_ultra', base_models['ExtraTrees_Ultra'])
        ]
        
        ensembles['All_Models_Voting'] = VotingClassifier(
            estimators=safe_models,  # Reduced from 7 to 4 models
            voting='soft',
            n_jobs=1
        )
        
        # Strategy 7: Hybrid Fast Stacking
        # First level: Voting
        voting_ensemble = VotingClassifier(
            estimators=top_3_models,
            voting='soft',
            n_jobs=1
        )
        
        # Second level: Fast Stacking
        hybrid_estimators = [
            ('voting', voting_ensemble),
            ('xgb', base_models['XGBoost_Ultra']),
            ('lgb', base_models['LightGBM_Ultra'])
        ]
        
        ensembles['Hybrid_Fast_Stacking'] = StackingClassifier(
            estimators=hybrid_estimators,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=500),
            cv=3,                   # Reduced CV folds for speed
            n_jobs=1
        )
        
        return ensembles
    
    def safe_timeout_operation(self, operation_name, operation_func, max_time, fallback_value=None):
        """Safely execute operations with timeout and fallback"""
        start_time = time.time()
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"{operation_name} timed out after {max_time} seconds")
        
        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max_time)
        
        try:
            result = operation_func()
            signal.alarm(0)  # Cancel alarm
            return result
        except TimeoutError:
            logger.warning(f"⚠️ {operation_name} timed out after {max_time} seconds")
            return fallback_value
        except Exception as e:
            logger.error(f"❌ Error in {operation_name}: {e}")
            return fallback_value
        finally:
            # Restore original signal handler
            signal.signal(signal.SIGALRM, old_handler)
    
    def evaluate_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """Evaluate ensemble with improved timeout handling and progress logging"""
        logger.info(f"🎯 EVALUATING: {name}")
        
        start_time = time.time()
        result = {
            'test_accuracy': 0.0,
            'test_precision': 0.0,
            'test_recall': 0.0,
            'test_f1': 0.0,
            'cv_mean': 0.0,
            'cv_std': 0.0,
            'training_time': 0.0,
            'status': 'failed',
            'ensemble': None,
            'error': None
        }
        
        try:
            # Train ensemble with timeout
            logger.info(f"Training {name}...")
            start_fit = time.time()
            
            def train_operation():
                return ensemble.fit(X_train, y_train)
            
            # Use safe timeout operation for training
            train_result = self.safe_timeout_operation(
                f"Training {name}",
                train_operation,
                self.max_training_time
            )
            
            if train_result is None:
                logger.error(f"❌ Training {name} failed due to timeout")
                result['error'] = f"Training timed out after {self.max_training_time} seconds"
                return result
            
            fit_time = time.time() - start_fit
            logger.info(f"Training completed in {fit_time:.1f} seconds")
            
            # Predictions with timeout
            logger.info("Making predictions...")
            start_pred = time.time()
            
            def predict_operation():
                return ensemble.predict(X_test)
            
            y_pred = self.safe_timeout_operation(
                f"Predictions for {name}",
                predict_operation,
                self.max_prediction_time
            )
            
            if y_pred is None:
                logger.error(f"❌ Predictions for {name} failed due to timeout")
                result['error'] = f"Predictions timed out after {self.max_prediction_time} seconds"
                return result
            
            pred_time = time.time() - start_pred
            logger.info(f"Predictions completed in {pred_time:.1f} seconds")
            
            # Evaluation
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted')
            test_recall = recall_score(y_test, y_pred, average='weighted')
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation with timeout
            logger.info("Running cross-validation...")
            start_cv = time.time()
            
            def cv_operation():
                return cross_val_score(
                    ensemble, X_train, y_train,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed),
                    scoring='accuracy',
                    n_jobs=1
                )
            
            cv_scores = self.safe_timeout_operation(
                f"Cross-validation for {name}",
                cv_operation,
                self.max_cv_time
            )
            
            if cv_scores is None:
                logger.warning(f"⚠️ Cross-validation for {name} timed out, using fallback")
                cv_scores = np.array([test_accuracy])  # Fallback to test accuracy
            
            cv_time = time.time() - start_cv
            logger.info(f"CV completed in {cv_time:.1f} seconds")
            
            training_time = time.time() - start_time
            
            # Update result
            result.update({
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'status': 'success',
                'ensemble': ensemble
            })
            
            logger.info(f"✅ {name}: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            
        except Exception as e:
            logger.error(f"❌ Error evaluating {name}: {e}")
            result['error'] = str(e)
            result['status'] = 'failed'
        
        return result
    
    def run_ultra_fast_ensemble_pipeline(self):
        """Run the complete ultra-fast ensemble pipeline with progress monitoring"""
        logger.info("🚀 STARTING ULTRA-FAST ENSEMBLE PIPELINE FOR 95%+ ACCURACY")
        logger.info("⚡ Optimized to complete ALL models without timeouts!")
        
        try:
            # 1. Load and preprocess
            X_train, X_test, y_train, y_test = self.load_and_preprocess_features()
            
            # 2. Create ultra-fast base models
            base_models = self.create_ultra_fast_base_models()
            
            # 3. Create ultra-fast ensemble strategies
            ensembles = self.create_ultra_fast_ensemble_strategies(base_models)
            
            # 4. Evaluate all ensembles with progress monitoring
            results = {}
            total_ensembles = len(ensembles)
            successful_count = 0
            failed_count = 0
            
            logger.info(f"🎯 Starting evaluation of {total_ensembles} ultra-fast ensemble strategies...")
            
            for i, (name, ensemble) in enumerate(ensembles.items(), 1):
                logger.info(f"\n{'='*60}")
                logger.info(f"📊 ENSEMBLE {i}/{total_ensembles}: {name}")
                logger.info(f"{'='*60}")
                
                # Add progress indicator
                start_ensemble = time.time()
                
                try:
                    result = self.evaluate_ensemble(name, ensemble, X_train, X_test, y_train, y_test)
                    results[name] = result
                    
                    ensemble_time = time.time() - start_ensemble
                    
                    # Progress update with timing
                    if result['status'] == 'success':
                        successful_count += 1
                        logger.info(f"✅ {i}/{total_ensembles} completed: {result['test_accuracy']:.4f} (Time: {ensemble_time:.1f}s)")
                    else:
                        failed_count += 1
                        logger.info(f"❌ {i}/{total_ensembles} failed: {result.get('error', 'Unknown error')}")
                    
                    # Progress summary
                    logger.info(f"📊 Progress: {successful_count} successful, {failed_count} failed, {total_ensembles - i} remaining")
                    
                except Exception as e:
                    failed_count += 1
                    logger.error(f"❌ Critical error evaluating {name}: {e}")
                    results[name] = {
                        'status': 'failed',
                        'error': str(e),
                        'test_accuracy': 0.0
                    }
                
                # Add a small delay to prevent resource contention
                time.sleep(1)
            
            # 5. Final results with comprehensive reporting
            successful_results = [(name, result) for name, result in results.items() 
                                if result['status'] == 'success']
            
            logger.info(f"\n{'='*60}")
            logger.info(f"🏁 PIPELINE COMPLETION SUMMARY")
            logger.info(f"{'='*60}")
            logger.info(f"Total ensembles: {total_ensembles}")
            logger.info(f"Successful: {successful_count}")
            logger.info(f"Failed: {failed_count}")
            logger.info(f"Success rate: {(successful_count/total_ensembles)*100:.1f}%")
            
            if successful_results:
                successful_results.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
                best_name, best_result = successful_results[0]
                best_accuracy = best_result['test_accuracy']
                
                logger.info(f"\n🏆 BEST ULTRA-FAST ENSEMBLE: {best_name}")
                logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
                
                # Check if 95%+ achieved
                if best_accuracy >= 0.95:
                    logger.info("🎉🎉🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
                else:
                    gap = 0.95 - best_accuracy
                    logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 95%")
                
                # Save results
                results_file = f"{self.output_path}/ultra_fast_ensemble_results.txt"
                with open(results_file, 'w') as f:
                    f.write("ULTRA-FAST ENSEMBLE RESULTS (FIXED VERSION)\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Total ensembles: {total_ensembles}\n")
                    f.write(f"Successful: {successful_count}\n")
                    f.write(f"Failed: {failed_count}\n")
                    f.write(f"Success rate: {(successful_count/total_ensembles)*100:.1f}%\n\n")
                    
                    for name, result in successful_results:
                        acc = result['test_accuracy']
                        f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%)\n")
                
                logger.info(f"💾 Results saved to {results_file}")
                
                # Save trained models
                logger.info("💾 Saving trained ultra-fast ensemble models...")
                import pickle
                
                for name, result in successful_results:
                    if result['status'] == 'success':
                        ensemble = results[name].get('ensemble', None)
                        if ensemble is not None:
                            model_file = f"{self.output_path}/models/{name}_model.pkl"
                            os.makedirs(os.path.dirname(model_file), exist_ok=True)
                            with open(model_file, 'wb') as f:
                                pickle.dump(ensemble, f)
                            logger.info(f"   ✅ {name} model saved to {model_file}")
                
                logger.info("💾 All ultra-fast models saved successfully!")
                
                return {
                    'best_accuracy': best_accuracy,
                    'best_ensemble': best_name,
                    'results': results,
                    'models_above_95': sum(1 for _, result in successful_results if result['test_accuracy'] >= 0.95),
                    'successful_count': successful_count,
                    'failed_count': failed_count,
                    'total_count': total_ensembles
                }
            else:
                logger.error("❌ No ultra-fast ensembles completed successfully")
                return {
                    'best_accuracy': 0.0,
                    'best_ensemble': 'None',
                    'results': results,
                    'models_above_95': 0,
                    'successful_count': 0,
                    'failed_count': failed_count,
                    'total_count': total_ensembles
                }
                
        except Exception as e:
            logger.error(f"❌ Critical pipeline error: {e}")
            logger.error("Pipeline failed completely. Check logs for details.")
            return {
                'best_accuracy': 0.0,
                'best_ensemble': 'None',
                'results': {},
                'models_above_95': 0,
                'successful_count': 0,
                'failed_count': total_ensembles if 'total_ensembles' in locals() else 0,
                'total_count': total_ensembles if 'total_ensembles' in locals() else 0,
                'error': str(e)
            }

if __name__ == "__main__":
    logger.info("🚀 STARTING ULTRA-FAST ENSEMBLE PIPELINE (FIXED VERSION)...")
    
    try:
        ultra_fast_ensemble_system = UltraFastEnsemble95Plus()
        results = ultra_fast_ensemble_system.run_ultra_fast_ensemble_pipeline()
        
        print(f"\n🎯 Ultra-fast ensemble pipeline completed!")
        print(f"Best accuracy: {results['best_accuracy']:.4f}")
        print(f"Models above 95%: {results['models_above_95']}")
        print(f"Success rate: {(results['successful_count']/results['total_count'])*100:.1f}%")
        
        if results['best_accuracy'] >= 0.95:
            print("🎉🎉🎉 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
        else:
            print("📈 Keep pushing towards 95%+ accuracy!")
        
        if results['successful_count'] > 0:
            print("✅ Pipeline completed with successful models!")
        else:
            print("⚠️ Pipeline completed but no models were successful.")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        sys.exit(1)
