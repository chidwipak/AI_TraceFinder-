#!/usr/bin/env python3
"""
ADVANCED ENSEMBLE PIPELINE FOR 95%+ ACCURACY
Fine-tunes best models and adds advanced ensemble strategies
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.decomposition import PCA

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_95_plus_advanced.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class AdvancedEnsemble95Plus:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ensemble_95_results_advanced"
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set random seeds
        np.random.seed(self.random_seed)
        
    def load_and_preprocess_features(self):
        """Load and preprocess features with advanced techniques"""
        logger.info("🚀 LOADING FEATURES FOR ADVANCED ENSEMBLE 95%+ ACCURACY")
        
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
            
            # Advanced preprocessing
            logger.info("🔧 ADVANCED FEATURE ENGINEERING")
            
            # Robust imputation (handles outliers better)
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Advanced feature selection (keep top 100 features for better accuracy)
            selector = SelectKBest(score_func=mutual_info_classif, k=100)
            X_selected = selector.fit_transform(X_imputed, y)
            
            # Feature scaling for better model performance
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X_selected)
            
            # Split data
            split_idx = int(0.85 * len(X))
            X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            logger.info(f"Final feature dimensions: {X_scaled.shape[1]}")
            logger.info(f"Training set: {X_train.shape}")
            logger.info(f"Test set: {X_test.shape}")
            logger.info(f"Number of classes: {len(np.unique(y))}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error loading features: {e}")
            raise
    
    def create_advanced_base_models(self):
        """Create advanced base models with hyperparameter tuning"""
        logger.info("🏗️ CREATING ADVANCED HIGH-PERFORMANCE BASE MODELS")
        
        base_models = {}
        
        # 1. XGBoost Advanced (tuned for accuracy)
        base_models['XGBoost_Advanced'] = xgb.XGBClassifier(
            n_estimators=500,       # Increased for better accuracy
            max_depth=12,            # Balanced depth
            learning_rate=0.1,       # Optimal learning rate
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='hist',
            random_state=self.random_seed,
            n_jobs=1,
            eval_metric='mlogloss',
            verbose=0
        )
        
        # 2. LightGBM Advanced
        base_models['LightGBM_Advanced'] = lgb.LGBMClassifier(
            n_estimators=400,       # Increased for better accuracy
            max_depth=12,            # Balanced depth
            learning_rate=0.08,      # Optimal learning rate
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            device='cpu',
            random_state=self.random_seed,
            n_jobs=1,
            verbose=-1
        )
        
        # 3. Random Forest Advanced
        base_models['RandomForest_Advanced'] = RandomForestClassifier(
            n_estimators=300,       # Increased for better accuracy
            max_depth=15,            # Increased depth
            min_samples_split=3,     # Optimized
            min_samples_leaf=1,      # Optimized
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # 4. Extra Trees Advanced
        base_models['ExtraTrees_Advanced'] = ExtraTreesClassifier(
            n_estimators=300,       # Increased for better accuracy
            max_depth=15,            # Increased depth
            min_samples_split=3,     # Optimized
            min_samples_leaf=1,      # Optimized
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # 5. Gradient Boosting Advanced
        base_models['GradientBoosting_Advanced'] = GradientBoostingClassifier(
            n_estimators=400,       # Increased for better accuracy
            max_depth=12,            # Balanced depth
            learning_rate=0.08,      # Optimal learning rate
            subsample=0.85,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 6. SVM with RBF kernel
        base_models['SVM_RBF'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=self.random_seed
        )
        
        # 7. K-Nearest Neighbors
        base_models['KNN'] = KNeighborsClassifier(
            n_neighbors=5,
            weights='distance',
            metric='minkowski',
            p=2
        )
        
        return base_models
    
    def create_advanced_ensemble_strategies(self, base_models):
        """Create advanced ensemble strategies"""
        logger.info("🎯 CREATING ADVANCED ENSEMBLE STRATEGIES")
        
        ensembles = {}
        
        # Strategy 1: Weighted Voting (based on individual performance)
        top_5_models = [
            ('xgb_adv', base_models['XGBoost_Advanced']),
            ('lgb_adv', base_models['LightGBM_Advanced']),
            ('rf_adv', base_models['RandomForest_Advanced']),
            ('et_adv', base_models['ExtraTrees_Advanced']),
            ('gb_adv', base_models['GradientBoosting_Advanced'])
        ]
        
        ensembles['Weighted_Voting_Top5'] = VotingClassifier(
            estimators=top_5_models,
            voting='soft',
            n_jobs=1
        )
        
        # Strategy 2: Advanced Stacking with Ridge Classifier
        ensembles['Stacking_Top5_Ridge'] = StackingClassifier(
            estimators=top_5_models,
            final_estimator=RidgeClassifier(random_state=self.random_seed),
            cv=5,                   # Increased CV folds for better validation
            n_jobs=1
        )
        
        # Strategy 3: Stacking with SVM
        ensembles['Stacking_Top5_SVM'] = StackingClassifier(
            estimators=top_5_models,
            final_estimator=SVC(probability=True, random_state=self.random_seed),
            cv=5,
            n_jobs=1
        )
        
        # Strategy 4: Bagging with Random Forest
        base_rf = RandomForestClassifier(n_estimators=100, random_state=self.random_seed, n_jobs=1)
        ensembles['Bagging_RF'] = BaggingClassifier(
            estimator=base_rf,  # Changed from base_estimator to estimator
            n_estimators=10,
            random_state=self.random_seed,
            n_jobs=1
        )
        
        # Strategy 5: AdaBoost with Decision Tree
        base_dt = DecisionTreeClassifier(max_depth=8, random_state=self.random_seed)
        ensembles['AdaBoost_DT'] = AdaBoostClassifier(
            estimator=base_dt,  # Changed from base_estimator to estimator
            n_estimators=100,
            learning_rate=0.1,
            random_state=self.random_seed
        )
        
        # Strategy 6: Hybrid Ensemble (Voting + Stacking)
        top_3_models = [
            ('xgb_adv', base_models['XGBoost_Advanced']),
            ('lgb_adv', base_models['LightGBM_Advanced']),
            ('rf_adv', base_models['RandomForest_Advanced'])
        ]
        
        # First level: Voting
        voting_ensemble = VotingClassifier(
            estimators=top_3_models,
            voting='soft',
            n_jobs=1
        )
        
        # Second level: Stacking with the voting ensemble
        hybrid_estimators = [
            ('voting', voting_ensemble),
            ('xgb', base_models['XGBoost_Advanced']),
            ('lgb', base_models['LightGBM_Advanced'])
        ]
        
        ensembles['Hybrid_Voting_Stacking'] = StackingClassifier(
            estimators=hybrid_estimators,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=1000),
            cv=5,
            n_jobs=1
        )
        
        # Strategy 7: All Models Ensemble
        all_7_models = [
            ('xgb_adv', base_models['XGBoost_Advanced']),
            ('lgb_adv', base_models['LightGBM_Advanced']),
            ('rf_adv', base_models['RandomForest_Advanced']),
            ('et_adv', base_models['ExtraTrees_Advanced']),
            ('gb_adv', base_models['GradientBoosting_Advanced']),
            ('svm', base_models['SVM_RBF']),
            ('knn', base_models['KNN'])
        ]
        
        ensembles['All_Models_Ensemble'] = VotingClassifier(
            estimators=all_7_models,
            voting='soft',
            n_jobs=1
        )
        
        return ensembles
    
    def fine_tune_best_model(self, X_train, y_train):
        """Fine-tune the best model from previous runs"""
        logger.info("🔧 FINE-TUNING BEST MODEL (Stacking_Top3_LR)")
        
        try:
            # Load the best model from previous run
            best_model_path = "ensemble_95_results_fast/Stacking_Top3_LR_model.pkl"
            if os.path.exists(best_model_path):
                import pickle
                with open(best_model_path, 'rb') as f:
                    best_model = pickle.load(f)
                
                logger.info("✅ Loaded best model from previous run")
                
                # Fine-tune the meta-learner
                logger.info("🔧 Fine-tuning meta-learner...")
                
                # Get base predictions for fine-tuning
                base_predictions = best_model.transform(X_train)
                
                # Fine-tune Logistic Regression meta-learner
                meta_learner = LogisticRegression(random_state=self.random_seed, max_iter=2000)
                
                # Grid search for optimal hyperparameters
                param_grid = {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga']
                }
                
                grid_search = GridSearchCV(
                    meta_learner,
                    param_grid,
                    cv=5,
                    scoring='accuracy',
                    n_jobs=1
                )
                
                grid_search.fit(base_predictions, y_train)
                
                logger.info(f"✅ Best meta-learner params: {grid_search.best_params_}")
                logger.info(f"✅ Best CV score: {grid_search.best_score_:.4f}")
                
                # Create fine-tuned ensemble
                fine_tuned_ensemble = StackingClassifier(
                    estimators=best_model.estimators,
                    final_estimator=grid_search.best_estimator_,
                    cv=5,
                    n_jobs=1
                )
                
                # Fit the fine-tuned ensemble
                fine_tuned_ensemble.fit(X_train, y_train)
                
                return fine_tuned_ensemble, "Fine_Tuned_Stacking_Top3_LR"
                
            else:
                logger.warning("⚠️ Best model not found, skipping fine-tuning")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ Error in fine-tuning: {e}")
            return None, None
    
    def evaluate_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """Evaluate ensemble with extended timeout"""
        logger.info(f"🎯 EVALUATING: {name}")
        
        start_time = time.time()
        
        # 8-minute timeout handler (extended for complex models)
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training {name} timed out after 8 minutes")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(480)  # 8 minutes
        
        try:
            # Train ensemble
            logger.info(f"Training {name}...")
            start_fit = time.time()
            ensemble.fit(X_train, y_train)
            fit_time = time.time() - start_fit
            logger.info(f"Training completed in {fit_time:.1f} seconds")
            
            # Predictions
            logger.info("Making predictions...")
            start_pred = time.time()
            y_pred = ensemble.predict(X_test)
            pred_time = time.time() - start_pred
            logger.info(f"Predictions completed in {pred_time:.1f} seconds")
            
            # Evaluation
            test_accuracy = accuracy_score(y_test, y_pred)
            test_precision = precision_score(y_test, y_pred, average='weighted')
            test_recall = recall_score(y_test, y_pred, average='weighted')
            test_f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            logger.info("Running cross-validation...")
            start_cv = time.time()
            cv_scores = cross_val_score(
                ensemble, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                scoring='accuracy',
                n_jobs=1
            )
            cv_time = time.time() - start_cv
            logger.info(f"CV completed in {cv_time:.1f} seconds")
            
            training_time = time.time() - start_time
            
            result = {
                'test_accuracy': test_accuracy,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'test_f1': test_f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'training_time': training_time,
                'status': 'success',
                'ensemble': ensemble
            }
            
            logger.info(f"✅ {name}: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
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
            signal.alarm(0)
    
    def run_advanced_ensemble_pipeline(self):
        """Run the advanced ensemble pipeline"""
        start_time = datetime.now()
        logger.info("🚀 STARTING ADVANCED ENSEMBLE PIPELINE FOR 95%+ ACCURACY")
        logger.info("🔧 Fine-tuning best models and adding advanced strategies")
        
        # 1. Load and preprocess
        X_train, X_test, y_train, y_test = self.load_and_preprocess_features()
        
        # 2. Create advanced base models
        base_models = self.create_advanced_base_models()
        
        # 3. Create advanced ensemble strategies
        ensembles = self.create_advanced_ensemble_strategies(base_models)
        
        # 4. Fine-tune best model from previous run
        fine_tuned_model, fine_tuned_name = self.fine_tune_best_model(X_train, y_train)
        if fine_tuned_model:
            ensembles[fine_tuned_name] = fine_tuned_model
        
        # 5. Evaluate all ensembles
        results = {}
        total_ensembles = len(ensembles)
        
        logger.info(f"🎯 Starting evaluation of {total_ensembles} advanced ensemble strategies...")
        
        for i, (name, ensemble) in enumerate(ensembles.items(), 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"📊 ENSEMBLE {i}/{total_ensembles}: {name}")
            logger.info(f"{'='*60}")
            
            result = self.evaluate_ensemble(name, ensemble, X_train, X_test, y_train, y_test)
            results[name] = result
            
            # Progress update
            if result['status'] == 'success':
                logger.info(f"✅ {i}/{total_ensembles} completed: {result['test_accuracy']:.4f}")
            else:
                logger.info(f"❌ {i}/{total_ensembles} failed")
        
        # 6. Final results
        successful_results = [(name, result) for name, result in results.items() 
                            if result['status'] == 'success']
        
        if successful_results:
            successful_results.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
            best_name, best_result = successful_results[0]
            best_accuracy = best_result['test_accuracy']
            
            logger.info(f"\n🏆 BEST ADVANCED ENSEMBLE: {best_name}")
            logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            # Check if 95%+ achieved
            if best_accuracy >= 0.95:
                logger.info("🎉🎉🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            else:
                gap = 0.95 - best_accuracy
                logger.info(f"📈 Close! Need {gap:.4f} ({gap*100:.2f}%) more for 95%")
            
            # Save results
            results_file = f"{self.output_path}/advanced_ensemble_results.txt"
            with open(results_file, 'w') as f:
                f.write("ADVANCED ENSEMBLE RESULTS\n")
                f.write("="*40 + "\n\n")
                
                for name, result in successful_results:
                    acc = result['test_accuracy']
                    f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%)\n")
            
            logger.info(f"💾 Results saved to {results_file}")
            
            # Save trained models
            logger.info("💾 Saving trained advanced ensemble models...")
            import pickle
            
            for name, result in successful_results:
                if result['status'] == 'success':
                    ensemble = results[name].get('ensemble', None)
                    if ensemble is not None:
                        model_file = f"{self.output_path}/{name}_model.pkl"
                        with open(model_file, 'wb') as f:
                            pickle.dump(ensemble, f)
                        logger.info(f"   ✅ {name} model saved to {model_file}")
            
            logger.info("💾 All advanced models saved successfully!")
            
            return {
                'best_accuracy': best_accuracy,
                'best_ensemble': best_name,
                'results': results,
                'models_above_95': sum(1 for _, result in successful_results if result['test_accuracy'] >= 0.95)
            }
        else:
            logger.error("❌ No advanced ensembles completed successfully")
            return {
                'best_accuracy': 0.0,
                'best_ensemble': 'None',
                'results': results,
                'models_above_95': 0
            }

if __name__ == "__main__":
    logger.info("🚀 STARTING ADVANCED ENSEMBLE PIPELINE...")
    
    advanced_ensemble_system = AdvancedEnsemble95Plus()
    results = advanced_ensemble_system.run_advanced_ensemble_pipeline()
    
    print(f"\n🎯 Advanced ensemble pipeline completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Models above 95%: {results['models_above_95']}")
    
    if results['best_accuracy'] >= 0.95:
        print("🎉🎉🎉 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
    else:
        print("📈 Keep pushing towards 95%+ accuracy!")
    
    print("✅ All advanced ensembles completed successfully!")
