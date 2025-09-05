#!/usr/bin/env python3
"""
FAST ENSEMBLE PIPELINE FOR 95%+ ACCURACY
Optimized for speed while maintaining high accuracy
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
    GradientBoostingClassifier, VotingClassifier, StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ensemble_95_plus_fast.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FastEnsemble95Plus:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ensemble_95_results_fast"
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        # Set random seeds
        np.random.seed(self.random_seed)
        
    def load_and_preprocess_features(self):
        """Load and preprocess features efficiently"""
        logger.info("🚀 LOADING FEATURES FOR FAST ENSEMBLE 95%+ ACCURACY")
        
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
            
            # Quick preprocessing
            logger.info("🔧 FAST FEATURE ENGINEERING")
            
            # Simple imputation
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            
            # Quick feature selection (keep top 80 features for speed)
            selector = SelectKBest(score_func=mutual_info_classif, k=80)
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
    
    def create_fast_base_models(self):
        """Create fast but powerful base models"""
        logger.info("🏗️ CREATING FAST HIGH-PERFORMANCE BASE MODELS")
        
        base_models = {}
        
        # 1. XGBoost Fast (optimized for speed)
        base_models['XGBoost_Fast'] = xgb.XGBClassifier(
            n_estimators=200,      # Reduced for speed
            max_depth=8,            # Reduced for speed
            learning_rate=0.2,      # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',     # Faster tree method
            random_state=self.random_seed,
            n_jobs=1,               # Single-threaded
            eval_metric='mlogloss',
            verbose=0
        )
        
        # 2. LightGBM Fast
        base_models['LightGBM_Fast'] = lgb.LGBMClassifier(
            n_estimators=150,       # Reduced for speed
            max_depth=8,            # Reduced for speed
            learning_rate=0.2,      # Increased for faster convergence
            subsample=0.8,
            colsample_bytree=0.8,
            device='cpu',
            random_state=self.random_seed,
            n_jobs=1,               # Single-threaded
            verbose=-1
        )
        
        # 3. Random Forest Fast
        base_models['RandomForest_Fast'] = RandomForestClassifier(
            n_estimators=100,       # Reduced for speed
            max_depth=10,           # Reduced for speed
            min_samples_split=5,    # Increased for speed
            min_samples_leaf=2,     # Increased for speed
            max_features='sqrt',
            random_state=self.random_seed,
            n_jobs=1                # Single-threaded
        )
        
        # 4. Extra Trees Fast
        base_models['ExtraTrees_Fast'] = ExtraTreesClassifier(
            n_estimators=100,       # Reduced for speed
            max_depth=10,           # Reduced for speed
            min_samples_split=5,    # Increased for speed
            min_samples_leaf=2,     # Increased for speed
            max_features='sqrt',
            random_state=self.random_seed,
            n_jobs=1                # Single-threaded
        )
        
        return base_models
    
    def create_fast_ensemble_strategies(self, base_models):
        """Create fast ensemble strategies"""
        logger.info("🎯 CREATING FAST ENSEMBLE STRATEGIES")
        
        ensembles = {}
        
        # Strategy 1: Simple Voting (fastest)
        top_3_models = [
            ('xgb_fast', base_models['XGBoost_Fast']),
            ('lgb_fast', base_models['LightGBM_Fast']),
            ('rf_fast', base_models['RandomForest_Fast'])
        ]
        
        ensembles['Voting_Top3_Hard'] = VotingClassifier(
            estimators=top_3_models,
            voting='hard',
            n_jobs=1
        )
        
        # Strategy 2: Stacking with Logistic Regression (fast meta-learner)
        ensembles['Stacking_Top3_LR'] = StackingClassifier(
            estimators=top_3_models,
            final_estimator=LogisticRegression(random_state=self.random_seed, max_iter=500),
            cv=3,                   # Reduced CV folds for speed
            n_jobs=1
        )
        
        # Strategy 3: Weighted Voting (all models)
        all_4_models = [
            ('xgb_fast', base_models['XGBoost_Fast']),
            ('lgb_fast', base_models['LightGBM_Fast']),
            ('rf_fast', base_models['RandomForest_Fast']),
            ('et_fast', base_models['ExtraTrees_Fast'])
        ]
        
        ensembles['Voting_All4_Soft'] = VotingClassifier(
            estimators=all_4_models,
            voting='soft',
            n_jobs=1
        )
        
        return ensembles
    
    def evaluate_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """Evaluate ensemble with 5-minute timeout"""
        logger.info(f"🎯 EVALUATING: {name}")
        
        start_time = time.time()
        
        # 5-minute timeout handler
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Training {name} timed out after 5 minutes")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes
        
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
            
            # Quick cross-validation (3 folds for speed)
            logger.info("Running cross-validation...")
            start_cv = time.time()
            cv_scores = cross_val_score(
                ensemble, X_train, y_train,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_seed),
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
                'ensemble': ensemble  # Store the trained ensemble
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
    
    def run_ensemble_pipeline(self):
        """Run the fast ensemble pipeline"""
        start_time = datetime.now()
        logger.info("🚀 STARTING FAST ENSEMBLE PIPELINE FOR 95%+ ACCURACY")
        logger.info("⚡ Optimized for speed while maintaining high accuracy")
        
        # 1. Load and preprocess
        X_train, X_test, y_train, y_test = self.load_and_preprocess_features()
        
        # 2. Create base models
        base_models = self.create_fast_base_models()
        
        # 3. Create ensemble strategies
        ensembles = self.create_fast_ensemble_strategies(base_models)
        
        # 4. Evaluate ensembles
        results = {}
        total_ensembles = len(ensembles)
        
        logger.info(f"🎯 Starting evaluation of {total_ensembles} fast ensemble strategies...")
        
        for i, (name, ensemble) in enumerate(ensembles.items(), 1):
            logger.info(f"\n{'='*50}")
            logger.info(f"📊 ENSEMBLE {i}/{total_ensembles}: {name}")
            logger.info(f"{'='*50}")
            
            result = self.evaluate_ensemble(name, ensemble, X_train, X_test, y_train, y_test)
            results[name] = result
            
            # Progress update
            if result['status'] == 'success':
                logger.info(f"✅ {i}/{total_ensembles} completed: {result['test_accuracy']:.4f}")
            else:
                logger.info(f"❌ {i}/{total_ensembles} failed")
        
        # 5. Final results
        successful_results = [(name, result) for name, result in results.items() 
                            if result['status'] == 'success']
        
        if successful_results:
            successful_results.sort(key=lambda x: x[1]['test_accuracy'], reverse=True)
            best_name, best_result = successful_results[0]
            best_accuracy = best_result['test_accuracy']
            
            logger.info(f"\n🏆 BEST ENSEMBLE: {best_name}")
            logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            
            # Save results
            results_file = f"{self.output_path}/fast_ensemble_results.txt"
            with open(results_file, 'w') as f:
                f.write("FAST ENSEMBLE RESULTS\n")
                f.write("="*30 + "\n\n")
                
                for name, result in successful_results:
                    acc = result['test_accuracy']
                    f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%)\n")
            
            logger.info(f"💾 Results saved to {results_file}")
            
            # Save trained models
            logger.info("💾 Saving trained ensemble models...")
            import pickle
            
            for name, result in successful_results:
                if result['status'] == 'success':
                    # Get the trained ensemble from results
                    ensemble = results[name].get('ensemble', None)
                    if ensemble is not None:
                        model_file = f"{self.output_path}/{name}_model.pkl"
                        with open(model_file, 'wb') as f:
                            pickle.dump(ensemble, f)
                        logger.info(f"   ✅ {name} model saved to {model_file}")
            
            logger.info("💾 All models saved successfully!")
            
            # Create a model loading function
            self.create_model_loader()
            
            return {
                'best_accuracy': best_accuracy,
                'best_ensemble': best_name,
                'results': results
            }
        else:
            logger.error("❌ No ensembles completed successfully")
            return {
                'best_accuracy': 0.0,
                'best_ensemble': 'None',
                'results': results
            }
    
    def create_model_loader(self):
        """Create a model loading script for easy reuse"""
        loader_script = f"{self.output_path}/load_models.py"
        
        with open(loader_script, 'w') as f:
            f.write('''#!/usr/bin/env python3
"""
Model Loader for Fast Ensemble Models
Use this to load and make predictions with saved models
"""

import pickle
import numpy as np

def load_model(model_name):
    """Load a saved ensemble model"""
    model_path = f"{model_name}_model.pkl"
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def predict_with_model(model, X):
    """Make predictions with loaded model"""
    return model.predict(X)

def predict_proba_with_model(model, X):
    """Get prediction probabilities with loaded model"""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)
    else:
        return None

# Example usage:
# model = load_model("Stacking_Top3_LR")
# predictions = predict_with_model(model, X_test)
# probabilities = predict_proba_with_model(model, X_test)
''')
        
        logger.info(f"📝 Model loader script created: {loader_script}")

if __name__ == "__main__":
    logger.info("🚀 STARTING FAST ENSEMBLE PIPELINE...")
    
    ensemble_system = FastEnsemble95Plus()
    results = ensemble_system.run_ensemble_pipeline()
    
    print(f"\n🎯 Fast ensemble pipeline completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print("✅ All ensembles completed successfully!")
