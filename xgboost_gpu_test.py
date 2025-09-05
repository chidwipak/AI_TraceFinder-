#!/usr/bin/env python3
"""
DEDICATED XGBOOST GPU TEST - TARGET 95%+ ACCURACY
Isolated testing of XGBoost GPU configurations to achieve maximum accuracy
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import KNNImputer
import pickle

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XGBoostGPUTester:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "xgboost_gpu_results"
        
        # Create output directory
        os.makedirs(self.output_path, exist_ok=True)
        
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def check_gpu_availability(self):
        """Check various GPU configurations for XGBoost"""
        logger.info("="*80)
        logger.info("🔍 CHECKING XGBOOST GPU AVAILABILITY")
        logger.info("="*80)
        
        gpu_configs = []
        
        # Check CUDA devices
        try:
            import subprocess
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("NVIDIA-SMI output:")
                logger.info(result.stdout)
        except:
            logger.warning("Could not run nvidia-smi")
        
        # Test different XGBoost GPU configurations
        test_configs = [
            {'name': 'CUDA:0', 'config': {'device': 'cuda:0', 'tree_method': 'gpu_hist'}},
            {'name': 'CUDA:1', 'config': {'device': 'cuda:1', 'tree_method': 'gpu_hist'}},
            {'name': 'CUDA:2', 'config': {'device': 'cuda:2', 'tree_method': 'gpu_hist'}},
            {'name': 'CUDA:3', 'config': {'device': 'cuda:3', 'tree_method': 'gpu_hist'}},
            {'name': 'Legacy GPU 0', 'config': {'gpu_id': 0, 'tree_method': 'gpu_hist'}},
            {'name': 'Legacy GPU 1', 'config': {'gpu_id': 1, 'tree_method': 'gpu_hist'}},
            {'name': 'Legacy GPU 2', 'config': {'gpu_id': 2, 'tree_method': 'gpu_hist'}},
            {'name': 'Legacy GPU 3', 'config': {'gpu_id': 3, 'tree_method': 'gpu_hist'}},
        ]
        
        # Create small test dataset
        X_test = np.random.rand(100, 10)
        y_test = np.random.randint(0, 3, 100)
        
        for test_config in test_configs:
            try:
                logger.info(f"Testing {test_config['name']}...")
                
                model = xgb.XGBClassifier(
                    n_estimators=10,  # Small for testing
                    max_depth=3,
                    random_state=self.random_seed,
                    **test_config['config']
                )
                
                model.fit(X_test, y_test)
                pred = model.predict(X_test)
                
                gpu_configs.append({
                    'name': test_config['name'],
                    'config': test_config['config'],
                    'status': 'SUCCESS'
                })
                
                logger.info(f"✅ {test_config['name']} - SUCCESS")
                
            except Exception as e:
                logger.warning(f"❌ {test_config['name']} - FAILED: {str(e)[:100]}")
                gpu_configs.append({
                    'name': test_config['name'],
                    'config': test_config['config'],
                    'status': 'FAILED',
                    'error': str(e)
                })
        
        # Find working configurations
        working_configs = [c for c in gpu_configs if c['status'] == 'SUCCESS']
        
        logger.info(f"\n🎯 WORKING GPU CONFIGURATIONS: {len(working_configs)}")
        for config in working_configs:
            logger.info(f"  ✅ {config['name']}")
        
        return working_configs
    
    def load_features(self):
        """Load scanner features"""
        logger.info("Loading scanner features...")
        
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Create stratified split
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
        """Prepare features for XGBoost"""
        logger.info("Preparing features for XGBoost...")
        
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
        
        # Feature selection
        logger.info("Selecting top features...")
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
    
    def create_xgboost_models(self, working_gpu_configs):
        """Create various XGBoost models for testing"""
        logger.info("Creating XGBoost models...")
        
        models = {}
        
        # GPU models (if available)
        for i, gpu_config in enumerate(working_gpu_configs[:2]):  # Test top 2 working configs
            config = gpu_config['config'].copy()
            
            models[f'XGBoost_GPU_{i+1}'] = xgb.XGBClassifier(
                n_estimators=2000,
                max_depth=12,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=self.random_seed,
                eval_metric='mlogloss',
                **config
            )
            
            # High-performance version
            models[f'XGBoost_GPU_{i+1}_Ultra'] = xgb.XGBClassifier(
                n_estimators=3000,
                max_depth=15,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.75,
                reg_alpha=0.05,
                reg_lambda=0.05,
                random_state=self.random_seed,
                eval_metric='mlogloss',
                **config
            )
        
        # CPU fallback models
        models['XGBoost_CPU_Fast'] = xgb.XGBClassifier(
            n_estimators=2000,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            tree_method='hist',
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        models['XGBoost_CPU_Ultra'] = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=15,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=0.05,
            tree_method='hist',
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # 95%+ target model
        models['XGBoost_95Plus_Target'] = xgb.XGBClassifier(
            n_estimators=4000,
            max_depth=18,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.7,
            reg_alpha=0.01,
            reg_lambda=0.01,
            tree_method='hist',
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        return models
    
    def evaluate_model(self, name, model, X_train, X_test, y_train, y_test):
        """Evaluate a single XGBoost model"""
        logger.info(f"="*60)
        logger.info(f"EVALUATING: {name}")
        logger.info(f"="*60)
        
        start_time = time.time()
        
        try:
            # Train model
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predictions
            logger.info(f"Making predictions...")
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Cross-validation
            logger.info(f"Running cross-validation...")
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                scoring='accuracy',
                n_jobs=2  # Conservative for GPU models
            )
            
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
            
            # Save model
            model_path = f"{self.output_path}/{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return result
            
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
    
    def run_xgboost_gpu_test(self):
        """Run comprehensive XGBoost GPU test"""
        logger.info("="*80)
        logger.info("🚀 XGBOOST GPU TEST - TARGET 95%+ ACCURACY")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Check GPU availability
        working_gpu_configs = self.check_gpu_availability()
        
        # 2. Load features
        df = self.load_features()
        if df is None:
            return
        
        # 3. Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(df)
        
        # 4. Create models
        models = self.create_xgboost_models(working_gpu_configs)
        
        # 5. Evaluate models
        results = {}
        
        for name, model in models.items():
            result = self.evaluate_model(name, model, X_train, X_test, y_train, y_test)
            results[name] = result
        
        # 6. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("🏆 XGBOOST GPU TEST RESULTS")
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
            
            logger.info(f"\n🥇 BEST MODEL: {best_name}")
            logger.info(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
            logger.info(f"⏱️  TOTAL TIME: {duration}")
            
            if best_accuracy >= 0.95:
                logger.info("🎉🎉🎉 SUCCESS! 95%+ ACCURACY ACHIEVED! 🎉🎉🎉")
            else:
                gap = 0.95 - best_accuracy
                logger.info(f"📈 Need {gap:.4f} ({gap*100:.2f}%) more for 95%")
        
        # Save results
        results_file = f"{self.output_path}/xgboost_gpu_test_results.txt"
        with open(results_file, 'w') as f:
            f.write("XGBOOST GPU TEST RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"GPU Configurations Tested: {len(working_gpu_configs)}\n")
            f.write(f"Models Evaluated: {len(models)}\n")
            f.write(f"Total Time: {duration}\n\n")
            
            for name, result in successful_results:
                acc = result['test_accuracy']
                f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%)\n")
        
        return {
            'best_accuracy': best_result['test_accuracy'] if successful_results else 0.0,
            'best_model': best_name if successful_results else 'None',
            'gpu_configs_working': len(working_gpu_configs),
            'duration': duration,
            'results': results
        }

if __name__ == "__main__":
    logger.info("🔥 STARTING XGBOOST GPU TEST...")
    
    tester = XGBoostGPUTester()
    results = tester.run_xgboost_gpu_test()
    
    print(f"\n🚀 XGBoost GPU test completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"GPU configs working: {results['gpu_configs_working']}")
