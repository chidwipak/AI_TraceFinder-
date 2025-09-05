#!/usr/bin/env python3
"""
GPU-OPTIMIZED ENSEMBLE METHOD FOR 95%+ ACCURACY
Using PyTorch, CuPy, and RAPIDS for maximum GPU acceleration
"""

import os
import sys
import time
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# GPU Libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import cupy as cp
import cuml
from cuml.ensemble import RandomForestClassifier as cuRFRandomForestClassifier
from cuml.ensemble import RandomForestClassifier as cuRF
from cuml.linear_model import LogisticRegression as cuLogisticRegression
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.model_selection import train_test_split as cu_train_test_split
from cuml.metrics import accuracy_score as cu_accuracy_score
from cuml.feature_selection import SelectKBest as cuSelectKBest
from cuml.impute import SimpleImputer as cuSimpleImputer

# ML libraries
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import pickle

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GPUEnsemble95Plus:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ensemble_95_results_gpu"
        
        # Create output directory
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gpu_count = torch.cuda.device_count()
        
        logger.info(f"🚀 GPU Setup: {self.device} with {self.gpu_count} GPUs")
        if torch.cuda.is_available():
            for i in range(self.gpu_count):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        self.label_encoder = LabelEncoder()
        
    def load_features(self):
        """Load and prepare features"""
        logger.info("="*80)
        logger.info("🚀 LOADING FEATURES FOR GPU ENSEMBLE 95%+ ACCURACY")
        logger.info("="*80)
        
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Create consistent stratified split
        train_idx, test_idx = cu_train_test_split(
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
        """Prepare features with GPU acceleration"""
        logger.info("🔧 GPU-ACCELERATED FEATURE ENGINEERING")
        
        # Separate features from metadata
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Convert to GPU arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y)
        
        # Handle missing values with GPU
        logger.info("GPU imputing missing values...")
        imputer = cuSimpleImputer(strategy='mean')
        X_gpu = imputer.fit_transform(X_gpu)
        
        # Split data
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train = X_gpu[train_mask]
        X_test = X_gpu[test_mask]
        y_train = y_gpu[train_mask]
        y_test = y_gpu[test_mask]
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(cp.asnumpy(y_train))
        y_test_encoded = self.label_encoder.transform(cp.asnumpy(y_test))
        
        # Feature selection with GPU
        logger.info("GPU-accelerated feature selection...")
        selector = cuSelectKBest(k=min(120, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train, y_train_encoded)
        X_test_selected = selector.transform(X_test)
        
        # Scale features with GPU
        scaler = cuStandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        logger.info(f"Final feature dimensions: {X_train_scaled.shape[1]}")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        logger.info(f"Number of classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def create_gpu_base_models(self):
        """Create GPU-accelerated base models"""
        logger.info("🏗️  CREATING GPU-ACCELERATED BASE MODELS")
        
        base_models = {}
        
        # 1. XGBoost with GPU support
        base_models['XGBoost_GPU'] = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=15,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.75,
            reg_alpha=0.05,
            reg_lambda=0.05,
            tree_method='gpu_hist',  # GPU acceleration
            gpu_id=0,
            random_state=self.random_seed,
            eval_metric='mlogloss'
        )
        
        # 2. LightGBM with GPU support
        base_models['LightGBM_GPU'] = lgb.LGBMClassifier(
            n_estimators=1500,
            max_depth=10,
            learning_rate=0.06,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            device='gpu',  # GPU acceleration
            gpu_platform_id=0,
            gpu_device_id=0,
            random_state=self.random_seed,
            verbose=-1
        )
        
        # 3. RAPIDS Random Forest (GPU-native)
        base_models['RandomForest_GPU'] = cuRF(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 4. RAPIDS Extra Trees (GPU-native)
        base_models['ExtraTrees_GPU'] = cuRF(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 5. PyTorch Neural Network (GPU)
        base_models['NeuralNet_GPU'] = self.create_neural_network()
        
        return base_models
    
    def create_neural_network(self):
        """Create a PyTorch neural network for GPU"""
        class NeuralNet(nn.Module):
            def __init__(self, input_size, num_classes):
                super(NeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_size, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, num_classes)
                self.dropout = nn.Dropout(0.3)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.relu(self.fc3(x))
                x = self.dropout(x)
                x = self.fc4(x)
                return x
        
        return NeuralNet
    
    def create_gpu_ensemble_strategies(self, base_models, X_train, y_train, num_classes):
        """Create GPU-optimized ensemble strategies"""
        logger.info("🎯 CREATING GPU-OPTIMIZED ENSEMBLE STRATEGIES")
        
        ensembles = {}
        
        # Strategy 1: GPU Voting Classifier
        top_3_models = [
            ('xgb_gpu', base_models['XGBoost_GPU']),
            ('lgb_gpu', base_models['LightGBM_GPU']),
            ('rf_gpu', base_models['RandomForest_GPU'])
        ]
        
        # Strategy 2: GPU Stacking with RAPIDS Logistic Regression
        meta_learner = cuLogisticRegression(random_state=self.random_seed, max_iter=1000)
        
        # Strategy 3: PyTorch Neural Ensemble
        neural_ensemble = self.create_neural_ensemble(X_train, y_train, num_classes)
        ensembles['Neural_Ensemble_GPU'] = neural_ensemble
        
        return ensembles
    
    def create_neural_ensemble(self, X_train, y_train, num_classes):
        """Create a neural ensemble using PyTorch"""
        logger.info("🧠 Creating PyTorch Neural Ensemble...")
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(cp.asnumpy(X_train)).to(self.device)
        y_train_tensor = torch.LongTensor(y_train).to(self.device)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_train_tensor, y_train_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
        
        # Create model
        model = self.create_neural_network()(X_train.shape[1], num_classes).to(self.device)
        
        # Training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(50):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return model
    
    def evaluate_gpu_ensemble(self, name, ensemble, X_train, X_test, y_train, y_test):
        """Evaluate GPU ensemble performance"""
        logger.info(f"="*70)
        logger.info(f"🎯 EVALUATING GPU ENSEMBLE: {name}")
        logger.info(f"="*70)
        
        start_time = time.time()
        
        try:
            if 'Neural' in name:
                # PyTorch neural network evaluation
                model = ensemble
                model.eval()
                
                X_test_tensor = torch.FloatTensor(cp.asnumpy(X_test)).to(self.device)
                with torch.no_grad():
                    outputs = model(X_test_tensor)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred = predicted.cpu().numpy()
                    y_pred_proba = torch.softmax(outputs, dim=1).cpu().numpy()
                
                accuracy = cu_accuracy_score(y_test, y_pred)
                
            else:
                # Traditional ML models
                if hasattr(ensemble, 'fit'):
                    ensemble.fit(X_train, y_train)
                
                y_pred = ensemble.predict(X_test)
                y_pred_proba = ensemble.predict_proba(X_test) if hasattr(ensemble, 'predict_proba') else None
                
                accuracy = cu_accuracy_score(y_test, y_pred)
            
            # Additional metrics
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            training_time = time.time() - start_time
            
            result = {
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'training_time': training_time,
                'status': 'success'
            }
            
            logger.info(f"✅ {name} RESULTS:")
            logger.info(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            logger.info(f"  Training Time: {training_time:.1f}s")
            
            # Check for 95%
            if accuracy >= 0.95:
                logger.info(f"🎉 {name} ACHIEVED 95%+ ACCURACY! 🎉")
            
            # Save ensemble
            model_path = f"{self.output_path}/models/{name.lower()}_gpu_ensemble.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(ensemble, f)
            
            return result
            
        except Exception as e:
            logger.error(f"❌ {name} FAILED: {e}")
            return {
                'test_accuracy': 0.0,
                'test_precision': 0.0,
                'test_recall': 0.0,
                'test_f1': 0.0,
                'training_time': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
    
    def run_gpu_ensemble_pipeline(self):
        """Run the complete GPU ensemble pipeline"""
        logger.info("="*80)
        logger.info("🎯 GPU ENSEMBLE PIPELINE FOR 95%+ ACCURACY")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Load features
        df = self.load_features()
        if df is None:
            return
        
        # 2. Prepare features
        X_train, X_test, y_train, y_test = self.prepare_features(df)
        
        # 3. Create GPU base models
        base_models = self.create_gpu_base_models()
        
        # 4. Create GPU ensemble strategies
        ensembles = self.create_gpu_ensemble_strategies(base_models, X_train, y_train, len(self.label_encoder.classes_))
        
        # 5. Evaluate ensembles
        results = {}
        
        for name, ensemble in ensembles.items():
            result = self.evaluate_gpu_ensemble(name, ensemble, X_train, X_test, y_train, y_test)
            results[name] = result
        
        # 6. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("\n" + "="*80)
        logger.info("🏆 FINAL GPU ENSEMBLE RESULTS")
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
            
            logger.info(f"\n🥇 BEST GPU ENSEMBLE: {best_name}")
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
        results_file = f"{self.output_path}/gpu_ensemble_results.txt"
        with open(results_file, 'w') as f:
            f.write("GPU ENSEMBLE 95%+ ACCURACY RESULTS\n")
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
    logger.info("🚀 STARTING GPU ENSEMBLE PIPELINE FOR 95%+ ACCURACY...")
    
    gpu_ensemble_system = GPUEnsemble95Plus()
    results = gpu_ensemble_system.run_gpu_ensemble_pipeline()
    
    print(f"\n🎯 GPU Ensemble pipeline completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Models above 95%: {results['models_above_95']}")
