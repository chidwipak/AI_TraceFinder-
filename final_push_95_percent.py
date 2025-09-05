#!/usr/bin/env python3
"""
FINAL PUSH TO 95%+ ACCURACY
Building on 90.87% LightGBM success with advanced ensemble and optimization
"""

import os
import time
import logging
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder, RobustScaler, QuantileTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE, SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
import pickle

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

class FinalPush95Percent:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "final_push_95_results"
        
        # Create output directories
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()
        
    def load_and_engineer_ultimate_features(self):
        """Load features and perform ultimate feature engineering"""
        logger.info("="*80)
        logger.info("🚀 FINAL PUSH - ULTIMATE FEATURE ENGINEERING")
        logger.info("Building on 90.87% success to reach 95%+")
        logger.info("="*80)
        
        # Load features
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)-5} features")
        
        # Create stratified split
        df['split'] = 'unknown'
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
        
        # Advanced feature engineering
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
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
        
        # Advanced missing value handling
        logger.info("Advanced missing value imputation...")
        imputer = KNNImputer(n_neighbors=7, weights='distance')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Multiple feature selection methods
        logger.info("Multiple feature selection approaches...")
        
        # 1. Mutual information
        mi_selector = SelectKBest(mutual_info_classif, k=min(120, X_train_imputed.shape[1]))
        X_train_mi = mi_selector.fit_transform(X_train_imputed, y_train_encoded)
        X_test_mi = mi_selector.transform(X_test_imputed)
        
        # 2. Random Forest feature importance
        rf_selector = RandomForestClassifier(n_estimators=200, random_state=self.random_seed, n_jobs=-1)
        rf_selector.fit(X_train_mi, y_train_encoded)
        
        # Select top features based on importance
        importances = rf_selector.feature_importances_
        top_indices = np.argsort(importances)[-100:]  # Top 100 features
        
        X_train_selected = X_train_mi[:, top_indices]
        X_test_selected = X_test_mi[:, top_indices]
        
        # 3. Advanced scaling with QuantileTransformer
        logger.info("Advanced feature scaling...")
        scaler = QuantileTransformer(n_quantiles=min(1000, X_train_selected.shape[0]), 
                                   output_distribution='normal', 
                                   random_state=self.random_seed)
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # 4. Feature interactions (top features only)
        logger.info("Creating feature interactions...")
        n_top = min(20, X_train_scaled.shape[1])
        top_feature_indices = np.argsort(importances[top_indices])[-n_top:]
        
        # Create polynomial features for top features
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_interactions = poly.fit_transform(X_train_scaled[:, top_feature_indices])
        X_test_interactions = poly.transform(X_test_scaled[:, top_feature_indices])
        
        # Combine original features with interactions
        X_train_final = np.hstack([X_train_scaled, X_train_interactions])
        X_test_final = np.hstack([X_test_scaled, X_test_interactions])
        
        logger.info(f"Final feature dimensions: {X_train_final.shape[1]}")
        logger.info(f"Training set: {X_train_final.shape}")
        logger.info(f"Test set: {X_test_final.shape}")
        
        return X_train_final, X_test_final, y_train_encoded, y_test_encoded
    
    def create_ultimate_ensemble_models(self):
        """Create the ultimate ensemble of models for 95%+"""
        logger.info("Creating ULTIMATE ensemble models...")
        
        models = {}
        
        # 1. Hypertuned LightGBM (our best performer at 90.87%)
        models['LightGBM_Ultimate'] = lgb.LGBMClassifier(
            n_estimators=3000,
            max_depth=15,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_samples=5,
            min_child_weight=0.001,
            random_state=self.random_seed,
            n_jobs=-1,
            verbose=-1
        )
        
        # 2. Hypertuned XGBoost
        models['XGBoost_Ultimate'] = xgb.XGBClassifier(
            n_estimators=3000,
            max_depth=12,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=0.1,
            min_child_weight=3,
            gamma=0.1,
            random_state=self.random_seed,
            n_jobs=-1,
            eval_metric='mlogloss'
        )
        
        # 3. Advanced Gradient Boosting
        models['GradientBoosting_Ultimate'] = GradientBoostingClassifier(
            n_estimators=2000,
            max_depth=12,
            learning_rate=0.02,
            subsample=0.85,
            max_features='sqrt',
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_seed
        )
        
        # 4. Extra Trees with high randomization
        models['ExtraTrees_Ultimate'] = ExtraTreesClassifier(
            n_estimators=2000,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            max_features='sqrt',
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # 5. Neural Network
        models['MLP_Ultimate'] = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.random_seed
        )
        
        # 6. Support Vector Machine
        models['SVM_Ultimate'] = SVC(
            kernel='rbf',
            C=10,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.random_seed
        )
        
        return models
    
    def hyperparameter_fine_tuning(self, models, X_train, y_train):
        """Fine-tune hyperparameters for best models"""
        logger.info("Fine-tuning hyperparameters for ultimate performance...")
        
        tuned_models = {}
        
        # Fine-tune LightGBM (our current best)
        logger.info("Fine-tuning LightGBM...")
        lgb_params = {
            'n_estimators': [2000, 3000, 4000],
            'max_depth': [12, 15, 18],
            'learning_rate': [0.02, 0.03, 0.05],
            'subsample': [0.8, 0.85, 0.9]
        }
        
        lgb_grid = GridSearchCV(
            lgb.LGBMClassifier(random_state=self.random_seed, n_jobs=-1, verbose=-1),
            lgb_params,
            cv=3,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        lgb_grid.fit(X_train, y_train)
        tuned_models['LightGBM_Tuned'] = lgb_grid.best_estimator_
        logger.info(f"Best LightGBM params: {lgb_grid.best_params_}")
        logger.info(f"Best LightGBM CV score: {lgb_grid.best_score_:.4f}")
        
        # Use pre-configured versions for other models to save time
        tuned_models.update({
            'XGBoost_Ultimate': models['XGBoost_Ultimate'],
            'GradientBoosting_Ultimate': models['GradientBoosting_Ultimate'],
            'ExtraTrees_Ultimate': models['ExtraTrees_Ultimate']
        })
        
        return tuned_models
    
    def create_meta_ensemble(self, base_models, X_train, y_train):
        """Create meta-ensemble using stacking"""
        logger.info("Creating meta-ensemble with stacking...")
        
        from sklearn.ensemble import StackingClassifier
        
        # Base models for stacking
        estimators = [
            ('lgb', base_models['LightGBM_Tuned']),
            ('xgb', base_models['XGBoost_Ultimate']),
            ('gb', base_models['GradientBoosting_Ultimate']),
            ('et', base_models['ExtraTrees_Ultimate'])
        ]
        
        # Meta-learner
        meta_learner = LogisticRegression(
            C=1.0,
            class_weight='balanced',
            random_state=self.random_seed,
            max_iter=1000
        )
        
        # Create stacking ensemble
        stacking_ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Also create voting ensemble
        voting_ensemble = VotingClassifier(
            estimators=estimators,
            voting='soft',
            n_jobs=-1
        )
        
        return {
            'Stacking_Ensemble': stacking_ensemble,
            'Voting_Ensemble': voting_ensemble
        }
    
    def ultimate_evaluation(self, models, X_train, X_test, y_train, y_test):
        """Ultimate evaluation targeting 95%+"""
        logger.info("="*80)
        logger.info("🎯 ULTIMATE EVALUATION FOR 95%+ ACCURACY")
        logger.info("="*80)
        
        results = {}
        
        for name, model in models.items():
            start_time = time.time()
            logger.info(f"🚀 Evaluating {name}...")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Robust cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train,
                    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=self.random_seed),
                    scoring='accuracy',
                    n_jobs=-1
                )
                
                elapsed = time.time() - start_time
                
                results[name] = {
                    'test_accuracy': accuracy,
                    'cv_mean': np.mean(cv_scores),
                    'cv_std': np.std(cv_scores),
                    'training_time': elapsed,
                    'y_pred': y_pred,
                    'model': model
                }
                
                logger.info(f"✅ {name}: {accuracy:.4f} ({accuracy*100:.2f}%) [CV: {np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}] in {elapsed:.1f}s")
                
                if accuracy >= 0.95:
                    logger.info(f"🎉🎉🎉 {name} ACHIEVED 95%+ ACCURACY! 🎉🎉🎉")
                
            except Exception as e:
                logger.error(f"❌ {name} failed: {e}")
                results[name] = {'test_accuracy': 0.0, 'error': str(e)}
        
        return results
    
    def create_final_visualizations(self, results):
        """Create comprehensive final visualizations"""
        logger.info("Creating final visualizations...")
        
        # Filter successful results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        
        if not successful_results:
            logger.error("No successful results to visualize")
            return
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        models = list(successful_results.keys())
        accuracies = [successful_results[m]['test_accuracy'] for m in models]
        cv_means = [successful_results[m]['cv_mean'] for m in models]
        cv_stds = [successful_results[m]['cv_std'] for m in models]
        
        # Test accuracies
        bars = axes[0, 0].bar(models, accuracies, 
                             color=['gold' if acc >= 0.95 else 'lightgreen' if acc >= 0.90 else 'skyblue' 
                                   for acc in accuracies])
        axes[0, 0].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 0].set_title('Test Accuracy - Final Push to 95%+', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Add accuracy labels
        for bar, acc in zip(bars, accuracies):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003, 
                           f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Cross-validation scores
        axes[0, 1].bar(models, cv_means, yerr=cv_stds, capsize=5, 
                      color='lightcoral', alpha=0.8)
        axes[0, 1].axhline(y=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[0, 1].set_title('Cross-Validation Accuracy', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('CV Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Best model confusion matrix
        best_model_name = max(successful_results.keys(), 
                             key=lambda x: successful_results[x]['test_accuracy'])
        best_y_pred = successful_results[best_model_name]['y_pred']
        best_y_test = [i for i in range(len(best_y_pred))]  # Simplified
        
        # Progress chart
        sorted_models = sorted(successful_results.items(), 
                              key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        axes[1, 0].barh([name for name, _ in sorted_models], 
                       [result['test_accuracy'] for _, result in sorted_models],
                       color='orange', alpha=0.8)
        axes[1, 0].axvline(x=0.95, color='red', linestyle='--', linewidth=2, label='95% Target')
        axes[1, 0].set_title('Model Ranking', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Accuracy')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Summary stats
        axes[1, 1].axis('off')
        best_acc = max(accuracies)
        summary_text = f"""
FINAL PUSH RESULTS

🎯 TARGET: 95%+ Accuracy
🥇 BEST: {best_model_name}
📊 ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)

{'🎉 TARGET ACHIEVED!' if best_acc >= 0.95 else f'Need {0.95-best_acc:.4f} more'}

TOP PERFORMERS:
"""
        for name, acc in sorted(zip(models, accuracies), key=lambda x: x[1], reverse=True)[:3]:
            summary_text += f"• {name}: {acc:.3f}\n"
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/final_push_results.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_final_push_pipeline(self):
        """Run the complete final push pipeline"""
        logger.info("="*80)
        logger.info("🚀 FINAL PUSH TO 95%+ ACCURACY")
        logger.info("Building on 90.87% LightGBM success!")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Ultimate feature engineering
        data = self.load_and_engineer_ultimate_features()
        if data is None:
            return
        
        X_train, X_test, y_train, y_test = data
        
        # 2. Create ultimate models
        base_models = self.create_ultimate_ensemble_models()
        
        # 3. Hyperparameter fine-tuning
        tuned_models = self.hyperparameter_fine_tuning(base_models, X_train, y_train)
        
        # 4. Create meta-ensemble
        ensemble_models = self.create_meta_ensemble(tuned_models, X_train, y_train)
        
        # 5. Combine all models
        all_models = {**tuned_models, **ensemble_models}
        
        # 6. Ultimate evaluation
        results = self.ultimate_evaluation(all_models, X_train, X_test, y_train, y_test)
        
        # 7. Create visualizations
        self.create_final_visualizations(results)
        
        # 8. Final report
        end_time = datetime.now()
        duration = end_time - start_time
        
        # Sort results
        successful_results = {k: v for k, v in results.items() if 'error' not in v}
        sorted_results = sorted(successful_results.items(), 
                              key=lambda x: x[1]['test_accuracy'], reverse=True)
        
        print("\n" + "="*80)
        print("🏆 FINAL PUSH RESULTS")
        print("="*80)
        
        for name, result in sorted_results:
            acc = result['test_accuracy']
            cv = result['cv_mean']
            print(f"🎯 {name}: {acc:.4f} ({acc*100:.2f}%) [CV: {cv:.4f}]")
        
        if sorted_results:
            best_name, best_result = sorted_results[0]
            best_acc = best_result['test_accuracy']
            
            print(f"\n🥇 CHAMPION: {best_name}")
            print(f"🎯 ACCURACY: {best_acc:.4f} ({best_acc*100:.2f}%)")
            print(f"⚡ DURATION: {duration}")
            
            models_95_plus = [name for name, result in sorted_results 
                             if result['test_accuracy'] >= 0.95]
            
            if models_95_plus:
                print(f"\n🎉🎉🎉 SUCCESS! {len(models_95_plus)} MODEL(S) ACHIEVED 95%+! 🎉🎉🎉")
                print("✅ TARGET REACHED!")
                for model in models_95_plus:
                    acc = successful_results[model]['test_accuracy']
                    print(f"🏅 {model}: {acc:.4f} ({acc*100:.2f}%)")
            elif best_acc >= 0.92:
                gap = 0.95 - best_acc
                print(f"\n🔥 SO CLOSE! Need only {gap:.4f} ({gap*100:.2f}%) more!")
                print("💪 Almost there!")
            else:
                gap = 0.95 - best_acc
                print(f"\n📈 Good progress! Need {gap:.4f} ({gap*100:.2f}%) more")
        
        # Save comprehensive results
        with open(f"{self.output_path}/results/final_push_summary.txt", 'w') as f:
            f.write("FINAL PUSH TO 95%+ ACCURACY RESULTS\n")
            f.write("="*50 + "\n\n")
            f.write(f"Duration: {duration}\n")
            f.write(f"Models evaluated: {len(results)}\n")
            f.write(f"Successful models: {len(successful_results)}\n\n")
            
            f.write("RESULTS:\n")
            for name, result in sorted_results:
                acc = result['test_accuracy']
                cv = result['cv_mean']
                f.write(f"{name}: {acc:.4f} ({acc*100:.2f}%) [CV: {cv:.4f}]\n")
            
            if sorted_results:
                best_acc = sorted_results[0][1]['test_accuracy']
                f.write(f"\nBest accuracy: {best_acc:.4f}\n")
                f.write(f"Target achieved: {'Yes' if best_acc >= 0.95 else 'No'}\n")
        
        logger.info("Final push pipeline complete!")
        
        return {
            'best_accuracy': sorted_results[0][1]['test_accuracy'] if sorted_results else 0.0,
            'best_model': sorted_results[0][0] if sorted_results else 'None',
            'target_achieved': sorted_results[0][1]['test_accuracy'] >= 0.95 if sorted_results else False,
            'duration': duration,
            'all_results': results
        }

if __name__ == "__main__":
    logger.info("🚀 FINAL PUSH TO 95%+ ACCURACY!")
    logger.info("Building on 90.87% LightGBM success...")
    
    final_push = FinalPush95Percent()
    results = final_push.run_final_push_pipeline()
    
    print(f"\n🚀 Final push completed!")
    print(f"Best accuracy: {results['best_accuracy']:.4f}")
    print(f"Target achieved: {results['target_achieved']}")
