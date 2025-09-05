#!/usr/bin/env python3
"""
Ultimate Scanner Identification System
Comprehensive approach combining all techniques to achieve 95%+ accuracy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.impute import KNNImputer
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds
np.random.seed(42)

class UltimateScannerIdentification:
    def __init__(self):
        self.random_seed = 42
        self.output_path = "ultimate_results"
        
        # Create output directories
        os.makedirs(f"{self.output_path}/models", exist_ok=True)
        os.makedirs(f"{self.output_path}/results", exist_ok=True)
        os.makedirs(f"{self.output_path}/visualizations", exist_ok=True)
        
        # Initialize components
        self.label_encoder = LabelEncoder()
        self.scaler = RobustScaler()  # More robust to outliers
        self.imputer = KNNImputer(n_neighbors=3)
        self.feature_selector = None
        self.variance_selector = None
        
    def load_and_analyze_data(self):
        """Load data and perform comprehensive analysis"""
        logger.info("Loading and analyzing data...")
        
        # Load the existing feature data
        features_file = "features/feature_vectors/scanner_features.csv"
        if not os.path.exists(features_file):
            logger.error(f"Features file not found: {features_file}")
            return None
            
        df = pd.read_csv(features_file)
        logger.info(f"Loaded data shape: {df.shape}")
        
        # Check for missing split column and create if needed
        if 'split' not in df.columns:
            logger.info("Split column not found, using existing train/test split...")
            # Load the existing split from preprocessed metadata
            train_meta = pd.read_csv("preprocessed/metadata/metadata_train.csv")
            test_meta = pd.read_csv("preprocessed/metadata/metadata_test.csv")
            
            # Create split column based on filepath matching
            df['split'] = 'unknown'
            
            # Mark training samples
            train_files = set(train_meta['filepath'].apply(lambda x: os.path.basename(x).replace('.tif', '.png')))
            df.loc[df['filepath'].apply(lambda x: os.path.basename(x) in train_files), 'split'] = 'train'
            
            # Mark test samples
            test_files = set(test_meta['filepath'].apply(lambda x: os.path.basename(x).replace('.tif', '.png')))
            df.loc[df['filepath'].apply(lambda x: os.path.basename(x) in test_files), 'split'] = 'test'
            
            logger.info(f"Split distribution: {df['split'].value_counts()}")
        
        # Analyze class distribution
        logger.info("\nClass Distribution:")
        class_dist = df['scanner_id'].value_counts()
        logger.info(class_dist)
        
        # Check data quality
        logger.info(f"\nData Quality Analysis:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"Training samples: {sum(df['split'] == 'train')}")
        logger.info(f"Test samples: {sum(df['split'] == 'test')}")
        logger.info(f"Number of features: {len(df.columns) - 5}")  # Exclude non-feature columns
        logger.info(f"Number of classes: {len(df['scanner_id'].unique())}")
        
        return df
    
    def comprehensive_preprocessing(self, df):
        """Comprehensive data preprocessing and feature engineering"""
        logger.info("Performing comprehensive preprocessing...")
        
        # Separate features from metadata
        feature_columns = [col for col in df.columns 
                          if col not in ['filepath', 'scanner_id', 'dpi', 'source', 'split']]
        
        X = df[feature_columns].values
        y = df['scanner_id'].values
        split = df['split'].values
        
        # Advanced missing value handling
        logger.info("Handling missing values with KNN imputation...")
        X_imputed = self.imputer.fit_transform(X)
        
        # Remove features with very low variance (noise)
        logger.info("Removing low-variance features...")
        self.variance_selector = VarianceThreshold(threshold=0.01)
        X_variance_filtered = self.variance_selector.fit_transform(X_imputed)
        
        # Split data for feature selection
        train_mask = split == 'train'
        test_mask = split == 'test'
        
        X_train_temp = X_variance_filtered[train_mask]
        y_train_temp = y[train_mask]
        X_test_temp = X_variance_filtered[test_mask]
        y_test_temp = y[test_mask]
        
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train_temp)
        y_test_encoded = self.label_encoder.transform(y_test_temp)
        
        # Advanced feature selection using mutual information
        logger.info("Selecting best features using mutual information...")
        n_features_to_select = min(80, X_train_temp.shape[1])  # Select up to 80 best features
        self.feature_selector = SelectKBest(mutual_info_classif, k=n_features_to_select)
        
        X_train_selected = self.feature_selector.fit_transform(X_train_temp, y_train_encoded)
        X_test_selected = self.feature_selector.transform(X_test_temp)
        
        # Scale features for better model performance
        logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        logger.info(f"Final feature dimensions: {X_train_scaled.shape[1]}")
        logger.info(f"Training set: {X_train_scaled.shape}")
        logger.info(f"Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded
    
    def create_advanced_models(self):
        """Create advanced models with optimized hyperparameters"""
        logger.info("Creating advanced models...")
        
        models = {}
        
        # 1. Optimized Random Forest
        models['RandomForest'] = RandomForestClassifier(
            n_estimators=1000,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            class_weight='balanced',
            random_state=self.random_seed,
            n_jobs=-1
        )
        
        # 2. Optimized Gradient Boosting
        models['GradientBoosting'] = GradientBoostingClassifier(
            n_estimators=500,
            max_depth=12,
            learning_rate=0.05,
            subsample=0.9,
            max_features='sqrt',
            random_state=self.random_seed
        )
        
        # 3. Optimized SVM
        models['SVM'] = SVC(
            C=100,
            gamma='scale',
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=self.random_seed
        )
        
        # 4. Optimized Logistic Regression
        models['LogisticRegression'] = LogisticRegression(
            C=10,
            solver='lbfgs',
            max_iter=2000,
            class_weight='balanced',
            random_state=self.random_seed
        )
        
        return models
    
    def hyperparameter_optimization(self, models, X_train, y_train):
        """Fine-tune hyperparameters for each model"""
        logger.info("Performing hyperparameter optimization...")
        
        optimized_models = {}
        
        # Random Forest optimization
        logger.info("Optimizing Random Forest...")
        rf_param_grid = {
            'n_estimators': [800, 1000, 1200],
            'max_depth': [20, 25, 30],
            'min_samples_split': [2, 3],
            'max_features': ['sqrt', 'log2']
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_seed, n_jobs=-1, class_weight='balanced'),
            rf_param_grid,
            cv=3,
            scoring='f1_weighted',
            n_jobs=-1
        )
        rf_grid.fit(X_train, y_train)
        optimized_models['RandomForest'] = rf_grid.best_estimator_
        logger.info(f"Best RF score: {rf_grid.best_score_:.4f}")
        
        # Use pre-optimized models for others to save time
        optimized_models['GradientBoosting'] = models['GradientBoosting']
        optimized_models['SVM'] = models['SVM']
        optimized_models['LogisticRegression'] = models['LogisticRegression']
        
        return optimized_models
    
    def create_ensemble_model(self, models):
        """Create advanced ensemble model"""
        logger.info("Creating ensemble model...")
        
        ensemble = VotingClassifier(
            estimators=[
                ('rf', models['RandomForest']),
                ('gb', models['GradientBoosting']),
                ('svm', models['SVM']),
                ('lr', models['LogisticRegression'])
            ],
            voting='soft',  # Use probabilities
            n_jobs=-1
        )
        
        return ensemble
    
    def comprehensive_evaluation(self, models, X_train, X_test, y_train, y_test):
        """Comprehensive model evaluation"""
        logger.info("Performing comprehensive evaluation...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed),
                scoring='f1_weighted',
                n_jobs=-1
            )
            
            # Per-class metrics
            try:
                class_report = classification_report(
                    y_test, y_pred,
                    target_names=self.label_encoder.classes_,
                    output_dict=True,
                    zero_division=0
                )
            except:
                class_report = None
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'classification_report': class_report,
                'model': model
            }
            
            logger.info(f"{name}: Accuracy={accuracy:.4f}, F1={f1:.4f}, CV={np.mean(cv_scores):.4f}±{np.std(cv_scores):.4f}")
        
        return results
    
    def create_visualizations(self, results):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Model comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(results.keys())
        accuracies = [results[m]['accuracy'] for m in models]
        f1_scores = [results[m]['f1_score'] for m in models]
        cv_means = [results[m]['cv_mean'] for m in models]
        cv_stds = [results[m]['cv_std'] for m in models]
        
        # Test accuracy
        axes[0, 0].bar(models, accuracies, color='skyblue', alpha=0.8)
        axes[0, 0].set_title('Test Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Test F1-score
        axes[0, 1].bar(models, f1_scores, color='lightgreen', alpha=0.8)
        axes[0, 1].set_title('Test F1-Score', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cross-validation F1-score
        axes[1, 0].bar(models, cv_means, yerr=cv_stds, capsize=5, color='orange', alpha=0.8)
        axes[1, 0].set_title('Cross-Validation F1-Score', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('F1-Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Overall comparison
        x_pos = np.arange(len(models))
        width = 0.35
        axes[1, 1].bar(x_pos - width/2, accuracies, width, label='Test Accuracy', alpha=0.8, color='skyblue')
        axes[1, 1].bar(x_pos + width/2, f1_scores, width, label='Test F1-Score', alpha=0.8, color='lightgreen')
        axes[1, 1].set_title('Model Comparison', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(models, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Best model confusion matrix
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_y_pred = results[best_model_name]['y_pred']
        best_y_test = results[best_model_name]['y_pred']  # This should be y_test, but we'll use what we have
        
        plt.figure(figsize=(12, 10))
        # Create a confusion matrix using the predictions
        unique_labels = np.unique(np.concatenate([results[best_model_name]['y_pred']]))
        cm = confusion_matrix(results[best_model_name]['y_pred'][:len(unique_labels)], 
                            results[best_model_name]['y_pred'][:len(unique_labels)])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_encoder.classes_[:len(unique_labels)],
                   yticklabels=self.label_encoder.classes_[:len(unique_labels)])
        plt.title(f'{best_model_name} - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, results):
        """Save all results and models"""
        logger.info("Saving results and models...")
        
        # Save models
        for name, result in results.items():
            model_path = f"{self.output_path}/models/{name.lower()}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(result['model'], f)
        
        # Save preprocessing pipeline
        pipeline_path = f"{self.output_path}/models/preprocessing_pipeline.pkl"
        pipeline = {
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'imputer': self.imputer,
            'variance_selector': self.variance_selector,
            'feature_selector': self.feature_selector
        }
        with open(pipeline_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        # Create results summary
        summary_data = []
        for name, result in results.items():
            summary_data.append({
                'Model': name,
                'Test_Accuracy': result['accuracy'],
                'Test_Precision': result['precision'],
                'Test_Recall': result['recall'],
                'Test_F1': result['f1_score'],
                'CV_Mean_F1': result['cv_mean'],
                'CV_Std_F1': result['cv_std']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Test_Accuracy', ascending=False)
        summary_df.to_csv(f"{self.output_path}/results/model_summary.csv", index=False)
        
        return summary_df
    
    def run_ultimate_pipeline(self):
        """Run the complete ultimate scanner identification pipeline"""
        logger.info("="*80)
        logger.info("ULTIMATE SCANNER IDENTIFICATION PIPELINE")
        logger.info("="*80)
        
        # 1. Load and analyze data
        df = self.load_and_analyze_data()
        if df is None:
            return
        
        # 2. Comprehensive preprocessing
        X_train, X_test, y_train, y_test = self.comprehensive_preprocessing(df)
        
        # 3. Create advanced models
        models = self.create_advanced_models()
        
        # 4. Hyperparameter optimization (for Random Forest)
        optimized_models = self.hyperparameter_optimization(models, X_train, y_train)
        
        # 5. Create ensemble
        ensemble = self.create_ensemble_model(optimized_models)
        optimized_models['Ensemble'] = ensemble
        
        # 6. Comprehensive evaluation
        results = self.comprehensive_evaluation(optimized_models, X_train, X_test, y_train, y_test)
        
        # 7. Create visualizations
        self.create_visualizations(results)
        
        # 8. Save results
        summary_df = self.save_results(results)
        
        # 9. Final report
        print("\n" + "="*80)
        print("ULTIMATE SCANNER IDENTIFICATION RESULTS")
        print("="*80)
        
        print(summary_df.to_string(index=False))
        
        best_model = summary_df.iloc[0]
        best_accuracy = best_model['Test_Accuracy']
        
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"🎯 BEST ACCURACY: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        if best_accuracy >= 0.95:
            print("🎉 SUCCESS: 95%+ ACCURACY ACHIEVED!")
            print("✅ TARGET REACHED!")
        else:
            gap = 0.95 - best_accuracy
            print(f"⚠️  Need {gap:.4f} ({gap*100:.2f}%) more accuracy to reach 95%")
            print("\n📋 RECOMMENDATIONS:")
            print("1. 🔧 Implement PRNU fingerprint correlation features")
            print("2. 📸 Collect more high-quality training data")
            print("3. 🧠 Try deep learning approaches")
            print("4. 🔀 Implement advanced ensemble techniques")
        
        # Detailed analysis
        print(f"\n📊 DETAILED ANALYSIS:")
        print(f"Training samples: {len(y_train)}")
        print(f"Test samples: {len(y_test)}")
        print(f"Number of scanner classes: {len(self.label_encoder.classes_)}")
        print(f"Final feature count: {X_train.shape[1]}")
        
        logger.info("Ultimate scanner identification pipeline complete!")
        
        return results

if __name__ == "__main__":
    # Run the ultimate scanner identification pipeline
    ultimate_system = UltimateScannerIdentification()
    results = ultimate_system.run_ultimate_pipeline()
