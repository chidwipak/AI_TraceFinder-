#!/usr/bin/env python3
"""
New Objective 2: Comprehensive Tampered/Original Detection - Data Preprocessing
Creates comprehensive train/test splits using ALL available data sources with proper class balancing
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML preprocessing
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN

# Data augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveObjective2DataPreprocessor:
    def __init__(self, analysis_path="new_objective2_analysis", output_path="new_objective2_preprocessed"):
        self.analysis_path = analysis_path
        self.output_path = output_path
        self.inventory_path = f"{analysis_path}/new_objective2_complete_inventory.csv"
        self.target_size = (224, 224)
        
        # Create output directories
        self.create_output_directories()
        
        # Data containers
        self.inventory_df = None
        self.train_df = None
        self.test_df = None
        self.class_weights = None
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path,
            f"{self.output_path}/train",
            f"{self.output_path}/test",
            f"{self.output_path}/metadata",
            f"{self.output_path}/augmented",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/reports"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_comprehensive_inventory(self):
        """Load the comprehensive inventory from analysis phase"""
        logger.info("="*80)
        logger.info("LOADING COMPREHENSIVE DATA INVENTORY")
        logger.info("="*80)
        
        if not os.path.exists(self.inventory_path):
            raise FileNotFoundError(f"Inventory file not found: {self.inventory_path}")
        
        self.inventory_df = pd.read_csv(self.inventory_path)
        logger.info(f"Loaded inventory with {len(self.inventory_df)} samples")
        logger.info(f"  Original samples: {len(self.inventory_df[self.inventory_df['label'] == 0])}")
        logger.info(f"  Tampered samples: {len(self.inventory_df[self.inventory_df['label'] == 1])}")
        
        # Print detailed breakdown
        logger.info("\nDetailed Breakdown:")
        for category in self.inventory_df['category'].unique():
            count = len(self.inventory_df[self.inventory_df['category'] == category])
            logger.info(f"  {category}: {count}")
        
        return self.inventory_df
    
    def create_stratified_train_test_split(self, test_size=0.2, random_state=42):
        """Create stratified train-test split with multiple strategies"""
        logger.info("="*80)
        logger.info("CREATING STRATIFIED TRAIN-TEST SPLIT")
        logger.info("="*80)
        
        # Prepare features and labels
        X = self.inventory_df[['image_path', 'tamper_type', 'category', 'has_mask']].values
        y = self.inventory_df['label'].values
        
        # Create stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=y, shuffle=True
        )
        
        # Convert back to DataFrames
        train_indices = []
        test_indices = []
        
        for i, (train_row, train_label) in enumerate(zip(X_train, y_train)):
            # Find matching row in original dataframe
            mask = (self.inventory_df['image_path'] == train_row[0]) & \
                   (self.inventory_df['tamper_type'] == train_row[1]) & \
                   (self.inventory_df['category'] == train_row[2]) & \
                   (self.inventory_df['has_mask'] == train_row[3])
            train_indices.extend(self.inventory_df[mask].index.tolist())
        
        for i, (test_row, test_label) in enumerate(zip(X_test, y_test)):
            # Find matching row in original dataframe
            mask = (self.inventory_df['image_path'] == test_row[0]) & \
                   (self.inventory_df['tamper_type'] == test_row[1]) & \
                   (self.inventory_df['category'] == test_row[2]) & \
                   (self.inventory_df['has_mask'] == test_row[3])
            test_indices.extend(self.inventory_df[mask].index.tolist())
        
        self.train_df = self.inventory_df.iloc[train_indices].reset_index(drop=True)
        self.test_df = self.inventory_df.iloc[test_indices].reset_index(drop=True)
        
        logger.info(f"Train set: {len(self.train_df)} samples")
        logger.info(f"  Original: {len(self.train_df[self.train_df['label'] == 0])}")
        logger.info(f"  Tampered: {len(self.train_df[self.train_df['label'] == 1])}")
        
        logger.info(f"Test set: {len(self.test_df)} samples")
        logger.info(f"  Original: {len(self.test_df[self.test_df['label'] == 0])}")
        logger.info(f"  Tampered: {len(self.test_df[self.test_df['label'] == 1])}")
        
        # Save splits
        self.train_df.to_csv(f"{self.output_path}/metadata/train_split.csv", index=False)
        self.test_df.to_csv(f"{self.output_path}/metadata/test_split.csv", index=False)
        
        logger.info("Train-test splits saved!")
        
        return self.train_df, self.test_df
    
    def compute_class_weights(self):
        """Compute class weights for handling imbalance"""
        logger.info("="*80)
        logger.info("COMPUTING CLASS WEIGHTS")
        logger.info("="*80)
        
        y_train = self.train_df['label'].values
        unique_classes = np.unique(y_train)
        
        # Compute class weights
        self.class_weights = compute_class_weight(
            'balanced', classes=unique_classes, y=y_train
        )
        
        class_weight_dict = {int(k): float(v) for k, v in zip(unique_classes, self.class_weights)}
        
        logger.info(f"Class weights: {class_weight_dict}")
        
        # Save class weights
        with open(f"{self.output_path}/metadata/class_weights.json", 'w') as f:
            json.dump(class_weight_dict, f, indent=2)
        
        return class_weight_dict
    
    def create_data_augmentation_pipeline(self):
        """Create comprehensive data augmentation pipeline"""
        logger.info("="*80)
        logger.info("CREATING DATA AUGMENTATION PIPELINE")
        logger.info("="*80)
        
        # Define augmentation strategies for different classes
        augmentation_strategies = {
            'original': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussNoise(p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.CLAHE(p=0.2),
            ]),
            
            'tampered': A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.4),
                A.GaussNoise(p=0.3),
                A.Blur(blur_limit=3, p=0.3),
                A.CLAHE(p=0.3),
                A.Emboss(p=0.2),
                A.Sharpen(p=0.2),
            ])
        }
        
        # Save augmentation configurations
        with open(f"{self.output_path}/metadata/augmentation_config.json", 'w') as f:
            config = {
                'original': str(augmentation_strategies['original']),
                'tampered': str(augmentation_strategies['tampered'])
            }
            json.dump(config, f, indent=2)
        
        logger.info("Data augmentation pipeline created!")
        return augmentation_strategies
    
    def apply_class_balancing_techniques(self):
        """Apply various class balancing techniques"""
        logger.info("="*80)
        logger.info("APPLYING CLASS BALANCING TECHNIQUES")
        logger.info("="*80)
        
        # Prepare data for balancing (using image paths as features for now)
        # In real implementation, you'd use extracted features
        X_train_paths = self.train_df['image_path'].values
        y_train = self.train_df['label'].values
        
        # Create dummy features for balancing (in practice, use actual extracted features)
        X_train_dummy = np.random.rand(len(X_train_paths), 100)  # Placeholder
        
        balancing_results = {}
        
        # 1. SMOTE (Synthetic Minority Oversampling Technique)
        try:
            smote = SMOTE(random_state=42, k_neighbors=3)
            X_smote, y_smote = smote.fit_resample(X_train_dummy, y_train)
            balancing_results['SMOTE'] = {
                'original_count': len(y_train),
                'augmented_count': len(y_smote),
                'original_ratio': (y_train == 0).sum() / len(y_train),
                'augmented_ratio': (y_smote == 0).sum() / len(y_smote)
            }
            logger.info(f"SMOTE: {len(y_train)} -> {len(y_smote)} samples")
        except Exception as e:
            logger.warning(f"SMOTE failed: {e}")
        
        # 2. ADASYN (Adaptive Synthetic Sampling)
        try:
            adasyn = ADASYN(random_state=42, n_neighbors=3)
            X_adasyn, y_adasyn = adasyn.fit_resample(X_train_dummy, y_train)
            balancing_results['ADASYN'] = {
                'original_count': len(y_train),
                'augmented_count': len(y_adasyn),
                'original_ratio': (y_train == 0).sum() / len(y_train),
                'augmented_ratio': (y_adasyn == 0).sum() / len(y_adasyn)
            }
            logger.info(f"ADASYN: {len(y_train)} -> {len(y_adasyn)} samples")
        except Exception as e:
            logger.warning(f"ADASYN failed: {e}")
        
        # 3. BorderlineSMOTE
        try:
            borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=3)
            X_borderline, y_borderline = borderline_smote.fit_resample(X_train_dummy, y_train)
            balancing_results['BorderlineSMOTE'] = {
                'original_count': len(y_train),
                'augmented_count': len(y_borderline),
                'original_ratio': (y_train == 0).sum() / len(y_train),
                'augmented_ratio': (y_borderline == 0).sum() / len(y_borderline)
            }
            logger.info(f"BorderlineSMOTE: {len(y_train)} -> {len(y_borderline)} samples")
        except Exception as e:
            logger.warning(f"BorderlineSMOTE failed: {e}")
        
        # 4. SMOTETomek (SMOTE + Tomek Links)
        try:
            smote_tomek = SMOTETomek(random_state=42)
            X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train_dummy, y_train)
            balancing_results['SMOTETomek'] = {
                'original_count': len(y_train),
                'augmented_count': len(y_smote_tomek),
                'original_ratio': (y_train == 0).sum() / len(y_train),
                'augmented_ratio': (y_smote_tomek == 0).sum() / len(y_smote_tomek)
            }
            logger.info(f"SMOTETomek: {len(y_train)} -> {len(y_smote_tomek)} samples")
        except Exception as e:
            logger.warning(f"SMOTETomek failed: {e}")
        
        # Save balancing results
        with open(f"{self.output_path}/metadata/balancing_results.json", 'w') as f:
            json.dump(balancing_results, f, indent=2)
        
        logger.info("Class balancing techniques applied!")
        return balancing_results
    
    def create_cross_validation_splits(self, n_splits=5):
        """Create cross-validation splits"""
        logger.info("="*80)
        logger.info("CREATING CROSS-VALIDATION SPLITS")
        logger.info("="*80)
        
        # Prepare data
        X_train = self.train_df[['image_path', 'tamper_type', 'category']].values
        y_train = self.train_df['label'].values
        
        # Create stratified K-fold
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        cv_splits = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            cv_splits.append({
                'fold': fold,
                'train_indices': train_idx.tolist(),
                'val_indices': val_idx.tolist(),
                'train_size': len(train_idx),
                'val_size': len(val_idx)
            })
            logger.info(f"Fold {fold}: Train={len(train_idx)}, Val={len(val_idx)}")
        
        # Save CV splits
        with open(f"{self.output_path}/metadata/cv_splits.json", 'w') as f:
            json.dump(cv_splits, f, indent=2)
        
        logger.info(f"Cross-validation splits created ({n_splits} folds)!")
        return cv_splits
    
    def generate_preprocessing_report(self):
        """Generate comprehensive preprocessing report"""
        logger.info("="*80)
        logger.info("GENERATING PREPROCESSING REPORT")
        logger.info("="*80)
        
        # 1. Dataset Statistics
        self._plot_dataset_statistics()
        
        # 2. Train-Test Split Analysis
        self._plot_train_test_analysis()
        
        # 3. Class Distribution Analysis
        self._plot_class_distribution_analysis()
        
        # 4. Generate text report
        self._generate_preprocessing_text_report()
        
        logger.info("Preprocessing report generated!")
    
    def _plot_dataset_statistics(self):
        """Plot dataset statistics"""
        plt.figure(figsize=(15, 10))
        
        # Overall statistics
        plt.subplot(2, 3, 1)
        total_samples = len(self.inventory_df)
        original_samples = len(self.inventory_df[self.inventory_df['label'] == 0])
        tampered_samples = len(self.inventory_df[self.inventory_df['label'] == 1])
        
        plt.pie([original_samples, tampered_samples], 
                labels=['Original', 'Tampered'], 
                autopct='%1.1f%%', 
                colors=['lightblue', 'lightcoral'])
        plt.title(f'Overall Dataset\n({total_samples} samples)')
        
        # Train-Test split
        plt.subplot(2, 3, 2)
        train_size = len(self.train_df)
        test_size = len(self.test_df)
        plt.pie([train_size, test_size], 
                labels=['Train', 'Test'], 
                autopct='%1.1f%%', 
                colors=['lightgreen', 'lightyellow'])
        plt.title(f'Train-Test Split\n({train_size + test_size} samples)')
        
        # Category distribution
        plt.subplot(2, 3, 3)
        category_counts = self.inventory_df['category'].value_counts()
        plt.bar(range(len(category_counts)), category_counts.values)
        plt.xticks(range(len(category_counts)), category_counts.index, rotation=45)
        plt.title('Category Distribution')
        plt.ylabel('Count')
        
        # Tamper type distribution
        plt.subplot(2, 3, 4)
        tamper_counts = self.inventory_df['tamper_type'].value_counts()
        plt.pie(tamper_counts.values, labels=tamper_counts.index, autopct='%1.1f%%')
        plt.title('Tamper Type Distribution')
        
        # Mask availability
        plt.subplot(2, 3, 5)
        mask_counts = self.inventory_df['has_mask'].value_counts()
        plt.bar(['No Mask', 'Has Mask'], mask_counts.values, color=['lightgray', 'darkgreen'])
        plt.title('Mask Availability')
        plt.ylabel('Count')
        
        # File size distribution
        plt.subplot(2, 3, 6)
        plt.hist(self.inventory_df['file_size'], bins=20, alpha=0.7, edgecolor='black')
        plt.title('File Size Distribution')
        plt.xlabel('File Size (bytes)')
        plt.ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/dataset_statistics.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_train_test_analysis(self):
        """Plot train-test split analysis"""
        plt.figure(figsize=(12, 8))
        
        # Train set class distribution
        plt.subplot(2, 2, 1)
        train_counts = self.train_df['label'].value_counts()
        plt.pie(train_counts.values, labels=['Original', 'Tampered'], 
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title(f'Train Set Class Distribution\n({len(self.train_df)} samples)')
        
        # Test set class distribution
        plt.subplot(2, 2, 2)
        test_counts = self.test_df['label'].value_counts()
        plt.pie(test_counts.values, labels=['Original', 'Tampered'], 
                autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        plt.title(f'Test Set Class Distribution\n({len(self.test_df)} samples)')
        
        # Category distribution in train set
        plt.subplot(2, 2, 3)
        train_category_counts = self.train_df['category'].value_counts()
        plt.bar(range(len(train_category_counts)), train_category_counts.values)
        plt.xticks(range(len(train_category_counts)), train_category_counts.index, rotation=45)
        plt.title('Train Set Category Distribution')
        plt.ylabel('Count')
        
        # Category distribution in test set
        plt.subplot(2, 2, 4)
        test_category_counts = self.test_df['category'].value_counts()
        plt.bar(range(len(test_category_counts)), test_category_counts.values)
        plt.xticks(range(len(test_category_counts)), test_category_counts.index, rotation=45)
        plt.title('Test Set Category Distribution')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/train_test_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_class_distribution_analysis(self):
        """Plot class distribution analysis"""
        plt.figure(figsize=(12, 6))
        
        # Original vs Tampered by category
        plt.subplot(1, 2, 1)
        category_label_counts = self.inventory_df.groupby(['category', 'label']).size().unstack(fill_value=0)
        category_label_counts.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Class Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Tampered'])
        
        # Original vs Tampered by tamper type
        plt.subplot(1, 2, 2)
        tamper_label_counts = self.inventory_df.groupby(['tamper_type', 'label']).size().unstack(fill_value=0)
        tamper_label_counts.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Class Distribution by Tamper Type')
        plt.xlabel('Tamper Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Tampered'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/class_distribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_preprocessing_text_report(self):
        """Generate comprehensive preprocessing text report"""
        report_path = f"{self.output_path}/reports/preprocessing_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE OBJECTIVE 2 DATA PREPROCESSING REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Preprocessing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Samples: {len(self.inventory_df)}\n")
            f.write(f"Original Samples: {len(self.inventory_df[self.inventory_df['label'] == 0])}\n")
            f.write(f"Tampered Samples: {len(self.inventory_df[self.inventory_df['label'] == 1])}\n")
            f.write(f"Class Balance: {len(self.inventory_df[self.inventory_df['label'] == 0])/len(self.inventory_df)*100:.1f}% Original\n\n")
            
            # Train-Test Split
            f.write("TRAIN-TEST SPLIT\n")
            f.write("-"*40 + "\n")
            f.write(f"Train Set: {len(self.train_df)} samples\n")
            f.write(f"  Original: {len(self.train_df[self.train_df['label'] == 0])}\n")
            f.write(f"  Tampered: {len(self.train_df[self.train_df['label'] == 1])}\n")
            f.write(f"Test Set: {len(self.test_df)} samples\n")
            f.write(f"  Original: {len(self.test_df[self.test_df['label'] == 0])}\n")
            f.write(f"  Tampered: {len(self.test_df[self.test_df['label'] == 1])}\n\n")
            
            # Class Weights
            if self.class_weights is not None:
                f.write("CLASS WEIGHTS\n")
                f.write("-"*40 + "\n")
                f.write(f"Original (0): {self.class_weights[0]:.4f}\n")
                f.write(f"Tampered (1): {self.class_weights[1]:.4f}\n\n")
            
            # Category Breakdown
            f.write("CATEGORY BREAKDOWN\n")
            f.write("-"*40 + "\n")
            for category in self.inventory_df['category'].unique():
                count = len(self.inventory_df[self.inventory_df['category'] == category])
                f.write(f"{category}: {count} samples\n")
            f.write("\n")
            
            # Tamper Type Breakdown
            f.write("TAMPER TYPE BREAKDOWN\n")
            f.write("-"*40 + "\n")
            for tamper_type in self.inventory_df['tamper_type'].unique():
                count = len(self.inventory_df[self.inventory_df['tamper_type'] == tamper_type])
                f.write(f"{tamper_type}: {count} samples\n")
            f.write("\n")
            
            # Mask Analysis
            f.write("MASK ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Images with masks: {len(self.inventory_df[self.inventory_df['has_mask'] == True])}\n")
            f.write(f"Images without masks: {len(self.inventory_df[self.inventory_df['has_mask'] == False])}\n")
            f.write(f"Mask coverage: {len(self.inventory_df[self.inventory_df['has_mask'] == True])/len(self.inventory_df)*100:.1f}%\n\n")
            
            # Preprocessing Recommendations
            f.write("PREPROCESSING RECOMMENDATIONS\n")
            f.write("-"*40 + "\n")
            f.write("1. Use class weights to handle imbalance\n")
            f.write("2. Apply data augmentation for minority classes\n")
            f.write("3. Utilize binary masks for region-specific features\n")
            f.write("4. Implement cross-validation for robust evaluation\n")
            f.write("5. Use ensemble methods for better performance\n")
            f.write("6. Extract mask-aware features for tampered regions\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_preprocessing(self):
        """Run complete data preprocessing pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE OBJECTIVE 2 DATA PREPROCESSING")
        logger.info("="*80)
        
        # Step 1: Load comprehensive inventory
        self.load_comprehensive_inventory()
        
        # Step 2: Create stratified train-test split
        self.create_stratified_train_test_split()
        
        # Step 3: Compute class weights
        self.compute_class_weights()
        
        # Step 4: Create data augmentation pipeline
        self.create_data_augmentation_pipeline()
        
        # Step 5: Apply class balancing techniques
        self.apply_class_balancing_techniques()
        
        # Step 6: Create cross-validation splits
        self.create_cross_validation_splits()
        
        # Step 7: Generate preprocessing report
        self.generate_preprocessing_report()
        
        logger.info("✅ COMPREHENSIVE DATA PREPROCESSING COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        
        return self.train_df, self.test_df

def main():
    """Main function to run comprehensive data preprocessing"""
    preprocessor = ComprehensiveObjective2DataPreprocessor()
    train_df, test_df = preprocessor.run_complete_preprocessing()
    
    print("\n" + "="*80)
    print("PREPROCESSING SUMMARY")
    print("="*80)
    print(f"Train samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Train class balance: {len(train_df[train_df['label'] == 0])/len(train_df)*100:.1f}% Original")
    print(f"Test class balance: {len(test_df[test_df['label'] == 0])/len(test_df)*100:.1f}% Original")
    print("\nNext step: Run feature extraction with this preprocessed data!")

if __name__ == "__main__":
    main()
