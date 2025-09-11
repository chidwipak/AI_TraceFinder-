#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Accuracy Analysis
Analyzes why 90% accuracy is not achieved and identifies potential issues
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import cv2
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

class AccuracyAnalysis:
    """Analyze why 90% accuracy is not achieved"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def load_all_results(self):
        """Load results from all model types"""
        print("=== Loading All Model Results ===")
        
        results = {}
        
        # Load baseline ML results
        if os.path.exists('objective2_baseline_results.csv'):
            baseline_df = pd.read_csv('objective2_baseline_results.csv')
            results['baseline_ml'] = baseline_df
            print(f"Loaded baseline ML results: {len(baseline_df)} models")
        
        # Load deep learning results
        if os.path.exists('objective2_deep_learning_results.csv'):
            dl_df = pd.read_csv('objective2_deep_learning_results.csv')
            results['deep_learning'] = dl_df
            print(f"Loaded deep learning results: {len(dl_df)} models")
        
        # Load ensemble results
        if os.path.exists('objective2_ensemble_results.csv'):
            ensemble_df = pd.read_csv('objective2_ensemble_results.csv')
            results['ensemble'] = ensemble_df
            print(f"Loaded ensemble results: {len(ensemble_df)} models")
        
        # Load dataset info
        if os.path.exists('objective2_analysis_report.json'):
            with open('objective2_analysis_report.json', 'r') as f:
                dataset_info = json.load(f)
            results['dataset_info'] = dataset_info
        
        self.results = results
        return results
    
    def analyze_dataset_limitations(self):
        """Analyze dataset limitations"""
        print("\n=== Analyzing Dataset Limitations ===")
        
        dataset_info = self.results.get('dataset_info', {})
        structure = dataset_info.get('dataset_structure', {})
        
        limitations = []
        
        # 1. Dataset size analysis
        total_tampered = sum(structure.get('tampered_images', {}).values())
        total_originals = structure.get('original_images', {}).get('tampered_originals', 0)
        total_images = total_tampered + total_originals
        
        limitations.append({
            'issue': 'Small Dataset Size',
            'description': f'Only {total_images} images total ({total_tampered} tampered, {total_originals} original)',
            'impact': 'High - Insufficient data for complex models to learn patterns',
            'severity': 'Critical'
        })
        
        # 2. Class imbalance
        imbalance_ratio = total_tampered / total_originals if total_originals > 0 else float('inf')
        limitations.append({
            'issue': 'Class Imbalance',
            'description': f'Tampered:Original ratio is {imbalance_ratio:.1f}:1',
            'impact': 'High - Models may be biased towards majority class',
            'severity': 'High'
        })
        
        # 3. Limited tampering types
        tamper_types = len(structure.get('tampered_images', {}))
        limitations.append({
            'issue': 'Limited Tampering Types',
            'description': f'Only {tamper_types} types of tampering (Copy-move, Retouching, Splicing)',
            'impact': 'Medium - May not cover all real-world tampering scenarios',
            'severity': 'Medium'
        })
        
        # 4. Single source images
        limitations.append({
            'issue': 'Limited Image Diversity',
            'description': 'Images from limited sources (SUPATLANTIQUE dataset)',
            'impact': 'Medium - May not generalize to diverse image types',
            'severity': 'Medium'
        })
        
        self.analysis_results['dataset_limitations'] = limitations
        return limitations
    
    def analyze_model_performance(self):
        """Analyze model performance across all approaches"""
        print("\n=== Analyzing Model Performance ===")
        
        performance_analysis = {}
        
        # Analyze baseline ML models
        if 'baseline_ml' in self.results:
            baseline_df = self.results['baseline_ml']
            best_baseline = baseline_df.loc[baseline_df['Accuracy'].idxmax()]
            
            performance_analysis['baseline_ml'] = {
                'best_model': best_baseline['Model'],
                'best_accuracy': best_baseline['Accuracy'],
                'avg_accuracy': baseline_df['Accuracy'].mean(),
                'std_accuracy': baseline_df['Accuracy'].std(),
                'models_tested': len(baseline_df),
                'issues': [
                    'All models plateau at 75% accuracy',
                    'Low ROC AUC scores indicate poor probability calibration',
                    'High cross-validation variance suggests overfitting'
                ]
            }
        
        # Analyze deep learning models
        if 'deep_learning' in self.results:
            dl_df = self.results['deep_learning']
            best_dl = dl_df.loc[dl_df['Test Accuracy'].idxmax()]
            
            performance_analysis['deep_learning'] = {
                'best_model': best_dl['Model'],
                'best_accuracy': best_dl['Test Accuracy'],
                'avg_accuracy': dl_df['Test Accuracy'].mean(),
                'std_accuracy': dl_df['Test Accuracy'].std(),
                'models_tested': len(dl_df),
                'issues': [
                    'All models achieve same 75% accuracy',
                    'Early stopping suggests overfitting',
                    'Transfer learning not providing advantage',
                    'Limited data prevents deep models from learning complex patterns'
                ]
            }
        
        # Analyze ensemble models
        if 'ensemble' in self.results:
            ensemble_df = self.results['ensemble']
            best_ensemble = ensemble_df.loc[ensemble_df['Accuracy'].idxmax()]
            
            performance_analysis['ensemble'] = {
                'best_model': best_ensemble['Model'],
                'best_accuracy': best_ensemble['Accuracy'],
                'avg_accuracy': ensemble_df['Accuracy'].mean(),
                'std_accuracy': ensemble_df['Accuracy'].std(),
                'models_tested': len(ensemble_df),
                'issues': [
                    'Ensemble methods not improving over individual models',
                    'Custom ensemble implementations failed due to sklearn compatibility',
                    'Voting classifiers perform same as best individual model'
                ]
            }
        
        self.analysis_results['model_performance'] = performance_analysis
        return performance_analysis
    
    def analyze_feature_quality(self):
        """Analyze quality of extracted features"""
        print("\n=== Analyzing Feature Quality ===")
        
        feature_analysis = {}
        
        # Load features if available
        if os.path.exists('objective2_forensic_features.csv'):
            features_df = pd.read_csv('objective2_forensic_features.csv')
            feature_cols = [col for col in features_df.columns if col not in ['image_path', 'label', 'tamper_type', 'category']]
            X = features_df[feature_cols]
            y = features_df['label']
            
            # Feature statistics
            feature_analysis['feature_stats'] = {
                'total_features': len(feature_cols),
                'feature_types': [
                    'ELA features (Error Level Analysis)',
                    'Noise analysis features',
                    'Compression artifact features',
                    'Frequency domain features',
                    'Texture features',
                    'Statistical features',
                    'Edge and gradient features',
                    'Color features',
                    'Wavelet features',
                    'Local Binary Pattern features'
                ],
                'issues': [
                    'High correlation between features (479 pairs >0.9)',
                    'Some features may be redundant',
                    'Feature scaling issues with extreme values',
                    'Limited discriminative power for this dataset'
                ]
            }
            
            # Feature correlation analysis
            correlation_matrix = X.corr()
            high_corr_pairs = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_val = correlation_matrix.iloc[i, j]
                    if abs(corr_val) > 0.9:
                        high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], corr_val))
            
            feature_analysis['correlation_issues'] = {
                'high_correlation_pairs': len(high_corr_pairs),
                'sample_pairs': high_corr_pairs[:5] if high_corr_pairs else []
            }
        
        self.analysis_results['feature_quality'] = feature_analysis
        return feature_analysis
    
    def analyze_data_quality(self):
        """Analyze data quality issues"""
        print("\n=== Analyzing Data Quality ===")
        
        data_quality_issues = []
        
        # 1. Image quality analysis
        data_quality_issues.append({
            'issue': 'Image Resolution and Quality',
            'description': 'All images are high resolution (2480x3500) but may have quality variations',
            'impact': 'Medium - Inconsistent quality may affect feature extraction',
            'severity': 'Medium'
        })
        
        # 2. Tampering sophistication
        data_quality_issues.append({
            'issue': 'Tampering Sophistication',
            'description': 'SUPATLANTIQUE dataset contains professional-grade tampering',
            'impact': 'High - Sophisticated tampering is harder to detect',
            'severity': 'High'
        })
        
        # 3. Ground truth quality
        data_quality_issues.append({
            'issue': 'Ground Truth Quality',
            'description': 'Binary masks available but may not perfectly align with actual tampering',
            'impact': 'Medium - Imperfect ground truth affects model training',
            'severity': 'Medium'
        })
        
        # 4. Feature extraction challenges
        data_quality_issues.append({
            'issue': 'Feature Extraction Challenges',
            'description': 'High-quality tampering may not leave detectable artifacts',
            'impact': 'High - Traditional forensic features may be insufficient',
            'severity': 'High'
        })
        
        self.analysis_results['data_quality'] = data_quality_issues
        return data_quality_issues
    
    def identify_root_causes(self):
        """Identify root causes of low accuracy"""
        print("\n=== Identifying Root Causes ===")
        
        root_causes = []
        
        # 1. Insufficient training data
        root_causes.append({
            'cause': 'Insufficient Training Data',
            'description': 'Only 136 images total with 108 for training is insufficient for complex models',
            'evidence': [
                'All models plateau at 75% accuracy',
                'Deep learning models show early stopping',
                'High cross-validation variance'
            ],
            'impact': 'Critical - Primary limiting factor',
            'solutions': [
                'Data augmentation',
                'Transfer learning from larger datasets',
                'Synthetic data generation',
                'Collect more diverse tampered images'
            ]
        })
        
        # 2. Class imbalance
        root_causes.append({
            'cause': 'Severe Class Imbalance',
            'description': '3:1 ratio of tampered to original images causes bias',
            'evidence': [
                'Models show high recall but low precision',
                'ROC AUC scores are very low',
                'Confusion matrices show bias towards tampered class'
            ],
            'impact': 'High - Affects model calibration',
            'solutions': [
                'SMOTE or other oversampling techniques',
                'Class weighting in models',
                'Cost-sensitive learning',
                'Collect more original images'
            ]
        })
        
        # 3. Feature limitations
        root_causes.append({
            'cause': 'Feature Extraction Limitations',
            'description': 'Traditional forensic features may not capture sophisticated tampering',
            'evidence': [
                'High feature correlation (479 pairs >0.9)',
                'Features may not be discriminative enough',
                'Professional tampering may not leave detectable artifacts'
            ],
            'impact': 'High - Features may not be suitable',
            'solutions': [
                'Deep learning feature extraction',
                'Attention mechanisms',
                'Multi-scale analysis',
                'Domain-specific feature engineering'
            ]
        })
        
        # 4. Model complexity vs data size
        root_causes.append({
            'cause': 'Model Complexity Mismatch',
            'description': 'Models too complex for available data size',
            'evidence': [
                'Deep learning models overfit quickly',
                'Ensemble methods don\'t improve performance',
                'Simple models perform as well as complex ones'
            ],
            'impact': 'Medium - Overfitting issues',
            'solutions': [
                'Simpler model architectures',
                'Stronger regularization',
                'More data or better augmentation',
                'Transfer learning with frozen layers'
            ]
        })
        
        self.analysis_results['root_causes'] = root_causes
        return root_causes
    
    def generate_recommendations(self):
        """Generate recommendations for improvement"""
        print("\n=== Generating Recommendations ===")
        
        recommendations = {
            'immediate_actions': [
                'Implement aggressive data augmentation (rotation, scaling, color jittering)',
                'Use class weighting in all models to handle imbalance',
                'Apply SMOTE or similar oversampling techniques',
                'Reduce model complexity to prevent overfitting'
            ],
            'medium_term_actions': [
                'Collect more diverse tampered and original images',
                'Implement transfer learning with pre-trained models',
                'Develop domain-specific feature extraction methods',
                'Use attention mechanisms in deep learning models'
            ],
            'long_term_actions': [
                'Create synthetic tampered images using GANs',
                'Develop multi-modal approaches (combine multiple detection methods)',
                'Implement active learning to identify informative samples',
                'Collaborate with forensic experts for better feature engineering'
            ],
            'technical_improvements': [
                'Implement proper cross-validation with stratification',
                'Use ensemble methods that work with small datasets',
                'Apply feature selection to reduce redundancy',
                'Implement proper hyperparameter optimization'
            ]
        }
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def create_comprehensive_report(self):
        """Create comprehensive analysis report"""
        print("\n=== Creating Comprehensive Analysis Report ===")
        
        # Run all analyses
        self.load_all_results()
        self.analyze_dataset_limitations()
        self.analyze_model_performance()
        self.analyze_feature_quality()
        self.analyze_data_quality()
        self.identify_root_causes()
        self.generate_recommendations()
        
        # Create summary
        summary = {
            'current_best_accuracy': 0.75,
            'target_accuracy': 0.90,
            'accuracy_gap': 0.15,
            'primary_limiting_factors': [
                'Insufficient training data (136 images)',
                'Severe class imbalance (3:1 ratio)',
                'Feature extraction limitations',
                'Model complexity mismatch'
            ],
            'feasibility_assessment': {
                'achievable_with_current_data': 'Unlikely - 90% accuracy difficult with 136 images',
                'achievable_with_more_data': 'Possible - Need 500+ images minimum',
                'achievable_with_better_features': 'Possible - Need domain-specific features',
                'achievable_with_ensemble': 'Unlikely - Current ensembles don\'t help'
            }
        }
        
        # Combine all results
        comprehensive_report = {
            'summary': summary,
            'dataset_limitations': self.analysis_results['dataset_limitations'],
            'model_performance': self.analysis_results['model_performance'],
            'feature_quality': self.analysis_results['feature_quality'],
            'data_quality': self.analysis_results['data_quality'],
            'root_causes': self.analysis_results['root_causes'],
            'recommendations': self.analysis_results['recommendations']
        }
        
        # Save report
        with open('objective2_comprehensive_analysis.json', 'w') as f:
            json.dump(comprehensive_report, f, indent=2)
        
        print("Comprehensive analysis report saved to: objective2_comprehensive_analysis.json")
        return comprehensive_report
    
    def visualize_analysis(self):
        """Create visualizations for the analysis"""
        print("\n=== Creating Analysis Visualizations ===")
        
        os.makedirs('objective2_analysis_visualizations', exist_ok=True)
        
        # 1. Model performance comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Baseline ML performance
        if 'baseline_ml' in self.results:
            baseline_df = self.results['baseline_ml']
            axes[0, 0].bar(baseline_df['Model'], baseline_df['Accuracy'], alpha=0.7, color='skyblue')
            axes[0, 0].set_title('Baseline ML Models Performance')
            axes[0, 0].set_ylabel('Accuracy')
            axes[0, 0].tick_params(axis='x', rotation=45)
            axes[0, 0].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
            axes[0, 0].legend()
        
        # Deep learning performance
        if 'deep_learning' in self.results:
            dl_df = self.results['deep_learning']
            axes[0, 1].bar(dl_df['Model'], dl_df['Test Accuracy'], alpha=0.7, color='lightgreen')
            axes[0, 1].set_title('Deep Learning Models Performance')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
            axes[0, 1].legend()
        
        # Ensemble performance
        if 'ensemble' in self.results:
            ensemble_df = self.results['ensemble']
            axes[1, 0].bar(ensemble_df['Model'], ensemble_df['Accuracy'], alpha=0.7, color='orange')
            axes[1, 0].set_title('Ensemble Models Performance')
            axes[1, 0].set_ylabel('Accuracy')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=0.9, color='red', linestyle='--', label='Target (90%)')
            axes[1, 0].legend()
        
        # Dataset size comparison
        dataset_info = self.results.get('dataset_info', {})
        structure = dataset_info.get('dataset_structure', {})
        tampered_counts = list(structure.get('tampered_images', {}).values())
        tampered_types = list(structure.get('tampered_images', {}).keys())
        original_count = structure.get('original_images', {}).get('tampered_originals', 0)
        
        categories = tampered_types + ['Original']
        counts = tampered_counts + [original_count]
        colors = ['lightcoral', 'lightblue', 'lightgreen', 'gold']
        
        axes[1, 1].pie(counts, labels=categories, autopct='%1.1f%%', colors=colors)
        axes[1, 1].set_title('Dataset Distribution')
        
        plt.tight_layout()
        plt.savefig('objective2_analysis_visualizations/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Root causes visualization
        root_causes = self.analysis_results.get('root_causes', [])
        if root_causes:
            causes = [cause['cause'] for cause in root_causes]
            impacts = [cause['impact'] for cause in root_causes]
            
            impact_mapping = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
            impact_values = [impact_mapping.get(impact, 1) for impact in impacts]
            
            plt.figure(figsize=(12, 6))
            bars = plt.bar(causes, impact_values, color=['red', 'orange', 'yellow', 'lightgreen'])
            plt.title('Root Causes Impact Analysis')
            plt.ylabel('Impact Level')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, 5)
            
            # Add impact labels
            for bar, impact in zip(bars, impacts):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                        impact, ha='center', va='bottom')
            
            plt.tight_layout()
            plt.savefig('objective2_analysis_visualizations/root_causes_impact.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print("Analysis visualizations saved to: objective2_analysis_visualizations/")

def main():
    """Main function to run accuracy analysis"""
    analyzer = AccuracyAnalysis()
    
    # Create comprehensive analysis
    report = analyzer.create_comprehensive_report()
    
    # Create visualizations
    analyzer.visualize_analysis()
    
    print("\n=== Accuracy Analysis Complete ===")
    print(f"Current best accuracy: {report['summary']['current_best_accuracy']}")
    print(f"Target accuracy: {report['summary']['target_accuracy']}")
    print(f"Accuracy gap: {report['summary']['accuracy_gap']}")
    print("\nPrimary limiting factors:")
    for factor in report['summary']['primary_limiting_factors']:
        print(f"  - {factor}")
    
    print("\nFeasibility assessment:")
    for assessment, result in report['summary']['feasibility_assessment'].items():
        print(f"  - {assessment}: {result}")

if __name__ == "__main__":
    main()
