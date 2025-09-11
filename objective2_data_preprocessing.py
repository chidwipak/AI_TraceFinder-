#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Data Preprocessing
Preprocesses the SUPATLANTIQUE dataset for tampered image detection
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

class TamperedImageDataset(Dataset):
    """Custom dataset class for tampered image detection"""
    
    def __init__(self, image_paths, labels, tamper_types, transform=None, target_size=(224, 224)):
        self.image_paths = image_paths
        self.labels = labels
        self.tamper_types = tamper_types
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image=image)['image']
        
        # Get label and tamper type
        label = self.labels[idx]
        tamper_type = self.tamper_types[idx]
        
        return {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'tamper_type': tamper_type,
            'image_path': image_path
        }

class SUPATLANTIQUEDataPreprocessor:
    def __init__(self, dataset_path, target_size=(224, 224)):
        self.dataset_path = Path(dataset_path)
        self.target_size = target_size
        self.inventory_path = 'objective2_data_inventory.csv'
        self.preprocessed_data = {}
        
    def load_data_inventory(self):
        """Load the data inventory created in analysis phase"""
        if not os.path.exists(self.inventory_path):
            raise FileNotFoundError(f"Data inventory not found: {self.inventory_path}")
        
        self.inventory_df = pd.read_csv(self.inventory_path)
        print(f"Loaded data inventory with {len(self.inventory_df)} images")
        print(f"  - Tampered: {len(self.inventory_df[self.inventory_df['label'] == 1])}")
        print(f"  - Original: {len(self.inventory_df[self.inventory_df['label'] == 0])}")
        
        return self.inventory_df
    
    def create_train_test_split(self, test_size=0.2, random_state=42, stratify=True):
        """Create train-test split with stratification"""
        print(f"\n=== Creating Train-Test Split (test_size={test_size}) ===")
        
        # Prepare data
        X = self.inventory_df[['image_path', 'tamper_type']].values
        y = self.inventory_df['label'].values
        
        if stratify:
            # Stratify by both label and tamper type for balanced splits
            stratify_col = self.inventory_df['label'].astype(str) + '_' + self.inventory_df['tamper_type']
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, 
                stratify=stratify_col
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Create train and test DataFrames
        train_df = pd.DataFrame({
            'image_path': X_train[:, 0],
            'tamper_type': X_train[:, 1],
            'label': y_train
        })
        
        test_df = pd.DataFrame({
            'image_path': X_test[:, 0],
            'tamper_type': X_test[:, 1],
            'label': y_test
        })
        
        print(f"Train set: {len(train_df)} images")
        print(f"  - Tampered: {len(train_df[train_df['label'] == 1])}")
        print(f"  - Original: {len(train_df[train_df['label'] == 0])}")
        
        print(f"Test set: {len(test_df)} images")
        print(f"  - Tampered: {len(test_df[test_df['label'] == 1])}")
        print(f"  - Original: {len(test_df[test_df['label'] == 0])}")
        
        # Save splits
        train_df.to_csv('objective2_train_split.csv', index=False)
        test_df.to_csv('objective2_test_split.csv', index=False)
        
        self.preprocessed_data['train_df'] = train_df
        self.preprocessed_data['test_df'] = test_df
        
        return train_df, test_df
    
    def create_data_augmentation_transforms(self):
        """Create data augmentation transforms for training"""
        print("\n=== Creating Data Augmentation Transforms ===")
        
        # Training transforms with augmentation
        train_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        # Validation/test transforms without augmentation
        val_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        print("Data augmentation transforms created:")
        print("  - Training: Horizontal/Vertical flip, rotation, brightness/contrast, noise, blur")
        print("  - Validation: Only resize and normalize")
        
        self.preprocessed_data['train_transform'] = train_transform
        self.preprocessed_data['val_transform'] = val_transform
        
        return train_transform, val_transform
    
    def create_data_loaders(self, batch_size=16, num_workers=4):
        """Create PyTorch data loaders"""
        print(f"\n=== Creating Data Loaders (batch_size={batch_size}) ===")
        
        if 'train_df' not in self.preprocessed_data:
            raise ValueError("Train-test split not created. Run create_train_test_split() first.")
        
        train_df = self.preprocessed_data['train_df']
        test_df = self.preprocessed_data['test_df']
        train_transform = self.preprocessed_data['train_transform']
        val_transform = self.preprocessed_data['val_transform']
        
        # Create datasets
        train_dataset = TamperedImageDataset(
            image_paths=train_df['image_path'].tolist(),
            labels=train_df['label'].tolist(),
            tamper_types=train_df['tamper_type'].tolist(),
            transform=train_transform,
            target_size=self.target_size
        )
        
        test_dataset = TamperedImageDataset(
            image_paths=test_df['image_path'].tolist(),
            labels=test_df['label'].tolist(),
            tamper_types=test_df['tamper_type'].tolist(),
            transform=val_transform,
            target_size=self.target_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"Data loaders created:")
        print(f"  - Train loader: {len(train_loader)} batches")
        print(f"  - Test loader: {len(test_loader)} batches")
        
        self.preprocessed_data['train_loader'] = train_loader
        self.preprocessed_data['test_loader'] = train_loader
        self.preprocessed_data['train_dataset'] = train_dataset
        self.preprocessed_data['test_dataset'] = test_dataset
        
        return train_loader, test_loader
    
    def create_sklearn_datasets(self, feature_extractor=None):
        """Create datasets for sklearn models (with feature extraction)"""
        print("\n=== Creating Sklearn Datasets ===")
        
        if 'train_df' not in self.preprocessed_data:
            raise ValueError("Train-test split not created. Run create_train_test_split() first.")
        
        train_df = self.preprocessed_data['train_df']
        test_df = self.preprocessed_data['test_df']
        
        # If no feature extractor provided, use basic image features
        if feature_extractor is None:
            feature_extractor = self._extract_basic_features
        
        print("Extracting features for sklearn models...")
        
        # Extract features for training set
        X_train = []
        y_train = train_df['label'].values
        
        for idx, row in train_df.iterrows():
            features = feature_extractor(row['image_path'])
            X_train.append(features)
        
        # Extract features for test set
        X_test = []
        y_test = test_df['label'].values
        
        for idx, row in test_df.iterrows():
            features = feature_extractor(row['image_path'])
            X_test.append(features)
        
        X_train = np.array(X_train)
        X_test = np.array(X_test)
        
        print(f"Sklearn datasets created:")
        print(f"  - X_train shape: {X_train.shape}")
        print(f"  - X_test shape: {X_test.shape}")
        print(f"  - y_train shape: {y_train.shape}")
        print(f"  - y_test shape: {y_test.shape}")
        
        self.preprocessed_data['X_train'] = X_train
        self.preprocessed_data['X_test'] = X_test
        self.preprocessed_data['y_train'] = y_train
        self.preprocessed_data['y_test'] = y_test
        
        return X_train, X_test, y_train, y_test
    
    def _extract_basic_features(self, image_path):
        """Extract basic image features for sklearn models"""
        # Load and resize image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (64, 64))  # Smaller size for basic features
        
        # Extract basic statistical features
        features = []
        
        # Color features
        for channel in range(3):
            channel_data = image[:, :, channel].flatten()
            features.extend([
                np.mean(channel_data),
                np.std(channel_data),
                np.median(channel_data),
                np.percentile(channel_data, 25),
                np.percentile(channel_data, 75)
            ])
        
        # Texture features (using simple gradient)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        features.extend([
            np.mean(grad_x),
            np.std(grad_x),
            np.mean(grad_y),
            np.std(grad_y)
        ])
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        features.extend([
            np.sum(edges > 0) / (edges.shape[0] * edges.shape[1]),  # Edge density
            np.mean(edges)
        ])
        
        return np.array(features)
    
    def visualize_preprocessing_results(self):
        """Visualize preprocessing results and data distribution"""
        print("\n=== Creating Preprocessing Visualizations ===")
        
        if 'train_df' not in self.preprocessed_data:
            raise ValueError("Train-test split not created. Run create_train_test_split() first.")
        
        train_df = self.preprocessed_data['train_df']
        test_df = self.preprocessed_data['test_df']
        
        # Create output directory
        os.makedirs('objective2_preprocessing', exist_ok=True)
        
        # 1. Train-test split visualization
        plt.figure(figsize=(15, 5))
        
        # Label distribution
        plt.subplot(1, 3, 1)
        train_labels = train_df['label'].value_counts()
        test_labels = test_df['label'].value_counts()
        
        x = np.arange(2)
        width = 0.35
        
        plt.bar(x - width/2, [train_labels[0], train_labels[1]], width, label='Train', alpha=0.8)
        plt.bar(x + width/2, [test_labels[0], test_labels[1]], width, label='Test', alpha=0.8)
        
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Train-Test Split by Label')
        plt.xticks(x, ['Original', 'Tampered'])
        plt.legend()
        
        # Tamper type distribution in train set
        plt.subplot(1, 3, 2)
        train_tamper = train_df[train_df['label'] == 1]['tamper_type'].value_counts()
        plt.pie(train_tamper.values, labels=train_tamper.index, autopct='%1.1f%%')
        plt.title('Tamper Type Distribution (Train)')
        
        # Tamper type distribution in test set
        plt.subplot(1, 3, 3)
        test_tamper = test_df[test_df['label'] == 1]['tamper_type'].value_counts()
        plt.pie(test_tamper.values, labels=test_tamper.index, autopct='%1.1f%%')
        plt.title('Tamper Type Distribution (Test)')
        
        plt.tight_layout()
        plt.savefig('objective2_preprocessing/data_split_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Sample images visualization
        self._visualize_sample_images()
        
        print("Preprocessing visualizations saved to: objective2_preprocessing/")
    
    def _visualize_sample_images(self):
        """Visualize sample images from different categories"""
        train_df = self.preprocessed_data['train_df']
        
        # Get sample images
        original_samples = train_df[train_df['label'] == 0].head(3)
        tampered_samples = train_df[train_df['label'] == 1].head(9)
        
        plt.figure(figsize=(15, 10))
        
        # Original images
        for i, (_, row) in enumerate(original_samples.iterrows()):
            plt.subplot(4, 3, i + 1)
            image = cv2.imread(row['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            plt.imshow(image)
            plt.title(f'Original\n{Path(row["image_path"]).name}')
            plt.axis('off')
        
        # Tampered images
        for i, (_, row) in enumerate(tampered_samples.iterrows()):
            plt.subplot(4, 3, i + 4)
            image = cv2.imread(row['image_path'])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (150, 150))
            plt.imshow(image)
            plt.title(f'{row["tamper_type"]}\n{Path(row["image_path"]).name}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('objective2_preprocessing/sample_images.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_preprocessing_summary(self):
        """Save preprocessing summary and configuration"""
        print("\n=== Saving Preprocessing Summary ===")
        
        summary = {
            'dataset_info': {
                'total_images': len(self.inventory_df),
                'target_size': self.target_size,
                'train_test_split': 0.8
            },
            'preprocessing_steps': [
                'Data inventory loading',
                'Train-test split with stratification',
                'Data augmentation transforms creation',
                'PyTorch data loaders creation',
                'Sklearn datasets creation'
            ],
            'data_augmentation': {
                'horizontal_flip': 0.5,
                'vertical_flip': 0.3,
                'random_rotation': 0.3,
                'brightness_contrast': 0.5,
                'hue_saturation': 0.5,
                'gaussian_noise': 0.3,
                'blur': 0.3
            },
            'normalization': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            }
        }
        
        if 'train_df' in self.preprocessed_data:
            train_df = self.preprocessed_data['train_df']
            test_df = self.preprocessed_data['test_df']
            
            summary['split_info'] = {
                'train_size': len(train_df),
                'test_size': len(test_df),
                'train_tampered': len(train_df[train_df['label'] == 1]),
                'train_original': len(train_df[train_df['label'] == 0]),
                'test_tampered': len(test_df[test_df['label'] == 1]),
                'test_original': len(test_df[test_df['label'] == 0])
            }
        
        with open('objective2_preprocessing_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("Preprocessing summary saved to: objective2_preprocessing_summary.json")
        return summary

def main():
    """Main preprocessing function"""
    dataset_path = "/home/mohanganesh/AI_TraceFinder/The SUPATLANTIQUE dataset"
    
    preprocessor = SUPATLANTIQUEDataPreprocessor(dataset_path, target_size=(224, 224))
    
    # Run complete preprocessing pipeline
    preprocessor.load_data_inventory()
    preprocessor.create_train_test_split(test_size=0.2, stratify=True)
    preprocessor.create_data_augmentation_transforms()
    preprocessor.create_data_loaders(batch_size=16)
    preprocessor.create_sklearn_datasets()
    preprocessor.visualize_preprocessing_results()
    preprocessor.save_preprocessing_summary()
    
    print("\n=== Preprocessing Complete ===")
    print("Next steps:")
    print("1. Review preprocessing results and visualizations")
    print("2. Proceed with feature extraction for forensic analysis")
    print("3. Implement baseline ML models")

if __name__ == "__main__":
    main()
