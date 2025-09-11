#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Deep Learning Models
Implements CNN and advanced deep learning models for tampered image detection
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import cv2
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

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

class SimpleCNN(nn.Module):
    """Simple CNN architecture for tampered image detection"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        
        return x

class AdvancedCNN(nn.Module):
    """Advanced CNN with residual connections"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(AdvancedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x

class ResidualBlock(nn.Module):
    """Residual block for advanced CNN"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = self.relu(out)
        
        return out

class TransferLearningModel(nn.Module):
    """Transfer learning model using pre-trained ResNet"""
    
    def __init__(self, num_classes=2, dropout_rate=0.5, pretrained=True):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet
        if pretrained:
            self.backbone = models.resnet50(pretrained=True)
        else:
            self.backbone = models.resnet50(pretrained=False)
        
        # Freeze early layers
        for param in list(self.backbone.parameters())[:-10]:
            param.requires_grad = False
        
        # Replace classifier
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class DeepLearningTrainer:
    """Trainer class for deep learning models"""
    
    def __init__(self, model, device, model_name):
        self.model = model.to(device)
        self.device = device
        self.model_name = model_name
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader, criterion, optimizer):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, batch in enumerate(train_loader):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader, criterion):
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, train_loader, val_loader, num_epochs=50, learning_rate=0.001, patience=10):
        """Train the model"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"Training {self.model_name} for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), f'best_{self.model_name.lower().replace(" ", "_")}.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.2f}%")
        return best_val_acc
    
    def evaluate(self, test_loader):
        """Evaluate the model on test set"""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(images)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title(f'{self.model_name} - Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title(f'{self.model_name} - Training Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.model_name.lower().replace(" ", "_")}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

class SUPATLANTIQUEDeepLearning:
    """Main class for deep learning models on SUPATLANTIQUE dataset"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.device = device
        self.models = {}
        self.results = {}
        
    def load_data(self, inventory_path='objective2_data_inventory.csv'):
        """Load data from inventory"""
        print("=== Loading Data ===")
        
        df = pd.read_csv(inventory_path)
        print(f"Loaded {len(df)} images")
        
        # Split data
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
        
        print(f"Train set: {len(train_df)} images")
        print(f"Test set: {len(test_df)} images")
        
        return train_df, test_df
    
    def create_transforms(self):
        """Create data transforms"""
        print("=== Creating Data Transforms ===")
        
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
        
        # Validation/test transforms
        val_transform = A.Compose([
            A.Resize(self.target_size[0], self.target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        return train_transform, val_transform
    
    def create_data_loaders(self, train_df, test_df, batch_size=16):
        """Create data loaders"""
        print("=== Creating Data Loaders ===")
        
        train_transform, val_transform = self.create_transforms()
        
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
            num_workers=4,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"Train loader: {len(train_loader)} batches")
        print(f"Test loader: {len(test_loader)} batches")
        
        return train_loader, test_loader
    
    def initialize_models(self):
        """Initialize deep learning models"""
        print("=== Initializing Deep Learning Models ===")
        
        self.models = {
            'Simple CNN': SimpleCNN(num_classes=2, dropout_rate=0.5),
            'Advanced CNN': AdvancedCNN(num_classes=2, dropout_rate=0.5),
            'Transfer Learning (ResNet50)': TransferLearningModel(num_classes=2, dropout_rate=0.5, pretrained=True)
        }
        
        print(f"Initialized {len(self.models)} deep learning models")
        return self.models
    
    def train_models(self, train_loader, test_loader, num_epochs=50):
        """Train all models"""
        print("=== Training Deep Learning Models ===")
        
        self.results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Create trainer
            trainer = DeepLearningTrainer(model, self.device, name)
            
            # Train model
            best_val_acc = trainer.train(train_loader, test_loader, num_epochs=num_epochs)
            
            # Evaluate on test set
            predictions, labels, probabilities = trainer.evaluate(test_loader)
            
            # Calculate metrics
            accuracy = accuracy_score(labels, predictions)
            roc_auc = roc_auc_score(labels, probabilities[:, 1])
            
            # Store results
            self.results[name] = {
                'model': model,
                'trainer': trainer,
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'best_val_acc': best_val_acc,
                'predictions': predictions,
                'labels': labels,
                'probabilities': probabilities
            }
            
            print(f"  - Test Accuracy: {accuracy:.4f}")
            print(f"  - ROC AUC: {roc_auc:.4f}")
            print(f"  - Best Val Accuracy: {best_val_acc:.2f}%")
            
            # Plot training history
            trainer.plot_training_history()
        
        return self.results
    
    def evaluate_models(self):
        """Evaluate all trained models"""
        print("\n=== Model Evaluation ===")
        
        # Create results DataFrame
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Model': name,
                'Test Accuracy': result['accuracy'],
                'ROC AUC': result['roc_auc'],
                'Best Val Accuracy': result['best_val_acc']
            })
        
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('Test Accuracy', ascending=False)
        
        print("\nDeep Learning Model Performance:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Save results
        results_df.to_csv('objective2_deep_learning_results.csv', index=False)
        print(f"\nResults saved to: objective2_deep_learning_results.csv")
        
        return results_df
    
    def visualize_results(self):
        """Create visualizations of results"""
        print("\n=== Creating Result Visualizations ===")
        
        # Create output directory
        os.makedirs('objective2_deep_learning_results', exist_ok=True)
        
        # 1. Model comparison
        results_data = []
        for name, result in self.results.items():
            results_data.append({
                'Model': name,
                'Test Accuracy': result['accuracy'],
                'ROC AUC': result['roc_auc']
            })
        
        results_df = pd.DataFrame(results_data)
        
        plt.figure(figsize=(12, 6))
        x = np.arange(len(results_df))
        width = 0.35
        
        plt.bar(x - width/2, results_df['Test Accuracy'], width, label='Test Accuracy', alpha=0.8)
        plt.bar(x + width/2, results_df['ROC AUC'], width, label='ROC AUC', alpha=0.8)
        
        plt.xlabel('Models')
        plt.ylabel('Score')
        plt.title('Deep Learning Model Performance Comparison')
        plt.xticks(x, results_df['Model'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('objective2_deep_learning_results/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        fig, axes = plt.subplots(1, len(self.results), figsize=(5*len(self.results), 4))
        if len(self.results) == 1:
            axes = [axes]
        
        for i, (name, result) in enumerate(self.results.items()):
            cm = confusion_matrix(result['labels'], result['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('objective2_deep_learning_results/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Visualizations saved to: objective2_deep_learning_results/")
    
    def save_models(self, output_dir='objective2_deep_learning_models'):
        """Save trained models"""
        print(f"\n=== Saving Models to {output_dir} ===")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for name, result in self.results.items():
            model_path = os.path.join(output_dir, f'{name.lower().replace(" ", "_")}.pth')
            torch.save(result['model'].state_dict(), model_path)
            print(f"Saved {name} to {model_path}")
    
    def generate_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== Generating Evaluation Report ===")
        
        # Find best model
        best_model = None
        best_accuracy = 0
        
        for name, result in self.results.items():
            if result['accuracy'] > best_accuracy:
                best_accuracy = result['accuracy']
                best_model = name
        
        # Generate report
        report = {
            'best_model': {
                'name': best_model,
                'accuracy': best_accuracy,
                'results': {
                    'accuracy': self.results[best_model]['accuracy'],
                    'roc_auc': self.results[best_model]['roc_auc'],
                    'best_val_acc': self.results[best_model]['best_val_acc']
                } if best_model else None
            },
            'model_performance': {
                name: {
                    'accuracy': result['accuracy'],
                    'roc_auc': result['roc_auc'],
                    'best_val_acc': result['best_val_acc']
                } for name, result in self.results.items()
            },
            'recommendations': [
                f"Best performing model: {best_model} with {best_accuracy:.4f} accuracy",
                "Consider ensemble methods combining multiple deep learning models",
                "Data augmentation and transfer learning show promise",
                "Further hyperparameter tuning may improve performance"
            ]
        }
        
        # Save report
        with open('objective2_deep_learning_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Evaluation report saved to: objective2_deep_learning_report.json")
        print(f"\nBest Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        
        return report

def main():
    """Main function to run deep learning models"""
    # Initialize deep learning pipeline
    dl_pipeline = SUPATLANTIQUEDeepLearning(target_size=(224, 224))
    
    # Load data
    train_df, test_df = dl_pipeline.load_data()
    
    # Create data loaders
    train_loader, test_loader = dl_pipeline.create_data_loaders(train_df, test_df, batch_size=16)
    
    # Initialize models
    dl_pipeline.initialize_models()
    
    # Train models
    dl_pipeline.train_models(train_loader, test_loader, num_epochs=30)
    
    # Evaluate models
    results_df = dl_pipeline.evaluate_models()
    
    # Create visualizations
    dl_pipeline.visualize_results()
    
    # Save models
    dl_pipeline.save_models()
    
    # Generate report
    dl_pipeline.generate_report()
    
    print("\n=== Deep Learning Models Complete ===")
    print("Next steps:")
    print("1. Review deep learning model results")
    print("2. Proceed with ensemble methods")
    print("3. Implement advanced optimization techniques")

if __name__ == "__main__":
    main()
