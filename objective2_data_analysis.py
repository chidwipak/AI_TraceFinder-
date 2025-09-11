#!/usr/bin/env python3
"""
Objective 2: Tampered/Original Detection - Data Analysis
Analyzes the SUPATLANTIQUE dataset structure and creates data inventory
"""

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

class SUPATLANTIQUEDataAnalyzer:
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.tampered_path = self.dataset_path / "Tampered images"
        self.originals_path = self.dataset_path / "Originals"
        self.analysis_results = {}
        
    def analyze_dataset_structure(self):
        """Analyze the complete dataset structure"""
        print("=== SUPATLANTIQUE Dataset Analysis ===")
        
        # Count images in each category
        structure = {
            'tampered_images': {
                'copy_move': len(list((self.tampered_path / "Tampered" / "Copy-move").glob("*.tif"))),
                'retouching': len(list((self.tampered_path / "Tampered" / "Retouching").glob("*.tif"))),
                'splicing': len(list((self.tampered_path / "Tampered" / "Splicing").glob("*.tif")))
            },
            'original_images': {
                'tampered_originals': len(list((self.tampered_path / "Original").glob("*.tif"))),
                'official_originals': len(list((self.originals_path / "official").glob("*.pdf"))),
                'wikipedia_originals': len(list((self.originals_path / "wikipedia").glob("*.pdf")))
            },
            'binary_masks': {
                'copy_move_masks': len(list((self.tampered_path / "Binary masks" / "Copy-move").glob("*.tif"))),
                'retouching_masks': len(list((self.tampered_path / "Binary masks" / "Retouching").glob("*.tif"))),
                'splicing_masks': len(list((self.tampered_path / "Binary masks" / "Splicing").glob("*.tif")))
            }
        }
        
        # Calculate totals
        total_tampered = sum(structure['tampered_images'].values())
        total_originals = structure['original_images']['tampered_originals']
        total_masks = sum(structure['binary_masks'].values())
        
        print(f"Total Tampered Images: {total_tampered}")
        print(f"  - Copy-Move: {structure['tampered_images']['copy_move']}")
        print(f"  - Retouching: {structure['tampered_images']['retouching']}")
        print(f"  - Splicing: {structure['tampered_images']['splicing']}")
        
        print(f"\nTotal Original Images: {total_originals}")
        print(f"  - From tampered dataset: {structure['original_images']['tampered_originals']}")
        print(f"  - Official PDFs: {structure['original_images']['official_originals']}")
        print(f"  - Wikipedia PDFs: {structure['original_images']['wikipedia_originals']}")
        
        print(f"\nTotal Binary Masks: {total_masks}")
        print(f"  - Copy-Move masks: {structure['binary_masks']['copy_move_masks']}")
        print(f"  - Retouching masks: {structure['binary_masks']['retouching_masks']}")
        print(f"  - Splicing masks: {structure['binary_masks']['splicing_masks']}")
        
        self.analysis_results['structure'] = structure
        return structure
    
    def analyze_image_properties(self, sample_size=10):
        """Analyze image properties from a sample"""
        print(f"\n=== Image Properties Analysis (Sample: {sample_size}) ===")
        
        properties = {
            'tampered': [],
            'original': []
        }
        
        # Sample tampered images
        tampered_dirs = [
            self.tampered_path / "Tampered" / "Copy-move",
            self.tampered_path / "Tampered" / "Retouching", 
            self.tampered_path / "Tampered" / "Splicing"
        ]
        
        for dir_path in tampered_dirs:
            images = list(dir_path.glob("*.tif"))[:sample_size//3]
            for img_path in images:
                try:
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        h, w, c = img.shape
                        properties['tampered'].append({
                            'path': str(img_path),
                            'width': w,
                            'height': h,
                            'channels': c,
                            'size_mb': os.path.getsize(img_path) / (1024*1024),
                            'type': dir_path.name
                        })
                except Exception as e:
                    print(f"Error reading {img_path}: {e}")
        
        # Sample original images
        original_images = list((self.tampered_path / "Original").glob("*.tif"))[:sample_size]
        for img_path in original_images:
            try:
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w, c = img.shape
                    properties['original'].append({
                        'path': str(img_path),
                        'width': w,
                        'height': h,
                        'channels': c,
                        'size_mb': os.path.getsize(img_path) / (1024*1024),
                        'type': 'original'
                    })
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
        
        # Analyze properties
        if properties['tampered']:
            tampered_df = pd.DataFrame(properties['tampered'])
            print("\nTampered Images Properties:")
            print(f"  Average size: {tampered_df['width'].mean():.0f}x{tampered_df['height'].mean():.0f}")
            print(f"  Size range: {tampered_df['width'].min()}-{tampered_df['width'].max()} x {tampered_df['height'].min()}-{tampered_df['height'].max()}")
            print(f"  Average file size: {tampered_df['size_mb'].mean():.2f} MB")
            print(f"  File size range: {tampered_df['size_mb'].min():.2f}-{tampered_df['size_mb'].max():.2f} MB")
        
        if properties['original']:
            original_df = pd.DataFrame(properties['original'])
            print("\nOriginal Images Properties:")
            print(f"  Average size: {original_df['width'].mean():.0f}x{original_df['height'].mean():.0f}")
            print(f"  Size range: {original_df['width'].min()}-{original_df['width'].max()} x {original_df['height'].min()}-{original_df['height'].max()}")
            print(f"  Average file size: {original_df['size_mb'].mean():.2f} MB")
            print(f"  File size range: {original_df['size_mb'].min():.2f}-{original_df['size_mb'].max():.2f} MB")
        
        self.analysis_results['properties'] = properties
        return properties
    
    def create_data_inventory(self):
        """Create comprehensive data inventory for training"""
        print("\n=== Creating Data Inventory ===")
        
        inventory = []
        
        # Add tampered images
        tampered_dirs = [
            ("Copy-move", self.tampered_path / "Tampered" / "Copy-move"),
            ("Retouching", self.tampered_path / "Tampered" / "Retouching"),
            ("Splicing", self.tampered_path / "Tampered" / "Splicing")
        ]
        
        for tamper_type, dir_path in tampered_dirs:
            for img_path in dir_path.glob("*.tif"):
                inventory.append({
                    'image_path': str(img_path),
                    'label': 1,  # Tampered
                    'tamper_type': tamper_type,
                    'category': 'tampered',
                    'has_mask': True,
                    'mask_path': str(self.tampered_path / "Binary masks" / tamper_type / f"{img_path.stem}_1.tif")
                })
        
        # Add original images
        original_images = list((self.tampered_path / "Original").glob("*.tif"))
        for img_path in original_images:
            inventory.append({
                'image_path': str(img_path),
                'label': 0,  # Original
                'tamper_type': 'none',
                'category': 'original',
                'has_mask': False,
                'mask_path': None
            })
        
        # Create DataFrame
        inventory_df = pd.DataFrame(inventory)
        
        print(f"Total images in inventory: {len(inventory_df)}")
        print(f"  - Tampered: {len(inventory_df[inventory_df['label'] == 1])}")
        print(f"  - Original: {len(inventory_df[inventory_df['label'] == 0])}")
        
        print("\nTampering type distribution:")
        tamper_dist = inventory_df[inventory_df['label'] == 1]['tamper_type'].value_counts()
        for tamper_type, count in tamper_dist.items():
            print(f"  - {tamper_type}: {count}")
        
        # Save inventory
        inventory_df.to_csv('objective2_data_inventory.csv', index=False)
        print(f"\nData inventory saved to: objective2_data_inventory.csv")
        
        self.analysis_results['inventory'] = inventory_df
        return inventory_df
    
    def visualize_dataset_distribution(self):
        """Create visualizations of dataset distribution"""
        print("\n=== Creating Dataset Visualizations ===")
        
        if 'inventory' not in self.analysis_results:
            self.create_data_inventory()
        
        inventory_df = self.analysis_results['inventory']
        
        # Create output directory
        os.makedirs('objective2_analysis', exist_ok=True)
        
        # 1. Label distribution
        plt.figure(figsize=(10, 6))
        plt.subplot(1, 2, 1)
        label_counts = inventory_df['label'].value_counts()
        plt.pie(label_counts.values, labels=['Original', 'Tampered'], autopct='%1.1f%%', 
                colors=['lightgreen', 'lightcoral'])
        plt.title('Original vs Tampered Distribution')
        
        # 2. Tampering type distribution
        plt.subplot(1, 2, 2)
        tamper_counts = inventory_df[inventory_df['label'] == 1]['tamper_type'].value_counts()
        plt.bar(tamper_counts.index, tamper_counts.values, color=['skyblue', 'orange', 'lightpink'])
        plt.title('Tampering Type Distribution')
        plt.xlabel('Tampering Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('objective2_analysis/dataset_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Dataset distribution visualization saved to: objective2_analysis/dataset_distribution.png")
    
    def save_analysis_report(self):
        """Save comprehensive analysis report"""
        print("\n=== Saving Analysis Report ===")
        
        report = {
            'dataset_structure': self.analysis_results.get('structure', {}),
            'image_properties': self.analysis_results.get('properties', {}),
            'data_inventory_summary': {
                'total_images': len(self.analysis_results.get('inventory', [])),
                'tampered_count': len(self.analysis_results.get('inventory', [])[self.analysis_results.get('inventory', [])['label'] == 1]) if 'inventory' in self.analysis_results else 0,
                'original_count': len(self.analysis_results.get('inventory', [])[self.analysis_results.get('inventory', [])['label'] == 0]) if 'inventory' in self.analysis_results else 0
            },
            'recommendations': [
                "Dataset is well-balanced with multiple tampering types",
                "Binary masks available for precise localization",
                "Suitable for both binary classification and multi-class tampering type detection",
                "Consider data augmentation to increase dataset size",
                "Use transfer learning for better performance with limited data"
            ]
        }
        
        with open('objective2_analysis_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("Analysis report saved to: objective2_analysis_report.json")
        return report

def main():
    """Main analysis function"""
    dataset_path = "/home/mohanganesh/AI_TraceFinder/The SUPATLANTIQUE dataset"
    
    analyzer = SUPATLANTIQUEDataAnalyzer(dataset_path)
    
    # Run complete analysis
    analyzer.analyze_dataset_structure()
    analyzer.analyze_image_properties(sample_size=20)
    analyzer.create_data_inventory()
    analyzer.visualize_dataset_distribution()
    analyzer.save_analysis_report()
    
    print("\n=== Analysis Complete ===")
    print("Next steps:")
    print("1. Review the data inventory and analysis report")
    print("2. Proceed with data preprocessing pipeline")
    print("3. Implement feature extraction for tampered detection")

if __name__ == "__main__":
    main()
