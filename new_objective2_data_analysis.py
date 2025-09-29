#!/usr/bin/env python3
"""
New Objective 2: Comprehensive Tampered/Original Detection - Data Analysis
Comprehensive analysis of ALL available data sources:
- Originals/ folder (PDFs to be converted to images)
- Tampered images/ folder (all subfolders)
- Binary masks/ for region-specific analysis
- Description/ for metadata analysis
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

# PDF processing
import fitz  # PyMuPDF
from pdf2image import convert_from_path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComprehensiveObjective2DataAnalyzer:
    def __init__(self, dataset_path="The SUPATLANTIQUE dataset"):
        self.dataset_path = Path(dataset_path)
        self.output_path = "new_objective2_analysis"
        self.converted_images_path = "new_objective2_converted_images"
        
        # Create output directories
        self.create_output_directories()
        
        # Data containers
        self.complete_inventory = []
        self.conversion_stats = {}
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            self.output_path,
            self.converted_images_path,
            f"{self.output_path}/visualizations",
            f"{self.output_path}/reports",
            f"{self.converted_images_path}/originals_official",
            f"{self.converted_images_path}/originals_wikipedia",
            f"{self.converted_images_path}/tampered_images",
            f"{self.converted_images_path}/binary_masks"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def convert_pdfs_to_images(self):
        """Convert all PDFs in Originals/ folder to images"""
        logger.info("="*80)
        logger.info("CONVERTING PDFs TO IMAGES")
        logger.info("="*80)
        
        originals_path = self.dataset_path / "Originals"
        
        # Process official PDFs
        official_path = originals_path / "official"
        if official_path.exists():
            logger.info("Processing Official PDFs...")
            official_count = self._convert_pdf_folder(official_path, "originals_official")
            self.conversion_stats['official_pdfs'] = official_count
        
        # Process wikipedia PDFs
        wikipedia_path = originals_path / "wikipedia"
        if wikipedia_path.exists():
            logger.info("Processing Wikipedia PDFs...")
            wikipedia_count = self._convert_pdf_folder(wikipedia_path, "originals_wikipedia")
            self.conversion_stats['wikipedia_pdfs'] = wikipedia_count
        
        logger.info(f"PDF Conversion Complete:")
        logger.info(f"  Official PDFs converted: {self.conversion_stats.get('official_pdfs', 0)}")
        logger.info(f"  Wikipedia PDFs converted: {self.conversion_stats.get('wikipedia_pdfs', 0)}")
        
    def _convert_pdf_folder(self, pdf_folder, output_subfolder):
        """Convert PDFs in a specific folder to images"""
        converted_count = 0
        
        for pdf_file in pdf_folder.glob("*.pdf"):
            try:
                logger.info(f"Converting: {pdf_file.name}")
                
                # Convert PDF to images using pdf2image
                pages = convert_from_path(pdf_file, dpi=300, first_page=1, last_page=1)
                
                if pages:
                    # Save first page as image
                    output_path = Path(self.converted_images_path) / output_subfolder
                    image_filename = pdf_file.stem + ".png"
                    image_path = output_path / image_filename
                    
                    pages[0].save(image_path, "PNG", quality=95)
                    
                    # Add to inventory
                    self.complete_inventory.append({
                        'image_path': str(image_path),
                        'source_pdf': str(pdf_file),
                        'label': 0,  # Original
                        'category': 'original',
                        'source_type': output_subfolder,
                        'tamper_type': 'none',
                        'has_mask': False,
                        'mask_path': '',
                        'file_size': image_path.stat().st_size,
                        'conversion_date': datetime.now().isoformat()
                    })
                    
                    converted_count += 1
                    logger.info(f"  ✓ Converted to: {image_filename}")
                else:
                    logger.warning(f"  ✗ No pages found in {pdf_file.name}")
                    
            except Exception as e:
                logger.error(f"  ✗ Failed to convert {pdf_file.name}: {e}")
                continue
        
        return converted_count
    
    def analyze_tampered_images_folder(self):
        """Analyze all subfolders in Tampered images/ folder"""
        logger.info("="*80)
        logger.info("ANALYZING TAMPERED IMAGES FOLDER")
        logger.info("="*80)
        
        tampered_path = self.dataset_path / "Tampered images"
        
        # 1. Analyze Original/ folder (original versions of tampered images)
        original_path = tampered_path / "Original"
        if original_path.exists():
            logger.info("Processing Original/ folder...")
            self._analyze_image_folder(original_path, label=0, category="original_tampered_pair")
        
        # 2. Analyze Tampered/ folder (all tampering types)
        tampered_images_path = tampered_path / "Tampered"
        if tampered_images_path.exists():
            for tamper_type in ["Copy-move", "Retouching", "Splicing"]:
                type_path = tampered_images_path / tamper_type
                if type_path.exists():
                    logger.info(f"Processing {tamper_type}/ folder...")
                    self._analyze_image_folder(type_path, label=1, category="tampered", tamper_type=tamper_type)
        
        # 3. Analyze Binary masks/ folder
        binary_masks_path = tampered_path / "Binary masks"
        if binary_masks_path.exists():
            logger.info("Processing Binary masks/ folder...")
            self._analyze_binary_masks_folder(binary_masks_path)
        
        # 4. Analyze Description/ folder
        description_path = tampered_path / "Description"
        if description_path.exists():
            logger.info("Processing Description/ folder...")
            self._analyze_description_folder(description_path)
    
    def _analyze_image_folder(self, folder_path, label, category, tamper_type="none"):
        """Analyze images in a specific folder"""
        for image_file in folder_path.glob("*.tif"):
            try:
                # Get file info
                file_size = image_file.stat().st_size
                
                # Load image to get dimensions
                img = cv2.imread(str(image_file))
                if img is not None:
                    height, width, channels = img.shape
                else:
                    height, width, channels = 0, 0, 0
                
                # Check for corresponding binary mask
                mask_path = ""
                has_mask = False
                if tamper_type != "none":
                    mask_path = self._find_corresponding_mask(image_file, tamper_type)
                    has_mask = os.path.exists(mask_path) if mask_path else False
                
                # Add to inventory
                self.complete_inventory.append({
                    'image_path': str(image_file),
                    'source_pdf': '',
                    'label': label,
                    'category': category,
                    'source_type': folder_path.name,
                    'tamper_type': tamper_type,
                    'has_mask': has_mask,
                    'mask_path': mask_path,
                    'file_size': file_size,
                    'width': width,
                    'height': height,
                    'channels': channels,
                    'conversion_date': ''
                })
                
                logger.info(f"  ✓ Added: {image_file.name} ({width}x{height})")
                
            except Exception as e:
                logger.error(f"  ✗ Failed to process {image_file.name}: {e}")
                continue
    
    def _find_corresponding_mask(self, image_file, tamper_type):
        """Find corresponding binary mask for a tampered image"""
        mask_folder = self.dataset_path / "Tampered images" / "Binary masks" / tamper_type
        
        # Extract base name from image file
        base_name = image_file.stem
        
        # Try different mask naming patterns
        possible_masks = [
            mask_folder / f"{base_name}_1.tif",
            mask_folder / f"{base_name}_mask.tif",
            mask_folder / f"{base_name}.tif"
        ]
        
        for mask_path in possible_masks:
            if mask_path.exists():
                return str(mask_path)
        
        return ""
    
    def _analyze_binary_masks_folder(self, masks_path):
        """Analyze binary masks and their statistics"""
        mask_stats = {}
        
        for tamper_type in ["Copy-move", "Retouching", "Splicing"]:
            type_path = masks_path / tamper_type
            if type_path.exists():
                mask_count = len(list(type_path.glob("*.tif")))
                mask_stats[tamper_type] = mask_count
                logger.info(f"  {tamper_type} masks: {mask_count}")
        
        # Save mask statistics
        with open(f"{self.output_path}/reports/mask_statistics.json", 'w') as f:
            json.dump(mask_stats, f, indent=2)
    
    def _analyze_description_folder(self, description_path):
        """Analyze description PDFs for metadata"""
        description_stats = {
            'total_descriptions': 0,
            'description_files': []
        }
        
        for pdf_file in description_path.glob("*.pdf"):
            try:
                file_size = pdf_file.stat().st_size
                description_stats['description_files'].append({
                    'filename': pdf_file.name,
                    'size': file_size
                })
                description_stats['total_descriptions'] += 1
                
            except Exception as e:
                logger.error(f"Failed to process description {pdf_file.name}: {e}")
        
        # Save description statistics
        with open(f"{self.output_path}/reports/description_statistics.json", 'w') as f:
            json.dump(description_stats, f, indent=2)
        
        logger.info(f"  Description files: {description_stats['total_descriptions']}")
    
    def create_comprehensive_inventory(self):
        """Create comprehensive data inventory CSV"""
        logger.info("="*80)
        logger.info("CREATING COMPREHENSIVE DATA INVENTORY")
        logger.info("="*80)
        
        # Convert to DataFrame
        inventory_df = pd.DataFrame(self.complete_inventory)
        
        # Save inventory
        inventory_path = f"{self.output_path}/new_objective2_complete_inventory.csv"
        inventory_df.to_csv(inventory_path, index=False)
        
        # Print statistics
        logger.info(f"Complete Inventory Created: {inventory_path}")
        logger.info(f"Total samples: {len(inventory_df)}")
        logger.info(f"  Original samples: {len(inventory_df[inventory_df['label'] == 0])}")
        logger.info(f"  Tampered samples: {len(inventory_df[inventory_df['label'] == 1])}")
        
        # Category breakdown
        logger.info("\nCategory Breakdown:")
        for category in inventory_df['category'].unique():
            count = len(inventory_df[inventory_df['category'] == category])
            logger.info(f"  {category}: {count}")
        
        # Tamper type breakdown
        logger.info("\nTamper Type Breakdown:")
        for tamper_type in inventory_df['tamper_type'].unique():
            count = len(inventory_df[inventory_df['tamper_type'] == tamper_type])
            logger.info(f"  {tamper_type}: {count}")
        
        # Mask availability
        logger.info(f"\nMask Availability:")
        logger.info(f"  Images with masks: {len(inventory_df[inventory_df['has_mask'] == True])}")
        logger.info(f"  Images without masks: {len(inventory_df[inventory_df['has_mask'] == False])}")
        
        return inventory_df
    
    def generate_comprehensive_analysis_report(self, inventory_df):
        """Generate comprehensive analysis report with visualizations"""
        logger.info("="*80)
        logger.info("GENERATING COMPREHENSIVE ANALYSIS REPORT")
        logger.info("="*80)
        
        # 1. Class Distribution Analysis
        self._plot_class_distribution(inventory_df)
        
        # 2. Tamper Type Analysis
        self._plot_tamper_type_analysis(inventory_df)
        
        # 3. Source Type Analysis
        self._plot_source_type_analysis(inventory_df)
        
        # 4. File Size Analysis
        self._plot_file_size_analysis(inventory_df)
        
        # 5. Mask Coverage Analysis
        self._plot_mask_coverage_analysis(inventory_df)
        
        # 6. Generate text report
        self._generate_text_report(inventory_df)
        
        logger.info("Comprehensive analysis report generated!")
    
    def _plot_class_distribution(self, df):
        """Plot class distribution"""
        plt.figure(figsize=(10, 6))
        
        # Overall distribution
        plt.subplot(1, 2, 1)
        class_counts = df['label'].value_counts()
        labels = ['Original', 'Tampered']
        colors = ['lightblue', 'lightcoral']
        plt.pie(class_counts.values, labels=labels, autopct='%1.1f%%', colors=colors)
        plt.title('Overall Class Distribution')
        
        # By category
        plt.subplot(1, 2, 2)
        category_counts = df.groupby(['category', 'label']).size().unstack(fill_value=0)
        category_counts.plot(kind='bar', stacked=True, color=['lightblue', 'lightcoral'])
        plt.title('Class Distribution by Category')
        plt.xlabel('Category')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Tampered'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/class_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_tamper_type_analysis(self, df):
        """Plot tamper type analysis"""
        plt.figure(figsize=(12, 8))
        
        # Tamper type distribution
        plt.subplot(2, 2, 1)
        tamper_counts = df['tamper_type'].value_counts()
        plt.pie(tamper_counts.values, labels=tamper_counts.index, autopct='%1.1f%%')
        plt.title('Tamper Type Distribution')
        
        # Tamper type by class
        plt.subplot(2, 2, 2)
        tamper_class = df.groupby(['tamper_type', 'label']).size().unstack(fill_value=0)
        tamper_class.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Tamper Types by Class')
        plt.xlabel('Tamper Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Tampered'])
        
        # Mask availability by tamper type
        plt.subplot(2, 2, 3)
        mask_tamper = df.groupby(['tamper_type', 'has_mask']).size().unstack(fill_value=0)
        mask_tamper.plot(kind='bar', color=['lightgray', 'darkgreen'])
        plt.title('Mask Availability by Tamper Type')
        plt.xlabel('Tamper Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['No Mask', 'Has Mask'])
        
        # File size by tamper type
        plt.subplot(2, 2, 4)
        df.boxplot(column='file_size', by='tamper_type', ax=plt.gca())
        plt.title('File Size Distribution by Tamper Type')
        plt.xlabel('Tamper Type')
        plt.ylabel('File Size (bytes)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/tamper_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_source_type_analysis(self, df):
        """Plot source type analysis"""
        plt.figure(figsize=(12, 6))
        
        # Source type distribution
        plt.subplot(1, 2, 1)
        source_counts = df['source_type'].value_counts()
        plt.pie(source_counts.values, labels=source_counts.index, autopct='%1.1f%%')
        plt.title('Source Type Distribution')
        
        # Source type by class
        plt.subplot(1, 2, 2)
        source_class = df.groupby(['source_type', 'label']).size().unstack(fill_value=0)
        source_class.plot(kind='bar', color=['lightblue', 'lightcoral'])
        plt.title('Source Types by Class')
        plt.xlabel('Source Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(['Original', 'Tampered'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/source_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_file_size_analysis(self, df):
        """Plot file size analysis"""
        plt.figure(figsize=(12, 6))
        
        # File size distribution
        plt.subplot(1, 2, 1)
        df[df['label'] == 0]['file_size'].hist(alpha=0.7, label='Original', bins=20)
        df[df['label'] == 1]['file_size'].hist(alpha=0.7, label='Tampered', bins=20)
        plt.xlabel('File Size (bytes)')
        plt.ylabel('Frequency')
        plt.title('File Size Distribution by Class')
        plt.legend()
        
        # File size by category
        plt.subplot(1, 2, 2)
        df.boxplot(column='file_size', by='category', ax=plt.gca())
        plt.title('File Size by Category')
        plt.xlabel('Category')
        plt.ylabel('File Size (bytes)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/file_size_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_mask_coverage_analysis(self, df):
        """Plot mask coverage analysis"""
        plt.figure(figsize=(10, 6))
        
        # Mask coverage by class
        mask_coverage = df.groupby(['label', 'has_mask']).size().unstack(fill_value=0)
        mask_coverage.plot(kind='bar', color=['lightgray', 'darkgreen'])
        plt.title('Mask Coverage by Class')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['Original', 'Tampered'], rotation=0)
        plt.legend(['No Mask', 'Has Mask'])
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/visualizations/mask_coverage_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_text_report(self, df):
        """Generate comprehensive text report"""
        report_path = f"{self.output_path}/reports/comprehensive_analysis_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("COMPREHENSIVE OBJECTIVE 2 DATA ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Dataset Overview
            f.write("DATASET OVERVIEW\n")
            f.write("-"*40 + "\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Original Samples: {len(df[df['label'] == 0])}\n")
            f.write(f"Tampered Samples: {len(df[df['label'] == 1])}\n")
            f.write(f"Class Balance: {len(df[df['label'] == 0])/len(df)*100:.1f}% Original, {len(df[df['label'] == 1])/len(df)*100:.1f}% Tampered\n\n")
            
            # Source Analysis
            f.write("SOURCE ANALYSIS\n")
            f.write("-"*40 + "\n")
            for source_type in df['source_type'].unique():
                count = len(df[df['source_type'] == source_type])
                f.write(f"{source_type}: {count} samples\n")
            f.write("\n")
            
            # Tamper Type Analysis
            f.write("TAMPER TYPE ANALYSIS\n")
            f.write("-"*40 + "\n")
            for tamper_type in df['tamper_type'].unique():
                count = len(df[df['tamper_type'] == tamper_type])
                f.write(f"{tamper_type}: {count} samples\n")
            f.write("\n")
            
            # Mask Analysis
            f.write("MASK ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Images with masks: {len(df[df['has_mask'] == True])}\n")
            f.write(f"Images without masks: {len(df[df['has_mask'] == False])}\n")
            f.write(f"Mask coverage: {len(df[df['has_mask'] == True])/len(df)*100:.1f}%\n\n")
            
            # File Size Analysis
            f.write("FILE SIZE ANALYSIS\n")
            f.write("-"*40 + "\n")
            f.write(f"Average file size: {df['file_size'].mean():.0f} bytes\n")
            f.write(f"Median file size: {df['file_size'].median():.0f} bytes\n")
            f.write(f"Min file size: {df['file_size'].min():.0f} bytes\n")
            f.write(f"Max file size: {df['file_size'].max():.0f} bytes\n\n")
            
            # Recommendations
            f.write("RECOMMENDATIONS FOR 90%+ ACCURACY\n")
            f.write("-"*40 + "\n")
            f.write("1. Use ALL available data sources (PDFs converted to images + existing images)\n")
            f.write("2. Utilize binary masks for region-specific feature extraction\n")
            f.write("3. Implement class balancing techniques (SMOTE, ADASYN)\n")
            f.write("4. Extract mask-aware features for tampered regions\n")
            f.write("5. Use ensemble methods combining ML and DL approaches\n")
            f.write("6. Implement data augmentation for minority classes\n")
            f.write("7. Use description files for additional metadata features\n")
            
        logger.info(f"Text report saved: {report_path}")
    
    def run_complete_analysis(self):
        """Run complete data analysis pipeline"""
        logger.info("🚀 STARTING COMPREHENSIVE OBJECTIVE 2 DATA ANALYSIS")
        logger.info("="*80)
        
        # Step 1: Convert PDFs to images
        self.convert_pdfs_to_images()
        
        # Step 2: Analyze tampered images folder
        self.analyze_tampered_images_folder()
        
        # Step 3: Create comprehensive inventory
        inventory_df = self.create_comprehensive_inventory()
        
        # Step 4: Generate comprehensive analysis report
        self.generate_comprehensive_analysis_report(inventory_df)
        
        logger.info("✅ COMPREHENSIVE DATA ANALYSIS COMPLETED!")
        logger.info(f"Results saved in: {self.output_path}")
        logger.info(f"Converted images saved in: {self.converted_images_path}")
        
        return inventory_df

def main():
    """Main function to run comprehensive data analysis"""
    analyzer = ComprehensiveObjective2DataAnalyzer()
    inventory_df = analyzer.run_complete_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    print(f"Total samples analyzed: {len(inventory_df)}")
    print(f"Original samples: {len(inventory_df[inventory_df['label'] == 0])}")
    print(f"Tampered samples: {len(inventory_df[inventory_df['label'] == 1])}")
    print(f"Class balance: {len(inventory_df[inventory_df['label'] == 0])/len(inventory_df)*100:.1f}% Original")
    print("\nNext step: Run data preprocessing with this comprehensive inventory!")

if __name__ == "__main__":
    main()
