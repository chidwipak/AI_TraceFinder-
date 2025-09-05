#!/usr/bin/env python3
"""
Preprocessing Pipeline for AI_TraceFinder - Objective 1: Scanner Identification
Handles metadata creation, image preprocessing, dataset splitting, and organization
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import tifffile
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging
from datetime import datetime
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ScannerIdentificationPreprocessor:
    def __init__(self, dataset_path="The SUPATLANTIQUE dataset", output_path="preprocessed"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.patch_size = (256, 256)
        self.random_seed = 42
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize data structures
        self.metadata = []
        self.failed_images = []
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/train",
            f"{self.output_path}/test",
            f"{self.output_path}/metadata"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def build_metadata_table(self):
        """Task 1: Build metadata table with filepath, scanner_id, dpi, source"""
        logger.info("Building metadata table...")
        
        # Define scanner folders to process
        scanner_folders = {
            'Official': ['Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2',
                        'EpsonV39-1', 'EpsonV39-2', 'EpsonV370-1', 'EpsonV370-2', 'EpsonV550', 'HP'],
            'Wikipedia': ['Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2',
                         'EpsonV39-1', 'EpsonV39-2', 'EpsonV370-1', 'EpsonV370-2', 'EpsonV550', 'HP']
        }
        
        for source in ['Official', 'Wikipedia']:
            source_path = os.path.join(self.dataset_path, source)
            if not os.path.exists(source_path):
                logger.warning(f"Source path not found: {source_path}")
                continue
                
            for scanner_id in scanner_folders[source]:
                scanner_path = os.path.join(source_path, scanner_id)
                if not os.path.exists(scanner_path):
                    logger.warning(f"Scanner path not found: {scanner_path}")
                    continue
                
                # Process DPI folders (150 and 300)
                for dpi in ['150', '300']:
                    dpi_path = os.path.join(scanner_path, dpi)
                    if not os.path.exists(dpi_path):
                        logger.warning(f"DPI path not found: {dpi_path}")
                        continue
                    
                    # Get all TIFF files
                    tiff_files = [f for f in os.listdir(dpi_path) if f.endswith('.tif')]
                    
                    for tiff_file in tiff_files:
                        filepath = os.path.join(dpi_path, tiff_file)
                        self.metadata.append({
                            'filepath': filepath,
                            'scanner_id': scanner_id,
                            'dpi': int(dpi),
                            'source': source
                        })
        
        # Convert to DataFrame
        self.metadata_df = pd.DataFrame(self.metadata)
        logger.info(f"Metadata table built with {len(self.metadata_df)} images")
        
        # Save metadata
        self.metadata_df.to_csv(f"{self.output_path}/metadata/full_metadata.csv", index=False)
        logger.info(f"Metadata saved to {self.output_path}/metadata/full_metadata.csv")
        
        return self.metadata_df
    
    def robust_tiff_reading(self, filepath):
        """Task 2: Robust TIFF reading with fallback methods"""
        try:
            # First try tifffile
            image = tifffile.imread(filepath)
            return image
        except Exception as e1:
            try:
                # Fallback to imageio
                image = imageio.imread(filepath)
                return image
            except Exception as e2:
                # Log failure
                self.failed_images.append({
                    'filepath': filepath,
                    'tifffile_error': str(e1),
                    'imageio_error': str(e2)
                })
                logger.error(f"Failed to read {filepath}: {e1}, {e2}")
                return None
    
    def preprocess_image(self, image):
        """Task 2: Preprocess image (grayscale, normalize, resize)"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB to grayscale
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            elif image.shape[2] == 4:
                # RGBA to grayscale
                image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
        
        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0
        
        # Resize to patch size
        from PIL import Image
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize(self.patch_size, Image.Resampling.LANCZOS)
        image = np.array(pil_image).astype(np.float32) / 255.0
        
        return image
    
    def process_images(self):
        """Task 2: Process all images and save preprocessed patches"""
        logger.info("Processing images...")
        
        processed_count = 0
        
        for idx, row in self.metadata_df.iterrows():
            filepath = row['filepath']
            scanner_id = row['scanner_id']
            dpi = row['dpi']
            source = row['source']
            
            # Read image
            image = self.robust_tiff_reading(filepath)
            if image is None:
                continue
            
            # Preprocess image
            try:
                processed_image = self.preprocess_image(image)
                
                # Save preprocessed image
                output_dir = f"{self.output_path}/temp/{scanner_id}/{dpi}"
                os.makedirs(output_dir, exist_ok=True)
                
                filename = os.path.basename(filepath).replace('.tif', '.png')
                output_path = os.path.join(output_dir, filename)
                
                # Save as PNG
                pil_image = Image.fromarray((processed_image * 255).astype(np.uint8))
                pil_image.save(output_path)
                
                processed_count += 1
                
                if processed_count % 100 == 0:
                    logger.info(f"Processed {processed_count} images...")
                    
            except Exception as e:
                self.failed_images.append({
                    'filepath': filepath,
                    'preprocessing_error': str(e)
                })
                logger.error(f"Failed to preprocess {filepath}: {e}")
        
        logger.info(f"Image processing complete. Processed: {processed_count}, Failed: {len(self.failed_images)}")
        
        # Save failed images log
        if self.failed_images:
            failed_df = pd.DataFrame(self.failed_images)
            failed_df.to_csv(f"{self.output_path}/metadata/failed_images.csv", index=False)
            logger.info(f"Failed images logged to {self.output_path}/metadata/failed_images.csv")
    
    def split_dataset(self):
        """Task 3: Perform 90/10 train/test split with stratification"""
        logger.info("Splitting dataset...")
        
        # Set random seed
        np.random.seed(self.random_seed)
        
        # Create stratification column
        self.metadata_df['stratify_col'] = self.metadata_df['scanner_id'] + '_' + self.metadata_df['dpi'].astype(str)
        
        # Perform split
        train_df, test_df = train_test_split(
            self.metadata_df,
            test_size=0.1,
            stratify=self.metadata_df['stratify_col'],
            random_state=self.random_seed
        )
        
        # Remove stratification column
        train_df = train_df.drop('stratify_col', axis=1)
        test_df = test_df.drop('stratify_col', axis=1)
        
        # Save splits
        train_df.to_csv(f"{self.output_path}/metadata/metadata_train.csv", index=False)
        test_df.to_csv(f"{self.output_path}/metadata/metadata_test.csv", index=False)
        
        logger.info(f"Dataset split complete. Train: {len(train_df)}, Test: {len(test_df)}")
        
        return train_df, test_df
    
    def organize_preprocessed_data(self, train_df, test_df):
        """Task 4: Organize preprocessed data into train/test folders"""
        logger.info("Organizing preprocessed data...")
        
        # Move files to final structure
        for df, split in [(train_df, 'train'), (test_df, 'test')]:
            for idx, row in df.iterrows():
                scanner_id = row['scanner_id']
                dpi = row['dpi']
                source_file = row['filepath']
                
                # Source path in temp directory
                temp_path = f"{self.output_path}/temp/{scanner_id}/{dpi}/{os.path.basename(source_file).replace('.tif', '.png')}"
                
                # Destination path
                dest_dir = f"{self.output_path}/{split}/{scanner_id}/{dpi}"
                os.makedirs(dest_dir, exist_ok=True)
                dest_path = os.path.join(dest_dir, os.path.basename(source_file).replace('.tif', '.png'))
                
                # Move file
                if os.path.exists(temp_path):
                    shutil.move(temp_path, dest_path)
        
        # Clean up temp directory
        if os.path.exists(f"{self.output_path}/temp"):
            shutil.rmtree(f"{self.output_path}/temp")
        
        logger.info("Data organization complete")
    
    def generate_report(self, train_df, test_df):
        """Task 5: Generate preprocessing report"""
        logger.info("Generating report...")
        
        # Dataset summary
        total_images = len(self.metadata_df)
        processed_images = total_images - len(self.failed_images)
        
        print("\n" + "="*60)
        print("PREPROCESSING REPORT - OBJECTIVE 1: SCANNER IDENTIFICATION")
        print("="*60)
        
        print(f"\nDataset Summary:")
        print(f"Total images: {total_images}")
        print(f"Successfully processed: {processed_images}")
        print(f"Failed images: {len(self.failed_images)}")
        print(f"Success rate: {processed_images/total_images*100:.2f}%")
        
        print(f"\nTrain/Test Split:")
        print(f"Training images: {len(train_df)}")
        print(f"Testing images: {len(test_df)}")
        
        print(f"\nPer-Scanner Distribution:")
        scanner_counts = self.metadata_df['scanner_id'].value_counts()
        for scanner, count in scanner_counts.items():
            print(f"  {scanner}: {count}")
        
        print(f"\nPer-DPI Distribution:")
        dpi_counts = self.metadata_df['dpi'].value_counts()
        for dpi, count in dpi_counts.items():
            print(f"  {dpi} DPI: {count}")
        
        print(f"\nPer-Source Distribution:")
        source_counts = self.metadata_df['source'].value_counts()
        for source, count in source_counts.items():
            print(f"  {source}: {count}")
        
        # Save report
        report_path = f"{self.output_path}/metadata/preprocessing_report.txt"
        with open(report_path, 'w') as f:
            f.write("PREPROCESSING REPORT - OBJECTIVE 1: SCANNER IDENTIFICATION\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total images: {total_images}\n")
            f.write(f"Successfully processed: {processed_images}\n")
            f.write(f"Failed images: {len(self.failed_images)}\n")
            f.write(f"Success rate: {processed_images/total_images*100:.2f}%\n\n")
            f.write(f"Training images: {len(train_df)}\n")
            f.write(f"Testing images: {len(test_df)}\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def display_sample_images(self, train_df, num_samples=6):
        """Task 5: Display sample preprocessed images"""
        logger.info("Displaying sample images...")
        
        # Get random samples
        samples = train_df.sample(n=min(num_samples, len(train_df)), random_state=self.random_seed)
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()
        
        for idx, (_, row) in enumerate(samples.iterrows()):
            scanner_id = row['scanner_id']
            dpi = row['dpi']
            source = row['source']
            
            # Load preprocessed image
            image_path = f"{self.output_path}/train/{scanner_id}/{dpi}/{os.path.basename(row['filepath']).replace('.tif', '.png')}"
            
            if os.path.exists(image_path):
                image = Image.open(image_path)
                axes[idx].imshow(image, cmap='gray')
                axes[idx].set_title(f"{scanner_id}\n{dpi} DPI - {source}")
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, 'Image not found', ha='center', va='center')
                axes[idx].set_title(f"{scanner_id}\n{dpi} DPI - {source}")
                axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/metadata/sample_images.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Sample images saved to {self.output_path}/metadata/sample_images.png")
    
    def run_full_pipeline(self):
        """Run the complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline for Objective 1...")
        
        # Task 1: Build metadata table
        self.build_metadata_table()
        
        # Task 2: Process images
        self.process_images()
        
        # Task 3: Split dataset
        train_df, test_df = self.split_dataset()
        
        # Task 4: Organize data
        self.organize_preprocessed_data(train_df, test_df)
        
        # Task 5: Generate report and display samples
        self.generate_report(train_df, test_df)
        self.display_sample_images(train_df)
        
        logger.info("Preprocessing pipeline complete!")

if __name__ == "__main__":
    preprocessor = ScannerIdentificationPreprocessor()
    preprocessor.run_full_pipeline()

