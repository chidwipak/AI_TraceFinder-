#!/usr/bin/env python3
"""
Create balanced dataset for tampered detection by converting PDFs and balancing classes
"""

import os
import pandas as pd
import numpy as np
from pdf2image import convert_from_path
from PIL import Image
import cv2
import glob
import shutil
from pathlib import Path
import random

class BalancedDatasetCreator:
    def __init__(self):
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.output_path = "balanced_objective2_dataset"
        self.originals_pdf_path = os.path.join(self.dataset_path, "Originals", "official")
        self.tampered_path = os.path.join(self.dataset_path, "Tampered images", "Tampered")
        
    def analyze_current_dataset(self):
        """Analyze current dataset distribution"""
        print("🔍 Analyzing Current Dataset")
        print("=" * 50)
        
        # Count tampered images
        tampered_files = []
        for root, dirs, files in os.walk(self.tampered_path):
            for file in files:
                if file.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                    tampered_files.append(os.path.join(root, file))
        
        print(f"📊 Tampered images found: {len(tampered_files)}")
        
        # Count original PDFs available
        pdf_files = glob.glob(os.path.join(self.originals_pdf_path, "*.pdf"))
        print(f"📊 Original PDFs available: {len(pdf_files)}")
        
        # Count existing original images (already converted)
        existing_original_images = []
        original_image_dirs = [
            os.path.join(self.dataset_path, "Official"),
            os.path.join(self.dataset_path, "Wikipedia"),
            os.path.join(self.dataset_path, "Flatfield")
        ]
        
        for img_dir in original_image_dirs:
            if os.path.exists(img_dir):
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                            existing_original_images.append(os.path.join(root, file))
        
        print(f"📊 Existing original images: {len(existing_original_images)}")
        
        return len(tampered_files), len(pdf_files), len(existing_original_images), tampered_files
    
    def convert_pdfs_to_images(self, num_needed):
        """Convert selected PDFs to images"""
        print(f"\n📄 Converting {num_needed} PDFs to images...")
        
        # Create output directory
        converted_dir = os.path.join(self.output_path, "converted_originals")
        os.makedirs(converted_dir, exist_ok=True)
        
        pdf_files = glob.glob(os.path.join(self.originals_pdf_path, "*.pdf"))
        selected_pdfs = random.sample(pdf_files, min(num_needed, len(pdf_files)))
        
        converted_images = []
        
        for i, pdf_path in enumerate(selected_pdfs):
            try:
                print(f"Converting {os.path.basename(pdf_path)} ({i+1}/{len(selected_pdfs)})")
                
                # Convert PDF to image (first page only)
                images = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=300)
                
                if images:
                    # Save as TIFF
                    output_filename = f"original_{i+1:03d}.tif"
                    output_path = os.path.join(converted_dir, output_filename)
                    images[0].save(output_path, 'TIFF')
                    converted_images.append(output_path)
                    
            except Exception as e:
                print(f"❌ Error converting {pdf_path}: {e}")
                continue
        
        print(f"✅ Successfully converted {len(converted_images)} PDFs to images")
        return converted_images
    
    def create_balanced_dataset(self):
        """Create balanced dataset with equal original and tampered images"""
        print("\n🎯 Creating Balanced Dataset")
        print("=" * 50)
        
        # Analyze current dataset
        num_tampered, num_pdfs, num_existing_originals, tampered_files = self.analyze_current_dataset()
        
        # Determine how many PDFs to convert
        target_originals = num_tampered  # Match number of tampered images
        pdfs_to_convert = max(0, target_originals - min(num_existing_originals, target_originals))
        
        print(f"\n📋 Dataset Balance Plan:")
        print(f"Target original images: {target_originals}")
        print(f"Existing original images: {num_existing_originals}")
        print(f"PDFs to convert: {pdfs_to_convert}")
        
        # Create output directory structure
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images", "original"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images", "tampered"), exist_ok=True)
        
        # Convert PDFs if needed
        converted_images = []
        if pdfs_to_convert > 0:
            converted_images = self.convert_pdfs_to_images(pdfs_to_convert)
        
        # Collect existing original images (sample if too many)
        existing_original_images = []
        original_image_dirs = [
            os.path.join(self.dataset_path, "Official"),
            os.path.join(self.dataset_path, "Wikipedia"),
            os.path.join(self.dataset_path, "Flatfield")
        ]
        
        for img_dir in original_image_dirs:
            if os.path.exists(img_dir):
                for root, dirs, files in os.walk(img_dir):
                    for file in files:
                        if file.lower().endswith(('.tif', '.tiff', '.jpg', '.jpeg', '.png')):
                            existing_original_images.append(os.path.join(root, file))
        
        # Sample existing images if we have too many
        if len(existing_original_images) > target_originals - len(converted_images):
            existing_original_images = random.sample(
                existing_original_images, 
                target_originals - len(converted_images)
            )
        
        # Copy original images to balanced dataset
        print(f"\n📋 Copying {len(existing_original_images)} existing original images...")
        original_image_paths = []
        
        for i, img_path in enumerate(existing_original_images):
            try:
                filename = f"original_existing_{i+1:03d}.tif"
                output_path = os.path.join(self.output_path, "images", "original", filename)
                
                # Load and save as TIFF to ensure consistency
                if img_path.lower().endswith(('.tif', '.tiff')):
                    shutil.copy2(img_path, output_path)
                else:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'TIFF')
                
                original_image_paths.append(output_path)
                
            except Exception as e:
                print(f"❌ Error copying {img_path}: {e}")
                continue
        
        # Copy converted images
        for i, img_path in enumerate(converted_images):
            filename = f"original_converted_{i+1:03d}.tif"
            output_path = os.path.join(self.output_path, "images", "original", filename)
            shutil.copy2(img_path, output_path)
            original_image_paths.append(output_path)
        
        # Copy tampered images (sample if too many to match originals)
        print(f"\n📋 Copying {min(len(tampered_files), len(original_image_paths))} tampered images...")
        
        if len(tampered_files) > len(original_image_paths):
            tampered_files = random.sample(tampered_files, len(original_image_paths))
        
        tampered_image_paths = []
        for i, img_path in enumerate(tampered_files):
            try:
                filename = f"tampered_{i+1:03d}.tif"
                output_path = os.path.join(self.output_path, "images", "tampered", filename)
                
                # Load and save as TIFF
                if img_path.lower().endswith(('.tif', '.tiff')):
                    shutil.copy2(img_path, output_path)
                else:
                    img = Image.open(img_path)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(output_path, 'TIFF')
                
                tampered_image_paths.append(output_path)
                
            except Exception as e:
                print(f"❌ Error copying {img_path}: {e}")
                continue
        
        # Create metadata CSV
        self.create_metadata(original_image_paths, tampered_image_paths)
        
        print(f"\n✅ Balanced Dataset Created!")
        print(f"📊 Final counts:")
        print(f"   Original images: {len(original_image_paths)}")
        print(f"   Tampered images: {len(tampered_image_paths)}")
        print(f"   Total images: {len(original_image_paths) + len(tampered_image_paths)}")
        print(f"   Balance ratio: 50:50")
        
        return len(original_image_paths), len(tampered_image_paths)
    
    def create_metadata(self, original_paths, tampered_paths):
        """Create metadata CSV for the balanced dataset"""
        print("\n📝 Creating metadata...")
        
        metadata = []
        
        # Add original images
        for path in original_paths:
            metadata.append({
                'image_path': path,
                'label': 0,  # 0 for original
                'category': 'original',
                'source': 'balanced_dataset'
            })
        
        # Add tampered images
        for path in tampered_paths:
            metadata.append({
                'image_path': path,
                'label': 1,  # 1 for tampered
                'category': 'tampered',
                'source': 'balanced_dataset'
            })
        
        # Create DataFrame and shuffle
        df = pd.DataFrame(metadata)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Save metadata
        metadata_path = os.path.join(self.output_path, "balanced_metadata.csv")
        df.to_csv(metadata_path, index=False)
        
        print(f"✅ Metadata saved to {metadata_path}")
        
        # Create train/test splits (80/20)
        train_size = int(0.8 * len(df))
        
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Save splits
        train_path = os.path.join(self.output_path, "train_split.csv")
        test_path = os.path.join(self.output_path, "test_split.csv")
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        print(f"✅ Train split saved to {train_path} ({len(train_df)} images)")
        print(f"✅ Test split saved to {test_path} ({len(test_df)} images)")
        
        return metadata_path, train_path, test_path

def main():
    creator = BalancedDatasetCreator()
    creator.create_balanced_dataset()

if __name__ == "__main__":
    main()

