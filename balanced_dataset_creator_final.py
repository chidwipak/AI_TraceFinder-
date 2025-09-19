#!/usr/bin/env python3
"""
Create perfectly balanced dataset by converting PDFs from Originals/ folder
to match the tampered images count (102 tampered vs 34 originals)
"""

import os
import sys
import random
import numpy as np
import cv2
import tifffile
from pdf2image import convert_from_path
from PIL import Image
import shutil

def convert_pdf_to_tif(pdf_path, output_dir, prefix="converted"):
    """Convert PDF to TIF format"""
    try:
        # Convert PDF to images
        images = convert_from_path(pdf_path, dpi=300, first_page=1, last_page=1)
        
        if not images:
            return None
        
        # Take first page
        image = images[0]
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"{prefix}_{base_name}.tif")
        
        # Save as TIF
        image.save(output_path, 'TIFF', compression='lzw')
        
        return output_path
        
    except Exception as e:
        print(f"Error converting {pdf_path}: {e}")
        return None

def collect_pdf_files():
    """Collect all PDF files from Originals/ folder"""
    pdf_files = []
    
    # Check official folder
    official_dir = "The SUPATLANTIQUE dataset/Originals/official"
    if os.path.exists(official_dir):
        for scanner_dir in os.listdir(official_dir):
            scanner_path = os.path.join(official_dir, scanner_dir)
            if os.path.isdir(scanner_path):
                for file in os.listdir(scanner_path):
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(scanner_path, file))
    
    # Check wikipedia folder (PDFs are directly in this folder)
    wikipedia_dir = "The SUPATLANTIQUE dataset/Originals/wikipedia"
    if os.path.exists(wikipedia_dir):
        for file in os.listdir(wikipedia_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(wikipedia_dir, file))
    
    return pdf_files

def create_balanced_dataset():
    """Create balanced dataset by converting PDFs to match tampered count"""
    print("🔄 Creating Balanced Dataset")
    print("=" * 50)
    
    # Count current images
    tampered_count = 102
    original_count = 34
    needed_originals = tampered_count - original_count
    
    print(f"📊 Current counts:")
    print(f"   Tampered images: {tampered_count}")
    print(f"   Original images: {original_count}")
    print(f"   Need to add: {needed_originals} more originals")
    
    # Collect PDF files
    print(f"\n🔍 Collecting PDF files from Originals/ folder...")
    pdf_files = collect_pdf_files()
    print(f"   Found {len(pdf_files)} PDF files")
    
    if len(pdf_files) < needed_originals:
        print(f"⚠️ Warning: Only {len(pdf_files)} PDFs available, need {needed_originals}")
        needed_originals = len(pdf_files)
    
    # Create output directory for converted images
    output_dir = "The SUPATLANTIQUE dataset/Tampered images/Original_Converted"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PDFs to TIF
    print(f"\n🔄 Converting {needed_originals} PDFs to TIF...")
    converted_files = []
    
    # Sample PDFs randomly
    selected_pdfs = random.sample(pdf_files, needed_originals)
    
    for i, pdf_path in enumerate(selected_pdfs):
        print(f"   Converting {i+1}/{needed_originals}: {os.path.basename(pdf_path)}")
        
        output_path = convert_pdf_to_tif(pdf_path, output_dir, f"conv_{i+1:03d}")
        if output_path:
            converted_files.append(output_path)
            print(f"     ✅ Saved: {os.path.basename(output_path)}")
        else:
            print(f"     ❌ Failed to convert")
    
    print(f"\n📊 Conversion Results:")
    print(f"   Successfully converted: {len(converted_files)}")
    print(f"   Failed conversions: {needed_originals - len(converted_files)}")
    
    # Copy converted files to Original folder
    original_dir = "The SUPATLANTIQUE dataset/Tampered images/Original"
    print(f"\n📁 Copying converted files to {original_dir}...")
    
    for converted_file in converted_files:
        filename = os.path.basename(converted_file)
        dest_path = os.path.join(original_dir, filename)
        shutil.copy2(converted_file, dest_path)
        print(f"   ✅ Copied: {filename}")
    
    # Verify final counts
    final_original_count = len([f for f in os.listdir(original_dir) if f.endswith('.tif')])
    final_tampered_count = len([f for f in os.listdir("The SUPATLANTIQUE dataset/Tampered images/Tampered/Copy-move") if f.endswith('.tif')]) + \
                          len([f for f in os.listdir("The SUPATLANTIQUE dataset/Tampered images/Tampered/Retouching") if f.endswith('.tif')]) + \
                          len([f for f in os.listdir("The SUPATLANTIQUE dataset/Tampered images/Tampered/Splicing") if f.endswith('.tif')])
    
    print(f"\n🎯 Final Balanced Dataset:")
    print(f"   Original images: {final_original_count}")
    print(f"   Tampered images: {final_tampered_count}")
    print(f"   Balance ratio: {final_original_count/final_tampered_count:.2f}:1")
    
    if final_original_count >= final_tampered_count:
        print(f"   ✅ Dataset is balanced!")
        return True
    else:
        print(f"   ⚠️ Dataset still imbalanced")
        return False

def main():
    print("🚀 Balanced Dataset Creator")
    print("=" * 50)
    
    success = create_balanced_dataset()
    
    if success:
        print(f"\n🎉 SUCCESS! Balanced dataset created successfully!")
        print(f"   Ready for training with equal original and tampered samples")
    else:
        print(f"\n❌ FAILED! Could not create balanced dataset")
    
    return success

if __name__ == "__main__":
    main()
