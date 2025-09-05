#!/usr/bin/env python3
"""
PRNU Fingerprint Extraction for AI_TraceFinder - Objective 1: Scanner Identification
Extracts Photo-Response Non-Uniformity patterns from flatfield images to create scanner fingerprints
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import tifffile
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
import pywt
import pickle
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PRNUFingerprintExtractor:
    def __init__(self, dataset_path="The SUPATLANTIQUE dataset", output_path="prnu_fingerprints"):
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.patch_size = (256, 256)
        self.random_seed = 42
        
        # Create output directories
        self.create_output_directories()
        
        # Initialize data structures
        self.flatfield_images = {}
        self.prnu_fingerprints = {}
        self.failed_images = []
        
    def create_output_directories(self):
        """Create necessary output directories"""
        dirs = [
            f"{self.output_path}/fingerprints",
            f"{self.output_path}/visualizations",
            f"{self.output_path}/metadata"
        ]
        
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            logger.info(f"Created directory: {dir_path}")
    
    def load_flatfield_images(self):
        """Task 1: Load flatfield images for each scanner × dpi"""
        logger.info("Loading flatfield images...")
        
        flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        if not os.path.exists(flatfield_path):
            logger.error(f"Flatfield directory not found: {flatfield_path}")
            return
        
        # Scanner folders
        scanner_folders = ['Canon120-1', 'Canon120-2', 'Canon220', 'Canon9000-1', 'Canon9000-2',
                          'EpsonV39-1', 'EpsonV39-2', 'EpsonV370-1', 'EpsonV370-2', 'EpsonV550', 'HP']
        
        for scanner_id in scanner_folders:
            scanner_path = os.path.join(flatfield_path, scanner_id)
            if not os.path.exists(scanner_path):
                logger.warning(f"Scanner flatfield path not found: {scanner_path}")
                continue
            
            self.flatfield_images[scanner_id] = {}
            
            # Process DPI files (150.tif and 300.tif)
            for dpi in ['150', '300']:
                dpi_filename = f"{dpi}.tif"
                filepath = os.path.join(scanner_path, dpi_filename)
                
                if not os.path.exists(filepath):
                    logger.warning(f"Flatfield file not found: {filepath}")
                    continue
                
                logger.info(f"Loading flatfield image for {scanner_id} at {dpi} DPI")
                
                # Load flatfield image
                image = self.robust_tiff_reading(filepath)
                if image is not None:
                    processed_image = self.preprocess_image(image)
                    self.flatfield_images[scanner_id][int(dpi)] = np.array([processed_image])
                    logger.info(f"Loaded flatfield image for {scanner_id} at {dpi} DPI")
                else:
                    self.failed_images.append({
                        'filepath': filepath,
                        'scanner_id': scanner_id,
                        'dpi': dpi,
                        'error': 'Failed to read image'
                    })
        
        logger.info(f"Flatfield loading complete. Loaded {len(self.flatfield_images)} scanners")
    
    def robust_tiff_reading(self, filepath):
        """Robust TIFF reading with fallback methods"""
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
                logger.error(f"Failed to read {filepath}: {e1}, {e2}")
                return None
    
    def preprocess_image(self, image):
        """Preprocess image (grayscale, normalize, resize)"""
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
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize(self.patch_size, Image.Resampling.LANCZOS)
        image = np.array(pil_image).astype(np.float32) / 255.0
        
        return image
    
    def extract_noise_residuals(self, image):
        """Task 2: Extract noise residuals using wavelet denoising"""
        # Apply wavelet denoising
        coeffs = pywt.wavedec2(image, 'db8', level=3)
        
        # Threshold coefficients to remove noise
        threshold = np.std(coeffs[0]) * 0.1
        coeffs_thresholded = []
        
        for i, coeff in enumerate(coeffs):
            if i == 0:
                # Keep approximation coefficients
                coeffs_thresholded.append(coeff)
            else:
                # Threshold detail coefficients (coeff is a tuple for wavedec2)
                if isinstance(coeff, tuple):
                    coeff_thresholded = tuple(pywt.threshold(c, threshold, mode='soft') for c in coeff)
                else:
                    coeff_thresholded = pywt.threshold(coeff, threshold, mode='soft')
                coeffs_thresholded.append(coeff_thresholded)
        
        # Reconstruct denoised image
        denoised_image = pywt.waverec2(coeffs_thresholded, 'db8')
        
        # Ensure same size as original
        if denoised_image.shape != image.shape:
            denoised_image = denoised_image[:image.shape[0], :image.shape[1]]
        
        # Calculate residuals
        residuals = image - denoised_image
        
        # Normalize residuals (zero-mean, unit-std)
        residuals = residuals - np.mean(residuals)
        residuals = residuals / (np.std(residuals) + 1e-8)  # Add small epsilon to avoid division by zero
        
        return residuals
    
    def create_prnu_fingerprints(self):
        """Task 3: Create PRNU fingerprints by averaging residuals per scanner × dpi"""
        logger.info("Creating PRNU fingerprints...")
        
        for scanner_id, dpi_images in self.flatfield_images.items():
            self.prnu_fingerprints[scanner_id] = {}
            
            for dpi, images in dpi_images.items():
                logger.info(f"Processing {scanner_id} at {dpi} DPI with {len(images)} images")
                
                # Extract residuals for all images
                residuals_list = []
                for image in images:
                    residuals = self.extract_noise_residuals(image)
                    residuals_list.append(residuals)
                
                # Average residuals to create stable PRNU fingerprint
                if residuals_list:
                    prnu_fingerprint = np.mean(residuals_list, axis=0)
                    
                    # Additional normalization
                    prnu_fingerprint = (prnu_fingerprint - np.mean(prnu_fingerprint)) / (np.std(prnu_fingerprint) + 1e-8)
                    
                    self.prnu_fingerprints[scanner_id][dpi] = prnu_fingerprint
                    
                    logger.info(f"Created PRNU fingerprint for {scanner_id} at {dpi} DPI")
        
        logger.info(f"PRNU fingerprint creation complete. Created {len(self.prnu_fingerprints)} scanner fingerprints")
    
    def save_fingerprints(self):
        """Task 4: Save fingerprints in NumPy/Pickle format"""
        logger.info("Saving PRNU fingerprints...")
        
        # Save as NumPy arrays
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            for dpi, fingerprint in dpi_fingerprints.items():
                filename = f"{scanner_id}_{dpi}dpi_prnu.npy"
                filepath = os.path.join(self.output_path, "fingerprints", filename)
                np.save(filepath, fingerprint)
                logger.info(f"Saved {filename}")
        
        # Save as pickle for easy loading
        pickle_filepath = os.path.join(self.output_path, "fingerprints", "all_prnu_fingerprints.pkl")
        with open(pickle_filepath, 'wb') as f:
            pickle.dump(self.prnu_fingerprints, f)
        logger.info(f"Saved all fingerprints to {pickle_filepath}")
        
        # Save metadata
        metadata = []
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            for dpi, fingerprint in dpi_fingerprints.items():
                metadata.append({
                    'scanner_id': scanner_id,
                    'dpi': dpi,
                    'fingerprint_shape': fingerprint.shape,
                    'fingerprint_mean': float(np.mean(fingerprint)),
                    'fingerprint_std': float(np.std(fingerprint)),
                    'filename': f"{scanner_id}_{dpi}dpi_prnu.npy"
                })
        
        metadata_df = pd.DataFrame(metadata)
        metadata_filepath = os.path.join(self.output_path, "metadata", "prnu_fingerprints_metadata.csv")
        metadata_df.to_csv(metadata_filepath, index=False)
        logger.info(f"Saved metadata to {metadata_filepath}")
    
    def visualize_fingerprints(self):
        """Task 5: Visualize PRNU fingerprints in a grid"""
        logger.info("Creating PRNU fingerprint visualizations...")
        
        # Create a grid of fingerprints
        scanners = list(self.prnu_fingerprints.keys())
        dpis = [150, 300]
        
        fig, axes = plt.subplots(len(scanners), len(dpis), figsize=(15, 25))
        
        if len(scanners) == 1:
            axes = axes.reshape(1, -1)
        
        for i, scanner_id in enumerate(scanners):
            for j, dpi in enumerate(dpis):
                ax = axes[i, j]
                
                if dpi in self.prnu_fingerprints[scanner_id]:
                    fingerprint = self.prnu_fingerprints[scanner_id][dpi]
                    
                    # Display fingerprint
                    im = ax.imshow(fingerprint, cmap='viridis', aspect='equal')
                    ax.set_title(f"{scanner_id}\n{dpi} DPI", fontsize=10)
                    ax.axis('off')
                    
                    # Add colorbar
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"{scanner_id}\n{dpi} DPI", fontsize=10)
                    ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, "visualizations", "prnu_fingerprints_grid.png"), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create individual fingerprint plots
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            if len(dpi_fingerprints) == 0:
                continue
                
            fig, axes = plt.subplots(1, len(dpi_fingerprints), figsize=(12, 4))
            
            if len(dpi_fingerprints) == 1:
                axes = [axes]
            
            for i, (dpi, fingerprint) in enumerate(dpi_fingerprints.items()):
                im = axes[i].imshow(fingerprint, cmap='viridis', aspect='equal')
                axes[i].set_title(f"{scanner_id} - {dpi} DPI")
                axes[i].axis('off')
                plt.colorbar(im, ax=axes[i])
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_path, "visualizations", f"{scanner_id}_fingerprints.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {self.output_path}/visualizations/")
    
    def generate_report(self):
        """Generate comprehensive PRNU extraction report"""
        logger.info("Generating PRNU extraction report...")
        
        total_scanners = len(self.prnu_fingerprints)
        total_fingerprints = sum(len(dpi_fingerprints) for dpi_fingerprints in self.prnu_fingerprints.values())
        
        print("\n" + "="*60)
        print("PRNU FINGERPRINT EXTRACTION REPORT")
        print("="*60)
        
        print(f"\nExtraction Summary:")
        print(f"Total scanners processed: {total_scanners}")
        print(f"Total fingerprints created: {total_fingerprints}")
        print(f"Failed images: {len(self.failed_images)}")
        
        print(f"\nPer-Scanner Fingerprint Status:")
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            dpis = list(dpi_fingerprints.keys())
            print(f"  {scanner_id}: {dpis}")
        
        print(f"\nFingerprint Statistics:")
        for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
            for dpi, fingerprint in dpi_fingerprints.items():
                print(f"  {scanner_id} {dpi} DPI:")
                print(f"    Shape: {fingerprint.shape}")
                print(f"    Mean: {np.mean(fingerprint):.6f}")
                print(f"    Std: {np.std(fingerprint):.6f}")
                print(f"    Min: {np.min(fingerprint):.6f}")
                print(f"    Max: {np.max(fingerprint):.6f}")
        
        # Save report
        report_path = os.path.join(self.output_path, "metadata", "prnu_extraction_report.txt")
        with open(report_path, 'w') as f:
            f.write("PRNU FINGERPRINT EXTRACTION REPORT\n")
            f.write("="*60 + "\n\n")
            f.write(f"Total scanners processed: {total_scanners}\n")
            f.write(f"Total fingerprints created: {total_fingerprints}\n")
            f.write(f"Failed images: {len(self.failed_images)}\n\n")
            
            for scanner_id, dpi_fingerprints in self.prnu_fingerprints.items():
                f.write(f"{scanner_id}:\n")
                for dpi, fingerprint in dpi_fingerprints.items():
                    f.write(f"  {dpi} DPI - Shape: {fingerprint.shape}, Mean: {np.mean(fingerprint):.6f}, Std: {np.std(fingerprint):.6f}\n")
                f.write("\n")
        
        logger.info(f"Report saved to {report_path}")
    
    def run_full_pipeline(self):
        """Run the complete PRNU fingerprint extraction pipeline"""
        logger.info("Starting PRNU fingerprint extraction pipeline...")
        
        # Task 1: Load flatfield images
        self.load_flatfield_images()
        
        # Task 2 & 3: Extract noise residuals and create fingerprints
        self.create_prnu_fingerprints()
        
        # Task 4: Save fingerprints
        self.save_fingerprints()
        
        # Task 5: Visualize fingerprints
        self.visualize_fingerprints()
        
        # Generate report
        self.generate_report()
        
        logger.info("PRNU fingerprint extraction pipeline complete!")

if __name__ == "__main__":
    extractor = PRNUFingerprintExtractor()
    extractor.run_full_pipeline()
