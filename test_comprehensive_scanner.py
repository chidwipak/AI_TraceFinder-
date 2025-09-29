#!/usr/bin/env python3
"""
Comprehensive Scanner Identification Test
Tests with more images from all scanners to verify accuracy aligns with 93.75%
"""

import os
import sys
import numpy as np
import pandas as pd
import random
import warnings
warnings.filterwarnings('ignore')

# Import the fixed scanner identifier
from scanner_identification_fixed import FixedScannerIdentifier

class ComprehensiveScannerTester:
    """Comprehensive test for scanner identification"""
    
    def __init__(self):
        self.identifier = FixedScannerIdentifier()
        self.results = []
        
        # Dataset paths
        self.dataset_path = "The SUPATLANTIQUE dataset"
        self.flatfield_path = os.path.join(self.dataset_path, "Flatfield")
        self.official_path = os.path.join(self.dataset_path, "Official")
        self.wikipedia_path = os.path.join(self.dataset_path, "Wikipedia")
    
    def get_comprehensive_scanner_images(self, num_per_scanner=10):
        """Get comprehensive sample of images from all scanners"""
        all_scanners = []
        
        # Collect unique scanner names
        for source_path in [self.flatfield_path, self.official_path, self.wikipedia_path]:
            if os.path.exists(source_path):
                for scanner_dir in os.listdir(source_path):
                    if scanner_dir not in all_scanners and scanner_dir != '.DS_Store':
                        all_scanners.append(scanner_dir)
        
        print(f"Found {len(all_scanners)} scanner types: {all_scanners}")
        
        scanner_images = []
        
        for scanner_dir in all_scanners:
            scanner_images_found = []
            
            # Collect from all sources
            for source_path in [self.flatfield_path, self.official_path, self.wikipedia_path]:
                if os.path.exists(source_path):
                    scanner_path = os.path.join(source_path, scanner_dir)
                    if os.path.isdir(scanner_path):
                        for dpi_dir in os.listdir(scanner_path):
                            if dpi_dir in ['150', '300']:
                                dpi_path = os.path.join(scanner_path, dpi_dir)
                                if os.path.isdir(dpi_path):
                                    for img_file in os.listdir(dpi_path):
                                        if img_file.lower().endswith(('.tif', '.tiff', '.png', '.jpg')):
                                            img_path = os.path.join(dpi_path, img_file)
                                            scanner_images_found.append((img_path, scanner_dir, dpi_dir))
            
            # Sample from this scanner
            if scanner_images_found:
                random.shuffle(scanner_images_found)
                selected = scanner_images_found[:num_per_scanner]
                scanner_images.extend(selected)
                print(f"Selected {len(selected)} images from {scanner_dir}")
        
        # Shuffle the final list
        random.shuffle(scanner_images)
        
        return scanner_images
    
    def test_scanner_identification(self, num_per_scanner=10):
        """Test scanner identification with comprehensive sampling"""
        print(f"\n🔍 Comprehensive Scanner Identification Test")
        print("=" * 60)
        
        # Load model
        if not self.identifier.load_model():
            print("❌ Failed to load model")
            return 0.0
        
        # Get comprehensive sample of images
        scanner_images = self.get_comprehensive_scanner_images(num_per_scanner)
        
        if not scanner_images:
            print("❌ No scanner images found")
            return 0.0
        
        print(f"Testing with {len(scanner_images)} images")
        
        correct_predictions = 0
        total_predictions = 0
        scanner_results = {}
        
        for i, (img_path, expected_scanner, dpi) in enumerate(scanner_images):
            try:
                print(f"Testing {i+1}/{len(scanner_images)}: {os.path.basename(img_path)}")
                
                # Load and preprocess image
                image = self.identifier.robust_image_reading(img_path)
                if image is None:
                    continue
                
                processed_image = self.identifier.preprocess_image(image)
                if processed_image is None:
                    continue
                
                # Predict
                predicted_scanner, confidence, status = self.identifier.predict_scanner(processed_image)
                
                # Check if correct
                is_correct = predicted_scanner == expected_scanner
                if is_correct:
                    correct_predictions += 1
                
                total_predictions += 1
                
                # Store results by scanner
                if expected_scanner not in scanner_results:
                    scanner_results[expected_scanner] = {'correct': 0, 'total': 0}
                scanner_results[expected_scanner]['total'] += 1
                if is_correct:
                    scanner_results[expected_scanner]['correct'] += 1
                
                # Store detailed results
                self.results.append({
                    'image_path': img_path,
                    'expected_scanner': expected_scanner,
                    'predicted_scanner': predicted_scanner,
                    'confidence': confidence,
                    'is_correct': is_correct,
                    'dpi': dpi,
                    'status': status
                })
                
                print(f"  Expected: {expected_scanner}")
                print(f"  Predicted: {predicted_scanner}")
                print(f"  Confidence: {confidence:.3f}")
                print(f"  Correct: {is_correct}")
                print()
                
            except Exception as e:
                print(f"  ❌ Error processing {img_path}: {e}")
                continue
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        print(f"\n📊 Comprehensive Scanner Identification Results:")
        print(f"   Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.1%}")
        print(f"   Expected Accuracy: 93.75%")
        print(f"   Difference: {abs(accuracy - 0.9375):.1%}")
        
        print(f"\n📊 Per-Scanner Results:")
        for scanner, stats in scanner_results.items():
            scanner_accuracy = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {scanner}: {stats['correct']}/{stats['total']} = {scanner_accuracy:.1%}")
        
        # Save detailed results
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv("comprehensive_scanner_test_results.csv", index=False)
            print(f"\n💾 Detailed results saved to comprehensive_scanner_test_results.csv")
        
        return accuracy
    
    def run_comprehensive_test(self):
        """Run comprehensive test with multiple sample sizes"""
        print("🧪 Comprehensive Scanner Identification Test")
        print("=" * 60)
        
        # Test with different sample sizes
        sample_sizes = [5, 10, 15]
        
        for num_per_scanner in sample_sizes:
            print(f"\n🔍 Testing with {num_per_scanner} images per scanner")
            print("-" * 40)
            
            accuracy = self.test_scanner_identification(num_per_scanner)
            
            if accuracy >= 0.85:  # 85% threshold
                print(f"✅ Test PASSED with {num_per_scanner} images per scanner!")
                print(f"   Accuracy: {accuracy:.1%} (Expected: 93.75%)")
            else:
                print(f"❌ Test FAILED with {num_per_scanner} images per scanner!")
                print(f"   Accuracy: {accuracy:.1%} (Expected: 93.75%)")
            
            # Reset results for next test
            self.results = []
        
        return True

def main():
    """Main test function"""
    tester = ComprehensiveScannerTester()
    success = tester.run_comprehensive_test()
    
    if success:
        print("\n🚀 Comprehensive scanner identification testing completed!")
        print("📱 Streamlit app is running at: http://localhost:8503")
    else:
        print("\n🔧 Comprehensive testing failed!")

if __name__ == "__main__":
    main()
