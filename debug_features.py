#!/usr/bin/env python3
"""
Debug feature extraction mismatch
"""

import sys
sys.path.append('.')
from combined_feature_extractors import CombinedFeatureExtractor
import cv2
import numpy as np
import pandas as pd
import joblib

def debug_feature_extraction():
    # Test feature extraction on a known image from the test set
    test_df = pd.read_csv('new_objective2_preprocessed/metadata/test_split.csv')
    features_df = pd.read_csv('new_objective2_features/feature_vectors/comprehensive_features.csv')
    
    # Load model and scaler
    model = joblib.load('new_objective2_baseline_ml_results/models/best_model.pkl')
    scaler = joblib.load('new_objective2_baseline_ml_results/models/scaler.pkl')
    
    # Get the first few test images
    for i in range(min(3, len(test_df))):
        test_image_path = test_df.iloc[i]['image_path']
        expected_label = test_df.iloc[i]['label']
        
        print(f'\n=== Testing image {i+1}: {test_image_path} ===')
        print(f'Expected label: {expected_label} ({"Tampered" if expected_label == 1 else "Original"})')
        
        try:
            # Load the image
            if test_image_path.lower().endswith(('.tif', '.tiff')):
                import tifffile
                image = tifffile.imread(test_image_path)
            else:
                image = cv2.imread(test_image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if image is None:
                print('❌ Failed to load image')
                continue
                
            print(f'✅ Image loaded: {image.shape}, dtype: {image.dtype}')
            
            # Extract features using my extractor
            extractor = CombinedFeatureExtractor()
            features = extractor.extract_tampered_features(image)
            print(f'Extracted features shape: {features.shape}')
            
            # Scale and predict
            features_scaled = scaler.transform(features.reshape(1, -1))
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0]
            
            print(f'Prediction: {pred} ({"Tampered" if pred == 1 else "Original"})')
            print(f'Probabilities: {proba}')
            print(f'Max probability: {np.max(proba):.6f}')
            print(f'Correct: {pred == expected_label}')
            
            # Compare with stored features
            stored_features_row = features_df[features_df['image_path'] == test_image_path]
            if len(stored_features_row) > 0:
                feature_columns = [col for col in features_df.columns 
                                  if col not in ['image_path', 'label', 'tamper_type', 'category', 'has_mask']]
                stored_features = stored_features_row[feature_columns].values[0]
                
                # Test stored features
                stored_scaled = scaler.transform(stored_features.reshape(1, -1))
                stored_pred = model.predict(stored_scaled)[0]
                stored_proba = model.predict_proba(stored_scaled)[0]
                
                print(f'Stored prediction: {stored_pred} ({"Tampered" if stored_pred == 1 else "Original"})')
                print(f'Stored probabilities: {stored_proba}')
                
                # Check feature differences
                diff = np.abs(features - stored_features)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)
                print(f'Feature differences - Max: {max_diff:.6f}, Mean: {mean_diff:.6f}')
                
                if max_diff < 1e-3:
                    print('✅ Features match closely!')
                else:
                    print('❌ Features do not match!')
                    # Find which features differ most
                    diff_indices = np.argsort(diff)[-5:]  # Top 5 differences
                    print('Top 5 feature differences:')
                    for idx in diff_indices:
                        print(f'  Feature {idx}: extracted={features[idx]:.6f}, stored={stored_features[idx]:.6f}, diff={diff[idx]:.6f}')
            else:
                print('❌ Image not found in stored features')
                
        except Exception as e:
            print(f'❌ Error: {e}')
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    debug_feature_extraction()

