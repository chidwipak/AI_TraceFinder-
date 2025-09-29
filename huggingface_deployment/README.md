---
title: Tampered Image Detection System
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Advanced forensic image analysis using machine learning to detect tampered images
---

<!-- This Space uses requirements.txt for dependencies. Make sure to keep it updated. -->

# 🔍 Tampered Image Detection System

Advanced forensic image analysis using machine learning to detect tampered images with 83.3% overall accuracy.

## Features

- **High Accuracy**: 83.3% overall accuracy with 100% tampered detection
- **Robust Ensemble**: Combines 5 machine learning algorithms
- **30 Forensic Features**: Optimized feature selection
- **Real-time Analysis**: Instant image analysis
- **Multiple Tampering Types**: Detects copy-move, retouching, and splicing

## Model Performance

- **Overall Accuracy**: 83.3%
- **Original Images**: 66.7% accuracy
- **Tampered Images**: 100% accuracy
- **Training Data**: 136 images from SUPATLANTIQUE dataset

## How It Works

1. **Feature Extraction**: Extracts 105 forensic features including:
   - Error Level Analysis (ELA)
   - Noise analysis
   - Compression artifacts
   - Frequency domain features
   - Texture analysis
   - Statistical features

2. **Preprocessing**: Applies robust scaling and selects 30 most relevant features

3. **Prediction**: Uses ensemble of 5 algorithms:
   - Random Forest
   - LightGBM
   - XGBoost
   - SVM (RBF)
   - Logistic Regression

## Usage

1. Upload an image using the file uploader
2. Click "Analyze Image"
3. View the prediction result and confidence scores
4. Check the detailed analysis

## Technical Details

- **Framework**: Gradio
- **Model**: Robust Ensemble with SMOTE+ENN preprocessing
- **Features**: 30 selected forensic features
- **Dataset**: SUPATLANTIQUE professional tampering dataset
- **Image Types**: JPG, JPEG, PNG, TIFF

## Deployment Notes

- **Requirements**: The `requirements.txt` file has been updated to use `PyWavelets` instead of `pywt` to fix dependency issues
- **Compatibility**: This app is optimized for Hugging Face Spaces deployment
- **Alternative**: A more comprehensive `requirements_gradio.txt` file is also available in the project root for other deployment platforms

## Limitations

- Model accuracy is limited by small training dataset
- May not detect very sophisticated tampering
- Results should be used as a preliminary analysis tool

## Dataset

- **SUPATLANTIQUE Dataset**: Professional tampering dataset
- **High Resolution**: 2480x3500 pixel images
- **Multiple Tampering Types**: Copy-move, retouching, splicing
