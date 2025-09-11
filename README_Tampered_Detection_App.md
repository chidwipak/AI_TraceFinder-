# Tampered Image Detection Streamlit App

## Overview
This Streamlit application provides a web interface for tampered image detection using the best performing model from Objective 2. The app uses LightGBM with SMOTE+ENN preprocessing, achieving 75% accuracy on the SUPATLANTIQUE dataset.

## Features
- **Web Interface**: User-friendly Streamlit web application
- **Image Upload**: Support for JPG, JPEG, PNG, TIFF formats
- **Real-time Analysis**: Instant tampered/original detection
- **Confidence Scores**: Probability scores for predictions
- **Feature Analysis**: Display of extracted forensic features
- **Sample Testing**: Test with pre-loaded sample images
- **Model Information**: Detailed model performance metrics

## Quick Start

### Option 1: Simple Version (Recommended)
```bash
# Run the simplified version
./run_simple_tampered_app.sh
```

### Option 2: Full Version (Requires model files)
```bash
# Run the full version with trained models
./run_tampered_detection_app.sh
```

## Requirements
- Python 3.8+
- Virtual environment with required packages
- Streamlit
- OpenCV, PIL, scikit-learn, matplotlib, pandas

## Installation
1. Ensure virtual environment is activated:
   ```bash
   source venv/bin/activate
   ```

2. Install additional requirements:
   ```bash
   pip install streamlit plotly
   ```

3. Run the application:
   ```bash
   ./run_simple_tampered_app.sh
   ```

## Usage

### Main Detection Page
1. Upload an image file (JPG, PNG, TIFF)
2. View the analysis results
3. Check confidence scores
4. Review extracted features

### Model Information Page
- View model performance metrics
- See feature categories and counts
- Understand model limitations

### Sample Testing Page
- Test with pre-loaded sample images
- Compare expected vs predicted results
- Validate model performance

### About Page
- Learn about the system
- Understand limitations
- View technical details

## Model Information

### Best Performing Model
- **Algorithm**: LightGBM with SMOTE+ENN
- **Accuracy**: 75% on test set
- **Features**: 36 selected forensic features
- **Training Data**: 136 images (102 tampered, 34 original)

### Feature Categories
1. **Statistical Features** (15): Mean, std, median, percentiles
2. **Edge Features** (5): Edge density, gradient magnitude
3. **Texture Features** (8): Gabor filters, local binary patterns
4. **Color Features** (6): HSV color space analysis
5. **Noise Analysis** (5): Gaussian noise estimation
6. **Compression Artifacts** (8): DCT coefficients, blocking artifacts

### Performance Metrics
- **Accuracy**: 75.0%
- **Precision**: 75.0%
- **Recall**: 100.0%
- **F1-Score**: 85.7%

## Dataset Information
- **SUPATLANTIQUE Dataset**: Professional tampering dataset
- **Total Images**: 136 (102 tampered, 34 original)
- **Tampering Types**: Copy-move, Retouching, Splicing
- **Image Format**: High-resolution TIFF files (2480x3500 pixels)
- **Class Imbalance**: 3:1 ratio (tampered:original)

## Limitations
1. **Dataset Size**: Limited to 136 training images
2. **Accuracy**: 75% accuracy may not be sufficient for critical applications
3. **Sophisticated Tampering**: May not detect very advanced tampering
4. **Generalization**: Trained on specific dataset, may not generalize well

## Technical Details

### Architecture
- **Frontend**: Streamlit web interface
- **Backend**: Python with scikit-learn
- **Feature Extraction**: Custom forensic analysis
- **Model**: LightGBM with class weighting

### File Structure
```
├── streamlit_tampered_detection_simple.py    # Simplified app
├── streamlit_tampered_detection_app.py       # Full app
├── run_simple_tampered_app.sh               # Simple launcher
├── run_tampered_detection_app.sh            # Full launcher
├── requirements_streamlit_tampered.txt      # Requirements
└── README_Tampered_Detection_App.md         # This file
```

### Dependencies
- streamlit==1.25.0
- opencv-python==4.8.1.78
- Pillow==10.0.1
- scikit-learn==1.3.0
- lightgbm==4.0.0
- matplotlib==3.7.2
- seaborn==0.12.2
- pandas==2.0.3
- numpy==1.24.3

## Troubleshooting

### Common Issues
1. **Model not found**: Use the simple version which doesn't require model files
2. **Import errors**: Ensure all dependencies are installed
3. **Port conflicts**: Change port in launcher script if 8501 is occupied
4. **Memory issues**: Reduce image size or use smaller images

### Error Messages
- **"Model not loaded"**: Use simple version or ensure model files exist
- **"Feature extraction failed"**: Check image format and file integrity
- **"Streamlit not found"**: Install streamlit: `pip install streamlit`

## Future Improvements
1. **More Training Data**: Collect larger, more diverse dataset
2. **Better Features**: Implement domain-specific forensic features
3. **Transfer Learning**: Use pre-trained models on larger datasets
4. **Real-time Processing**: Optimize for faster inference
5. **Batch Processing**: Support multiple image analysis

## Contact
This application is part of the AI TraceFinder project, Objective 2: Tampered Image Detection.

## License
Part of the AI TraceFinder research project.
