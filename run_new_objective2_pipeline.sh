#!/bin/bash

# New Objective 2: Comprehensive Tampered/Original Detection Pipeline Runner
# This script runs the complete pipeline to achieve 90%+ accuracy

echo "🚀 STARTING NEW OBJECTIVE 2 COMPREHENSIVE PIPELINE"
echo "=================================================="
echo "Pipeline Goal: Achieve 90%+ accuracy for tampered/original detection"
echo "Using ALL available data sources:"
echo "  - Originals/ folder (PDFs converted to images)"
echo "  - Tampered images/ folder (all subfolders)"
echo "  - Binary masks/ for region-specific analysis"
echo "  - Description/ for metadata analysis"
echo "=================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Creating one..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade requirements
echo "📦 Installing/upgrading requirements..."
pip install --upgrade pip
pip install -r requirements_new_objective2.txt

# Check if all required packages are installed
echo "🔍 Checking dependencies..."
python3 -c "
import sys
required_packages = ['pandas', 'numpy', 'cv2', 'sklearn', 'matplotlib', 'seaborn', 'scipy', 'skimage', 'pywt', 'mahotas', 'imblearn', 'albumentations', 'pdf2image', 'fitz']
missing = []
for package in required_packages:
    try:
        if package == 'cv2':
            import cv2
        elif package == 'sklearn':
            import sklearn
        elif package == 'skimage':
            import skimage
        elif package == 'pywt':
            import pywt
        elif package == 'imblearn':
            import imblearn
        elif package == 'fitz':
            import fitz
        else:
            __import__(package)
        print(f'✅ {package}')
    except ImportError:
        missing.append(package)
        print(f'❌ {package}')

if missing:
    print(f'❌ Missing packages: {missing}')
    print('Please install missing packages manually')
    sys.exit(1)
else:
    print('✅ All dependencies are available!')
"

if [ $? -ne 0 ]; then
    echo "❌ Dependency check failed. Please install missing packages."
    exit 1
fi

# Create log directory
mkdir -p logs

# Run the complete pipeline
echo "🚀 Running complete Objective 2 pipeline..."
python3 run_new_objective2_pipeline.py 2>&1 | tee logs/new_objective2_pipeline_$(date +%Y%m%d_%H%M%S).log

# Check pipeline result
if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 NEW OBJECTIVE 2 PIPELINE COMPLETED SUCCESSFULLY!"
    echo "=================================================="
    echo "✅ Data Analysis: PDFs converted to images, all data sources analyzed"
    echo "✅ Data Preprocessing: Comprehensive train/test splits with class balancing"
    echo "✅ Feature Extraction: Comprehensive forensic features extracted using all data"
    echo ""
    echo "📊 Results saved in:"
    echo "  - new_objective2_analysis/"
    echo "  - new_objective2_preprocessed/"
    echo "  - new_objective2_features/"
    echo ""
    echo "🎯 Next Steps:"
    echo "1. Run baseline ML models training"
    echo "2. Run deep learning models training"
    echo "3. Create ensemble models"
    echo "4. Evaluate final performance"
    echo "5. Analyze results and identify improvements for 90%+ accuracy"
    echo ""
    echo "📈 Expected improvements over previous 75% accuracy:"
    echo "  - More original training data (PDFs converted to images)"
    echo "  - Better class balance (more original samples)"
    echo "  - Mask-aware features for tampered regions"
    echo "  - Comprehensive forensic feature extraction"
    echo "  - Advanced preprocessing and augmentation"
else
    echo ""
    echo "❌ NEW OBJECTIVE 2 PIPELINE FAILED!"
    echo "====================================="
    echo "Please check the logs for detailed error information:"
    echo "  - logs/new_objective2_pipeline_*.log"
    echo "  - Individual script outputs"
    echo ""
    echo "🔧 Common issues and solutions:"
    echo "1. Missing dependencies: Install required packages"
    echo "2. PDF conversion issues: Check pdf2image and PyMuPDF installation"
    echo "3. Memory issues: Reduce batch size or use smaller images"
    echo "4. File permission issues: Check write permissions for output directories"
fi

echo ""
echo "Pipeline execution completed at: $(date)"
