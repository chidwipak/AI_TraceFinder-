#!/bin/bash

# CPU-Optimized CNN + MLP Pipeline for 97%+ Accuracy
# This script runs the advanced deep learning pipeline using existing preprocessed data

echo "🚀 Starting CPU-Optimized CNN + MLP Pipeline for 97%+ Accuracy"
echo "=============================================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if preprocessed data exists
if [ ! -d "preprocessed" ]; then
    echo "❌ Preprocessed data not found. Please run preprocessing first."
    echo "   Run: python preprocessing_scanner_identification.py"
    exit 1
fi

# Check if features exist
if [ ! -d "features" ]; then
    echo "❌ Features not found. Please run feature extraction first."
    echo "   Run: python feature_extraction.py"
    exit 1
fi

# Check if PRNU fingerprints exist
if [ ! -d "prnu_fingerprints" ]; then
    echo "❌ PRNU fingerprints not found. Please run PRNU extraction first."
    echo "   Run: python prnu_fingerprint_extraction.py"
    exit 1
fi

# Run the CPU-optimized pipeline
echo "🎯 Running CPU-optimized CNN + MLP pipeline..."
python3 cpu_optimized_cnn_mlp_97_plus.py

# Check if the pipeline completed successfully
if [ $? -eq 0 ]; then
    echo "✅ Pipeline completed successfully!"
    echo "📊 Results saved in: cpu_optimized_cnn_mlp_97_results/"
    echo "🏆 Check the results to see if 97%+ accuracy was achieved!"
else
    echo "❌ Pipeline failed. Check the logs for details."
    exit 1
fi

echo "🎉 CPU-optimized CNN + MLP pipeline finished!"
