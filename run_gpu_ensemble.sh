#!/bin/bash

echo "🚀 LAUNCHING GPU-OPTIMIZED ENSEMBLE PIPELINE"
echo "=============================================="

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits

# Check CUDA version
echo "🔍 Checking CUDA version..."
python -c "import torch; print(f'PyTorch CUDA: {torch.version.cuda}')"
python -c "import cupy as cp; print(f'CuPy CUDA: {cp.cuda.runtime.driverGetVersion()}')"

# Run GPU ensemble
echo "🎯 Starting GPU ensemble pipeline..."
python ensemble_95_plus_gpu.py

echo "✅ GPU ensemble pipeline completed!"
