#!/usr/bin/env python3
"""
Test the fixed ensemble code to ensure it runs without hanging
"""

import time
import subprocess

def test_ensemble():
    print("🧪 Testing fixed ensemble code...")
    
    # Start the ensemble process
    print("🚀 Starting ensemble process...")
    start_time = time.time()
    
    try:
        # Run with timeout of 5 minutes
        result = subprocess.run(
            ["python", "ensemble_95_plus.py"],
            timeout=300,  # 5 minutes
            capture_output=True,
            text=True
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ Ensemble completed successfully in {elapsed:.1f} seconds!")
            print("📊 Output preview:")
            print(result.stdout[-500:])  # Last 500 characters
        else:
            print(f"❌ Ensemble failed after {elapsed:.1f} seconds")
            print("Error output:")
            print(result.stderr)
            
    except subprocess.TimeoutExpired:
        print("⏰ Ensemble timed out after 5 minutes - still has issues")
    except Exception as e:
        print(f"❌ Error running ensemble: {e}")

if __name__ == "__main__":
    test_ensemble()
