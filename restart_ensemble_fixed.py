#!/usr/bin/env python3
"""
Restart Ensemble with Fixed Deadlock Issue
"""

import os
import subprocess
import time

def main():
    print("🔄 Restarting Ensemble Process with Deadlock Fix...")
    
    # Kill existing stuck processes
    print("🛑 Stopping stuck ensemble processes...")
    try:
        subprocess.run(["pkill", "-f", "ensemble_95_plus_gpu_k80.py"], check=False)
        time.sleep(5)  # Wait for processes to stop
        print("✅ Stopped existing processes")
    except Exception as e:
        print(f"⚠️ Warning: {e}")
    
    # Start the fixed ensemble process
    print("🚀 Starting fixed ensemble process...")
    try:
        # Start in background with nohup
        cmd = "nohup python3 ensemble_95_plus_gpu_k80.py > ensemble_output_fixed.log 2>&1 &"
        subprocess.run(cmd, shell=True, check=True)
        print("✅ Ensemble process started successfully!")
        print("📝 Log file: ensemble_output_fixed.log")
        print("🔍 Monitor with: tail -f ensemble_output_fixed.log")
    except Exception as e:
        print(f"❌ Failed to start ensemble: {e}")

if __name__ == "__main__":
    main()
