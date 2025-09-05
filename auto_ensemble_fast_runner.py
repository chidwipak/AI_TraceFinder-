#!/usr/bin/env python3
"""
Auto-Restart Fast Ensemble Runner
Optimized for the fast ensemble version that won't timeout
"""

import os
import sys
import time
import signal
import subprocess
import threading
from datetime import datetime

class AutoFastEnsembleRunner:
    def __init__(self):
        self.process = None
        self.is_running = False
        self.restart_count = 0
        self.max_restarts = 5
        self.last_output_time = time.time()
        self.output_timeout = 600  # 10 minutes without output = stuck
        self.script_name = "ensemble_95_plus_fast.py"
        self.log_file = "auto_ensemble_fast_runner.log"
        
    def log(self, message):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        
        # Also write to log file
        with open(self.log_file, "a") as f:
            f.write(log_message + "\n")
    
    def check_if_stuck(self):
        """Check if the process is stuck (no output for too long)"""
        while self.is_running and self.process:
            time.sleep(60)  # Check every minute
            
            if not self.is_running:
                break
                
            current_time = time.time()
            if current_time - self.last_output_time > self.output_timeout:
                self.log(f"⚠️ Process appears stuck (no output for {self.output_timeout//60} minutes)")
                self.log("🔄 Restarting stuck process...")
                self.restart_process()
                break
    
    def monitor_output(self, process):
        """Monitor process output and detect if it's stuck"""
        try:
            for line in iter(process.stdout.readline, ''):
                if line:
                    output = line.strip()
                    print(output)
                    self.last_output_time = time.time()
                    
                    # Check for completion indicators
                    if "✅ All ensembles completed successfully!" in output:
                        self.log("🎉 Fast ensemble process completed successfully!")
                        self.is_running = False
                        return
                    elif "❌" in output and "FAILED" in output:
                        self.log("❌ Fast ensemble process failed, will restart...")
                        self.is_running = False
                        return
                        
        except Exception as e:
            self.log(f"⚠️ Error monitoring output: {e}")
    
    def start_process(self):
        """Start the fast ensemble process"""
        try:
            self.log(f"🚀 Starting FAST ensemble process (attempt {self.restart_count + 1}/{self.max_restarts})")
            self.log("⚡ This version is optimized for speed and won't timeout!")
            
            # Start the process
            self.process = subprocess.Popen(
                ["python", self.script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                text=True
            )
            
            self.is_running = True
            self.last_output_time = time.time()
            
            # Start monitoring threads
            output_thread = threading.Thread(target=self.monitor_output, args=(self.process,))
            stuck_check_thread = threading.Thread(target=self.check_if_stuck)
            
            output_thread.daemon = True
            stuck_check_thread.daemon = True
            
            output_thread.start()
            stuck_check_thread.start()
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            if return_code == 0:
                self.log("✅ Fast ensemble process completed successfully!")
                self.is_running = False
                return True
            else:
                self.log(f"❌ Fast ensemble process failed with return code {return_code}")
                self.is_running = False
                return False
                
        except Exception as e:
            self.log(f"❌ Error starting process: {e}")
            self.is_running = False
            return False
    
    def restart_process(self):
        """Restart the fast ensemble process"""
        if self.process:
            try:
                self.log("🛑 Stopping current process...")
                self.process.terminate()
                time.sleep(5)
                
                if self.process.poll() is None:
                    self.log("🔴 Force killing process...")
                    self.process.kill()
                    time.sleep(2)
                    
            except Exception as e:
                self.log(f"⚠️ Error stopping process: {e}")
        
        self.is_running = False
        self.process = None
    
    def run_with_auto_restart(self):
        """Run fast ensemble with automatic restart on failure"""
        self.log("🎯 Starting Auto-Fast Ensemble Runner")
        self.log("⚡ Optimized for speed - should complete without timeouts!")
        self.log(f"📝 Log file: {self.log_file}")
        
        while self.restart_count < self.max_restarts:
            try:
                if self.start_process():
                    self.log("🎉 Fast ensemble completed successfully!")
                    break
                else:
                    self.restart_count += 1
                    if self.restart_count < self.max_restarts:
                        self.log(f"🔄 Restarting in 10 seconds... (attempt {self.restart_count + 1}/{self.max_restarts})")
                        time.sleep(10)
                    else:
                        self.log("❌ Maximum restart attempts reached. Giving up.")
                        break
                        
            except KeyboardInterrupt:
                self.log("⚠️ User interrupted. Stopping...")
                break
            except Exception as e:
                self.log(f"❌ Unexpected error: {e}")
                self.restart_count += 1
                if self.restart_count < self.max_restarts:
                    self.log(f"🔄 Restarting in 10 seconds... (attempt {self.restart_count + 1}/{self.max_restarts})")
                    time.sleep(10)
                else:
                    self.log("❌ Maximum restart attempts reached. Giving up.")
                    break
        
        self.log("🏁 Auto-Fast Ensemble Runner finished")
        
        if self.restart_count >= self.max_restarts:
            self.log("❌ Failed to complete fast ensemble after maximum restarts")
            return False
        else:
            self.log("✅ Fast ensemble completed successfully!")
            return True

def main():
    """Main function"""
    runner = AutoFastEnsembleRunner()
    
    # Handle Ctrl+C gracefully
    def signal_handler(signum, frame):
        print("\n⚠️ Received interrupt signal. Stopping gracefully...")
        runner.is_running = False
        if runner.process:
            runner.process.terminate()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the auto-restart system
    success = runner.run_with_auto_restart()
    
    if success:
        print("\n🎉 Fast ensemble training completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Fast ensemble training failed after maximum restarts")
        sys.exit(1)

if __name__ == "__main__":
    main()
