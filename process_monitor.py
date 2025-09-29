#!/usr/bin/env python3
"""
Process Monitor - Continuously monitors for stuck processes and handles them automatically
"""

import os
import time
import signal
import subprocess
import psutil
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('process_monitor.log'),
        logging.StreamHandler()
    ]
)

class ProcessMonitor:
    def __init__(self):
        self.stuck_threshold = 300  # 5 minutes
        self.check_interval = 30    # Check every 30 seconds
        self.max_cpu_usage = 95.0   # Max CPU usage before considering stuck
        self.max_memory_usage = 80.0  # Max memory usage before considering stuck
        
    def get_python_processes(self):
        """Get all Python processes"""
        python_processes = []
        for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
            try:
                if proc.info['name'] and 'python' in proc.info['name'].lower():
                    python_processes.append(proc)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return python_processes
    
    def is_process_stuck(self, proc):
        """Check if a process is stuck based on various criteria"""
        try:
            # Get process info
            cpu_percent = proc.cpu_percent()
            memory_percent = proc.memory_percent()
            create_time = proc.create_time()
            current_time = time.time()
            runtime = current_time - create_time
            
            # Check if process has been running too long with high resource usage
            if runtime > self.stuck_threshold:
                if cpu_percent > self.max_cpu_usage or memory_percent > self.max_memory_usage:
                    return True, f"High resource usage: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%, Runtime={runtime:.1f}s"
            
            # Check if process is consuming excessive resources
            if cpu_percent > 99.0 or memory_percent > 90.0:
                return True, f"Excessive resource usage: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%"
            
            # Check if process has been running for a very long time (1 hour)
            if runtime > 3600:
                return True, f"Process running too long: {runtime:.1f}s"
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False, "Process no longer exists"
        
        return False, "Process appears healthy"
    
    def handle_stuck_process(self, proc, reason):
        """Handle a stuck process"""
        try:
            pid = proc.pid
            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else 'Unknown'
            
            logging.warning(f"Stuck process detected - PID: {pid}, Reason: {reason}")
            logging.warning(f"Command: {cmdline}")
            
            # Try graceful termination first
            try:
                proc.terminate()
                logging.info(f"Sent SIGTERM to process {pid}")
                
                # Wait for graceful termination
                try:
                    proc.wait(timeout=10)
                    logging.info(f"Process {pid} terminated gracefully")
                    return True
                except psutil.TimeoutExpired:
                    logging.warning(f"Process {pid} did not terminate gracefully, forcing kill")
                    proc.kill()
                    logging.info(f"Process {pid} force killed")
                    return True
                    
            except psutil.NoSuchProcess:
                logging.info(f"Process {pid} already terminated")
                return True
            except psutil.AccessDenied:
                logging.error(f"Access denied when trying to terminate process {pid}")
                return False
                
        except Exception as e:
            logging.error(f"Error handling stuck process: {e}")
            return False
    
    def monitor_processes(self):
        """Main monitoring loop"""
        logging.info("Starting process monitor...")
        logging.info(f"Monitoring parameters: CPU threshold={self.max_cpu_usage}%, Memory threshold={self.max_memory_usage}%, Stuck threshold={self.stuck_threshold}s")
        
        while True:
            try:
                python_processes = self.get_python_processes()
                
                if not python_processes:
                    logging.info("No Python processes found")
                else:
                    logging.info(f"Monitoring {len(python_processes)} Python processes")
                    
                    for proc in python_processes:
                        try:
                            is_stuck, reason = self.is_process_stuck(proc)
                            
                            if is_stuck:
                                success = self.handle_stuck_process(proc, reason)
                                if success:
                                    logging.info(f"Successfully handled stuck process {proc.pid}")
                                else:
                                    logging.error(f"Failed to handle stuck process {proc.pid}")
                            else:
                                # Log process status periodically
                                cpu_percent = proc.cpu_percent()
                                memory_percent = proc.memory_percent()
                                logging.debug(f"Process {proc.pid}: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}% - {reason}")
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                        except Exception as e:
                            logging.error(f"Error monitoring process {proc.pid}: {e}")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logging.info("Process monitor stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)

def main():
    """Main function"""
    print("Process Monitor - Monitoring for stuck Python processes")
    print("Press Ctrl+C to stop monitoring")
    print("Logs will be saved to process_monitor.log")
    print("-" * 50)
    
    monitor = ProcessMonitor()
    monitor.monitor_processes()

if __name__ == "__main__":
    main()

