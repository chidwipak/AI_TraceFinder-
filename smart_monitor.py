#!/usr/bin/env python3
"""
Smart Process Monitor - Continuously monitors and automatically restarts stuck processes
"""

import os
import time
import signal
import subprocess
import psutil
import threading
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('smart_monitor.log'),
        logging.StreamHandler()
    ]
)

class SmartMonitor:
    def __init__(self):
        self.check_interval = 10  # Check every 10 seconds
        self.cpu_threshold = 80.0  # CPU threshold for stuck detection
        self.memory_threshold = 70.0  # Memory threshold
        self.runtime_threshold = 300  # 5 minutes runtime threshold
        self.max_restart_attempts = 3
        self.restart_cooldown = 60  # 60 seconds between restart attempts
        self.process_restart_count = {}
        self.last_restart_time = {}
        self.running = True
        
    def is_process_stuck(self, proc):
        """Enhanced stuck process detection"""
        try:
            cpu_percent = proc.cpu_percent()
            memory_percent = proc.memory_percent()
            create_time = proc.create_time()
            current_time = time.time()
            runtime = current_time - create_time
            
            # Get process info
            cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else 'Unknown'
            
            # Check for stuck conditions
            stuck_reasons = []
            
            # High CPU usage for extended period
            if runtime > 60 and cpu_percent > self.cpu_threshold:
                stuck_reasons.append(f"High CPU usage: {cpu_percent:.1f}% for {runtime:.1f}s")
            
            # High memory usage
            if memory_percent > self.memory_threshold:
                stuck_reasons.append(f"High memory usage: {memory_percent:.1f}%")
            
            # Process running too long with any significant resource usage
            if runtime > self.runtime_threshold and (cpu_percent > 50 or memory_percent > 30):
                stuck_reasons.append(f"Long runtime with resource usage: {runtime:.1f}s")
            
            # Excessive resource usage
            if cpu_percent > 95 or memory_percent > 90:
                stuck_reasons.append(f"Excessive resource usage: CPU={cpu_percent:.1f}%, Mem={memory_percent:.1f}%")
            
            # Check if process is unresponsive (no CPU but running long)
            if runtime > 600 and cpu_percent < 1:  # 10 minutes with no CPU
                stuck_reasons.append(f"Unresponsive process: {runtime:.1f}s with {cpu_percent:.1f}% CPU")
            
            return len(stuck_reasons) > 0, stuck_reasons, cmdline
            
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            return False, [], "Process no longer exists"
    
    def can_restart_process(self, pid, cmdline):
        """Check if we can restart this process"""
        current_time = time.time()
        
        # Check restart count
        restart_count = self.process_restart_count.get(pid, 0)
        if restart_count >= self.max_restart_attempts:
            return False, f"Max restart attempts ({self.max_restart_attempts}) exceeded"
        
        # Check cooldown period
        last_restart = self.last_restart_time.get(pid, 0)
        if current_time - last_restart < self.restart_cooldown:
            remaining = self.restart_cooldown - (current_time - last_restart)
            return False, f"Restart cooldown active: {remaining:.1f}s remaining"
        
        # Don't restart system processes
        if any(system_proc in cmdline.lower() for system_proc in ['systemd', 'kernel', 'unattended-upgrades']):
            return False, "System process - not restarting"
        
        return True, "Can restart"
    
    def restart_process(self, proc, cmdline):
        """Attempt to restart a process"""
        try:
            pid = proc.pid
            
            # Log the restart attempt
            logging.warning(f"Attempting to restart process {pid}")
            logging.warning(f"Command: {cmdline}")
            
            # Kill the stuck process
            proc.terminate()
            try:
                proc.wait(timeout=5)
                logging.info(f"Process {pid} terminated gracefully")
            except psutil.TimeoutExpired:
                proc.kill()
                logging.info(f"Process {pid} force killed")
            
            # Update restart tracking
            self.process_restart_count[pid] = self.process_restart_count.get(pid, 0) + 1
            self.last_restart_time[pid] = time.time()
            
            # Try to restart if it's a known command
            if 'streamlit' in cmdline.lower():
                self.restart_streamlit_process(cmdline)
            elif 'python' in cmdline.lower() and 'huggingface' in cmdline.lower():
                logging.info(f"Not restarting Hugging Face download - user should restart manually")
            else:
                logging.info(f"Unknown process type - not auto-restarting")
            
            return True
            
        except Exception as e:
            logging.error(f"Error restarting process {pid}: {e}")
            return False
    
    def restart_streamlit_process(self, cmdline):
        """Restart a Streamlit process"""
        try:
            # Extract port from command line
            port = None
            if '--server.port' in cmdline:
                parts = cmdline.split('--server.port')
                if len(parts) > 1:
                    port = parts[1].split()[0]
            
            if port:
                logging.info(f"Restarting Streamlit on port {port}")
                # Start in background
                subprocess.Popen(cmdline, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"Streamlit restarted on port {port}")
            else:
                logging.warning("Could not determine Streamlit port - not restarting")
                
        except Exception as e:
            logging.error(f"Error restarting Streamlit: {e}")
    
    def monitor_processes(self):
        """Main monitoring loop"""
        logging.info("Smart Monitor started")
        logging.info(f"Monitoring parameters: CPU threshold={self.cpu_threshold}%, Memory threshold={self.memory_threshold}%, Runtime threshold={self.runtime_threshold}s")
        
        while self.running:
            try:
                # Get all Python processes
                python_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
                    try:
                        if proc.info['name'] and 'python' in proc.info['name'].lower():
                            python_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if python_processes:
                    logging.debug(f"Monitoring {len(python_processes)} Python processes")
                    
                    for proc in python_processes:
                        try:
                            is_stuck, reasons, cmdline = self.is_process_stuck(proc)
                            
                            if is_stuck:
                                pid = proc.pid
                                logging.warning(f"Stuck process detected - PID: {pid}")
                                for reason in reasons:
                                    logging.warning(f"  - {reason}")
                                
                                can_restart, restart_reason = self.can_restart_process(pid, cmdline)
                                
                                if can_restart:
                                    success = self.restart_process(proc, cmdline)
                                    if success:
                                        logging.info(f"Successfully handled stuck process {pid}")
                                    else:
                                        logging.error(f"Failed to handle stuck process {pid}")
                                else:
                                    logging.warning(f"Cannot restart process {pid}: {restart_reason}")
                            else:
                                # Log healthy process status occasionally
                                cpu_percent = proc.cpu_percent()
                                memory_percent = proc.memory_percent()
                                if cpu_percent > 10 or memory_percent > 10:  # Only log if using resources
                                    logging.debug(f"Process {proc.pid}: CPU={cpu_percent:.1f}%, Memory={memory_percent:.1f}%")
                                
                        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                            continue
                        except Exception as e:
                            logging.error(f"Error monitoring process {proc.pid}: {e}")
                else:
                    logging.debug("No Python processes found")
                
                # Wait before next check
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logging.info("Smart monitor stopped by user")
                break
            except Exception as e:
                logging.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def stop(self):
        """Stop the monitor"""
        self.running = False
        logging.info("Smart monitor stopping...")

def main():
    """Main function"""
    print("🚀 Smart Process Monitor - Advanced Process Management")
    print("Press Ctrl+C to stop monitoring")
    print("Logs will be saved to smart_monitor.log")
    print("=" * 60)
    
    monitor = SmartMonitor()
    
    try:
        monitor.monitor_processes()
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped by user")
    finally:
        monitor.stop()

if __name__ == "__main__":
    main()

