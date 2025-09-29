#!/usr/bin/env python3
"""
Quick script to identify and kill stuck Python processes
"""

import psutil
import time
import signal
import sys

def find_stuck_processes():
    """Find potentially stuck Python processes"""
    stuck_processes = []
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
        try:
            if proc.info['name'] and 'python' in proc.info['name'].lower():
                cpu_percent = proc.cpu_percent()
                memory_percent = proc.memory_percent()
                create_time = proc.create_time()
                current_time = time.time()
                runtime = current_time - create_time
                
                # Check for stuck processes
                is_stuck = False
                reason = ""
                
                if runtime > 300 and (cpu_percent > 95 or memory_percent > 80):  # 5+ min with high usage
                    is_stuck = True
                    reason = f"High resource usage for {runtime:.1f}s (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
                elif cpu_percent > 99 or memory_percent > 90:  # Excessive usage
                    is_stuck = True
                    reason = f"Excessive resource usage (CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%)"
                elif runtime > 3600:  # Running too long
                    is_stuck = True
                    reason = f"Running too long: {runtime:.1f}s"
                
                if is_stuck:
                    cmdline = ' '.join(proc.cmdline()) if proc.cmdline() else 'Unknown'
                    stuck_processes.append({
                        'proc': proc,
                        'pid': proc.pid,
                        'reason': reason,
                        'cmdline': cmdline,
                        'cpu': cpu_percent,
                        'memory': memory_percent,
                        'runtime': runtime
                    })
                    
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return stuck_processes

def kill_process(proc_info):
    """Kill a stuck process"""
    try:
        pid = proc_info['pid']
        print(f"Attempting to kill process {pid}...")
        print(f"Command: {proc_info['cmdline']}")
        print(f"Reason: {proc_info['reason']}")
        
        # Try graceful termination first
        proc_info['proc'].terminate()
        print(f"Sent SIGTERM to process {pid}")
        
        # Wait for graceful termination
        try:
            proc_info['proc'].wait(timeout=5)
            print(f"✓ Process {pid} terminated gracefully")
            return True
        except psutil.TimeoutExpired:
            print(f"Process {pid} did not terminate gracefully, forcing kill...")
            proc_info['proc'].kill()
            print(f"✓ Process {pid} force killed")
            return True
            
    except psutil.NoSuchProcess:
        print(f"✓ Process {pid} already terminated")
        return True
    except psutil.AccessDenied:
        print(f"✗ Access denied when trying to kill process {pid}")
        return False
    except Exception as e:
        print(f"✗ Error killing process {pid}: {e}")
        return False

def main():
    """Main function"""
    print("Checking for stuck Python processes...")
    print("=" * 50)
    
    stuck_processes = find_stuck_processes()
    
    if not stuck_processes:
        print("✓ No stuck processes found")
        return
    
    print(f"Found {len(stuck_processes)} potentially stuck processes:")
    print()
    
    for i, proc_info in enumerate(stuck_processes, 1):
        print(f"{i}. PID: {proc_info['pid']}")
        print(f"   CPU: {proc_info['cpu']:.1f}%, Memory: {proc_info['memory']:.1f}%")
        print(f"   Runtime: {proc_info['runtime']:.1f}s")
        print(f"   Reason: {proc_info['reason']}")
        print(f"   Command: {proc_info['cmdline']}")
        print()
    
    # Ask for confirmation
    if len(sys.argv) > 1 and sys.argv[1] == '--auto':
        auto_kill = True
    else:
        response = input("Do you want to kill these processes? (y/N): ").strip().lower()
        auto_kill = response in ['y', 'yes']
    
    if auto_kill:
        print("\nKilling stuck processes...")
        print("-" * 30)
        
        success_count = 0
        for proc_info in stuck_processes:
            if kill_process(proc_info):
                success_count += 1
            print()
        
        print(f"Successfully killed {success_count}/{len(stuck_processes)} processes")
    else:
        print("No processes were killed")

if __name__ == "__main__":
    main()

