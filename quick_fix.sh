#!/bin/bash

# Quick fix for stuck processes - one command solution
echo "🔍 Checking for stuck Python processes..."

# Find and kill stuck processes
python3 -c "
import psutil, time, signal
stuck = []
for p in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent', 'create_time']):
    try:
        if p.info['name'] and 'python' in p.info['name'].lower():
            cpu = p.cpu_percent()
            mem = p.memory_percent()
            runtime = time.time() - p.create_time()
            if (runtime > 300 and (cpu > 95 or mem > 80)) or cpu > 99 or mem > 90 or runtime > 3600:
                print(f'🚨 Killing stuck process PID {p.pid} (CPU: {cpu:.1f}%, Mem: {mem:.1f}%, Runtime: {runtime:.1f}s)')
                p.terminate()
                try:
                    p.wait(timeout=3)
                    print(f'✅ Process {p.pid} terminated gracefully')
                except:
                    p.kill()
                    print(f'💀 Process {p.pid} force killed')
                stuck.append(p.pid)
    except:
        continue

if not stuck:
    print('✅ No stuck processes found')
else:
    print(f'🎯 Fixed {len(stuck)} stuck processes')
"

echo "✨ Done! All stuck processes have been handled."

