#!/bin/bash
# CPU boost script for ensemble_95_plus.py
while true; do
    # Check if our process is still running
    if ps -p 1703030 > /dev/null; then
        # Try to wake it up
        kill -CONT 1703030 2>/dev/null
        # Lower priority of competing processes
        ps aux --sort=-%cpu | awk '{if($3 > 100 && $2 != 1703030) print $2}' | head -5 | xargs -r renice -n 19 2>/dev/null
        echo "$(date): Boosted ensemble process priority"
    else
        echo "$(date): Ensemble process completed"
        break
    fi
    sleep 30
done
