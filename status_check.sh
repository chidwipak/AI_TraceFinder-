#!/bin/bash

# Status Check Script - Quick overview of all processes
echo "🔍 AI TraceFinder Process Status"
echo "================================="
echo

# Check Python processes
echo "📊 Python Processes:"
ps aux | grep -E "(python|python3)" | grep -v grep | while read line; do
    pid=$(echo $line | awk '{print $2}')
    cpu=$(echo $line | awk '{print $3}')
    mem=$(echo $line | awk '{print $4}')
    cmd=$(echo $line | awk '{for(i=11;i<=NF;i++) printf "%s ", $i; print ""}')
    
    # Color code based on resource usage
    if (( $(echo "$cpu > 80" | bc -l) )); then
        echo "🚨 PID $pid - CPU: ${cpu}% | MEM: ${mem}% | $cmd"
    elif (( $(echo "$cpu > 50" | bc -l) )); then
        echo "⚠️  PID $pid - CPU: ${cpu}% | MEM: ${mem}% | $cmd"
    else
        echo "✅ PID $pid - CPU: ${cpu}% | MEM: ${mem}% | $cmd"
    fi
done

echo
echo "🌐 Streamlit Apps:"
netstat -tlnp 2>/dev/null | grep -E ":(8508|8509|8510)" | while read line; do
    port=$(echo $line | awk '{print $4}' | cut -d: -f2)
    echo "   Port $port: Active"
done

echo
echo "📈 System Resources:"
echo "   CPU Load: $(uptime | awk -F'load average:' '{print $2}')"
echo "   Memory: $(free -h | grep '^Mem:' | awk '{print $3 "/" $2}')"
echo "   Disk: $(df -h . | tail -1 | awk '{print $3 "/" $2 " (" $5 " used)"}')"

echo
echo "🛠️  Monitoring Status:"
if pgrep -f "smart_monitor.py" > /dev/null; then
    echo "   ✅ Smart Monitor: Running (PID: $(pgrep -f smart_monitor.py))"
else
    echo "   ❌ Smart Monitor: Not running"
fi

echo
echo "💡 Quick Commands:"
echo "   ./quick_fix.sh          - Kill stuck processes"
echo "   ./status_check.sh       - This status check"
echo "   tail -f smart_monitor.log - Watch monitor logs"

