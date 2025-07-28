nohup python -u gpu_monitor.py > gpu_monitor.log 2>&1 &
echo $! > gpu_monitor_pid.log
