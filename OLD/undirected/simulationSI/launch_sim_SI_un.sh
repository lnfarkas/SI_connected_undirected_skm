#!/bin/bash

# Get timestamp
TS=$(date +"%Y-%m-%d-%H-%M-%S")

# Log file
LOG="log_${TS}.txt"

# PID file (fixed name, appending)
PIDFILE="process_id.txt"

# Run script in background, redirect output, and save PID
nohup python3 00_sim_SI_un.py > "$LOG" 2>&1 &

# Append PID and timestamp to file
echo "$! $TS" >> "$PIDFILE"

echo "Script started with PID $!. Logs -> $LOG"
