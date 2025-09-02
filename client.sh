#!/bin/bash

# Function to handle script termination
cleanup() {
    echo "Stopping client processes..."
    if [ -n "$PID1" ]; then
        kill "$PID1" 2>/dev/null
    fi
    if [ -n "$PID2" ]; then
        kill "$PID2" 2>/dev/null
    fi
    echo "All client processes stopped."
    exit 0
}

# Set up trap to catch termination signals
trap cleanup SIGINT SIGTERM

echo "Starting client processes in parallel..."

# Start first client in background
uv run client.py --server-url "127.0.0.1:8789" --data ./data/csv2.csv &
PID1=$!

# Start second client in background
uv run client.py --server-url "127.0.0.1:8789" --data ./data/csv1.csv &
PID2=$!

echo "Client 1 (csv2.csv) running with PID: $PID1"
echo "Client 2 (csv1.csv) running with PID: $PID2"
echo "Press Ctrl+C to stop all clients"

# Wait for both processes to complete
wait $PID1
wait $PID2