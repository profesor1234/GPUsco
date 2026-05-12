#!/bin/bash
# Scorbits GPU Miner — launch all GPUs
# Usage: ./mine.sh SCOyouraddresshere

ADDRESS=${1:-""}
if [ -z "$ADDRESS" ]; then
    echo "Usage: $0 SCOyouraddresshere"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "Found $GPU_COUNT GPU(s)"

# Kill any existing miners
pkill -f scorbits_gpu 2>/dev/null
sleep 1

# Launch one instance per GPU
for i in $(seq 0 $((GPU_COUNT - 1))); do
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | sed -n "$((i+1))p")
    echo "Starting GPU $i: $GPU_NAME"
    nohup ./scorbits_gpu --gpu $i --address $ADDRESS > log_gpu$i.txt 2>&1 &
    echo "  PID: $! | Log: log_gpu$i.txt"
    sleep 0.5
done

echo ""
echo "All GPUs mining! Monitor with:"
echo "  tail -f log_gpu0.txt"
echo "  watch -n 5 'grep -h \"MH/s\\|Found\\|Accepted\\|Rejected\" log_gpu*.txt | tail -20'"
echo ""
echo "Stop all: pkill -f scorbits_gpu"
