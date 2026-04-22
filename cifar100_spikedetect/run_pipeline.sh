#!/bin/bash
# Chained V3 → V4 → V5 → V6 pipeline runner.
# Monitors the current training, and when its best checkpoint is saved
# AND its process exits, launches the next stage.
#
# Usage:
#   bash cifar100_spikedetect/run_pipeline.sh
#
# Logs go to /workspace/SNN/logs/cifar100_spikedetect_<stage>.log

set -e
cd /workspace/SNN

STAGES=(V3_imagenet V4_mosaic V5_kd V6_spiking)
PYTHON=/workspace/SNN/venv/bin/python
TRAIN=/workspace/SNN/cifar100_spikedetect/train.py

run_stage () {
  local stage="$1"
  local logfile="/workspace/SNN/logs/cifar100_spikedetect_${stage}.log"
  local ckpt="/workspace/SNN/checkpoints/cifar100_spikedetect_${stage}_best.pth"

  if [ -f "$ckpt" ]; then
    # Skip if final epoch already trained — detect by checking log
    if grep -q "Training complete." "$logfile" 2>/dev/null; then
      echo "[$(date +%H:%M:%S)] skipping $stage (already complete)"
      return 0
    fi
  fi

  # Switch EXPERIMENT in train.py
  sed -i "s/^EXPERIMENT = \".*\"$/EXPERIMENT = \"$stage\"/" "$TRAIN"
  echo "[$(date +%H:%M:%S)] launching $stage"
  nohup $PYTHON $TRAIN > /dev/null 2>&1 &
  local pid=$!
  echo "[$(date +%H:%M:%S)]   PID $pid, log: $logfile"

  # Wait for completion
  wait $pid
  local rc=$?
  echo "[$(date +%H:%M:%S)] $stage exited rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "  ABORTING pipeline (stage failed)"
    exit 1
  fi
}

for stage in "${STAGES[@]}"; do
  run_stage "$stage"
done

echo "[$(date +%H:%M:%S)] PIPELINE COMPLETE"
