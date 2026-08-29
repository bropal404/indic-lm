#!/bin/bash
# Phase 2 Training Monitor
# Usage: ./monitor.sh [hindi|nepali|both|status|checkpoints]

LOG_DIR="phase2/logs"
CKPT_DIR="phase2/checkpoints"

case "${1:-both}" in
  hindi)
    tail -f "$LOG_DIR/train_hindi.log"
    ;;
  nepali)
    tail -f "$LOG_DIR/train_nepali.log"
    ;;
  both)
    echo "=== Hindi (Ctrl+C to switch) ==="
    tail -f "$LOG_DIR/train_hindi.log" &
    HINDI_PID=$!
    sleep 1
    echo "=== Nepali ==="
    tail -f "$LOG_DIR/train_nepali.log"
    kill $HINDI_PID 2>/dev/null
    ;;
  status)
    echo "=== Process Status ==="
    ps aux | grep -E "train.py" | grep -v grep
    echo ""
    echo "=== Latest Log Lines ==="
    echo "--- Hindi ---"
    tail -3 "$LOG_DIR/train_hindi.log" 2>/dev/null || echo "No log file"
    echo "--- Nepali ---"
    tail -3 "$LOG_DIR/train_nepali.log" 2>/dev/null || echo "No log file"
    ;;
  checkpoints)
    echo "=== Hindi Checkpoints ==="
    ls -lh "$CKPT_DIR/model_h/" 2>/dev/null || echo "No checkpoints"
    echo ""
    echo "=== Nepali Checkpoints ==="
    ls -lh "$CKPT_DIR/model_l/" 2>/dev/null || echo "No checkpoints"
    ;;
  steps)
    echo "=== Step Progress ==="
    echo "Hindi: $(grep -c 'Training:' "$LOG_DIR/train_hindi.log" 2>/dev/null || echo 0) / 5000"
    echo "Nepali: $(grep -c 'Training:' "$LOG_DIR/train_nepali.log" 2>/dev/null || echo 0) / 5000"
    ;;
  restart)
    model="${2:-hindi}"
    config="phase2/configs/model_${model}.yaml"
    log="$LOG_DIR/train_${model}.log"
    echo "Restarting $model..."
    nohup python -u phase2/train.py --config "$config" > "$log" 2>&1 & disown %1
    echo "Started $model (PID: $!)"
    ;;
  *)
    echo "Usage: $0 {hindi|nepali|both|status|checkpoints|steps|restart [hindi|nepali]}"
    echo ""
    echo "Commands:"
    echo "  hindi      - Follow Hindi training log"
    echo "  nepali     - Follow Nepali training log"
    echo "  both       - Show both logs (Hindi first, then Nepali)"
    echo "  status     - Quick status check (processes + latest logs)"
    echo "  checkpoints- List saved checkpoints"
    echo "  steps      - Show current step count"
    echo "  restart    - Restart a model (default: hindi)"
    ;;
esac