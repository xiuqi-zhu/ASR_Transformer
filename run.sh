#!/bin/bash
# Helper script: sets PYTHONPATH to transformer-from-scratch and runs a command.
# Usage: ./run.sh python main.py --mode train ...
# Edit TRANSFORMER_SCRATCH_DIR to your transformer-from-scratch clone path.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRANSFORMER_SCRATCH_DIR="${TRANSFORMER_SCRATCH_DIR:-$SCRIPT_DIR/../transformer-from-scratch}"

if [ ! -d "$TRANSFORMER_SCRATCH_DIR" ]; then
  echo "Error: transformer-from-scratch not found at: $TRANSFORMER_SCRATCH_DIR"
  echo "Clone it: git clone https://github.com/xiuqi-zhu/transformer-from-scratch.git"
  echo "Or set: export TRANSFORMER_SCRATCH_DIR=/path/to/transformer-from-scratch"
  exit 1
fi

export PYTHONPATH="$TRANSFORMER_SCRATCH_DIR:$PYTHONPATH"
cd "$SCRIPT_DIR"
exec "$@"
