#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=speed_signs.yaml
 
read -p "Press enter to continue"