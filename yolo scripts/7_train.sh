#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"

yolo detect train \
  model=yolov8n.pt \
  data=speed_signs.yaml \
  imgsz=512 \
  epochs=80 \
  batch=4 \
  workers=0 \
  device=0 \
  cache=False \
  patience=20
  
read -p "Press enter to continue"