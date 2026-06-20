#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
cd "$project_root"

yolo detect val \
  model=runs/detect/train/weights/best.pt \
  data=speed_signs.yaml
 
read -p "Press enter to continue"