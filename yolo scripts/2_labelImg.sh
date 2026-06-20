#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

labelImg "$project_root/dataset/images/NEW" \
         "$project_root/dataset/labels/predefined_classes.txt" \
         "$project_root/dataset/labels/NEW"

read -p "Press enter to continue"