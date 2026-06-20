#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

labelImg "$PROJECT_ROOT/dataset/images/NEW" \
         "$PROJECT_ROOT/dataset/labels/predefined_classes.txt" \
         "$PROJECT_ROOT/dataset/labels/NEW"