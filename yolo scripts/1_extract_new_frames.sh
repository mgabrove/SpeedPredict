#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

in_dir="$project_root/dataset/videos"
out_dir="$project_root/dataset/images/NEW"

mkdir -p "$out_dir"

shopt -s nullglob
for f in "$in_dir"/*.mp4 "$in_dir"/*.MP4 "$in_dir"/*.mov "$in_dir"/*.MOV "$in_dir"/*.mkv "$in_dir"/*.MKV; do
  base="$(basename "$f")"
  name="${base%.*}"
  ffmpeg -i "$f" -vf "fps=2,scale=1280:-2" "$out_dir/${name}_%06d.jpg"
done

read -p "Press enter to continue"