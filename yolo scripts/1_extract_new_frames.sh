#!/usr/bin/env bash
set -uo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

in_dir="$project_root/dataset/NEW videos"
old_dir="$project_root/dataset/OLD videos"
out_dir="$project_root/dataset/images/NEW"

mkdir -p "$out_dir"

shopt -s nullglob
for f in "$in_dir"/*.mp4 "$in_dir"/*.MP4 "$in_dir"/*.mov "$in_dir"/*.MOV "$in_dir"/*.mkv "$in_dir"/*.MKV; do
  base="$(basename "$f")"
  name="${base%.*}"
  ffmpeg -i "$f" -vf "fps=2,scale=1280:-2" "$out_dir/${name}_%06d.jpg"
  
  # move source video to OLD videos only if ffmpeg succeeded
  mv "$f" "$old_dir/$base"
done

read -p "Press enter to continue"