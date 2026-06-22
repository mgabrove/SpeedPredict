#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import cv2
from ultralytics import YOLO

SPEED_CLASSES = {"30", "40", "50", "60", "70", "80"}
OUT_FOLDERS = ["unknown", "30", "40", "50", "60", "70", "80"]


@dataclass
class TrackState:
    track_id: int
    cls_votes: Dict[int, float] = field(default_factory=dict)  # class_id -> conf-weighted votes
    conf_sum: float = 0.0
    conf_count: int = 0
    max_area: float = 0.0
    first_frame: int = -1
    last_frame: int = -1
    misses: int = 0
    frames_seen: int = 0

    def add_obs(self, frame_idx: int, cls_id: int, conf: float, area: float):
        if self.first_frame < 0:
            self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.frames_seen += 1
        self.conf_sum += conf
        self.conf_count += 1
        self.max_area = max(self.max_area, area)
        self.cls_votes[cls_id] = self.cls_votes.get(cls_id, 0.0) + conf
        self.misses = 0

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(1, self.conf_count)

    def voted_class(self) -> int:
        return max(self.cls_votes.items(), key=lambda kv: kv[1])[0]


@dataclass
class SignEvent:
    track_id: int
    start_frame: int
    end_frame: int
    voted_cls: int
    mean_conf: float
    frames_seen: int
    max_area: float
    score: float


class SegmentWriter:
    def __init__(self, clips_root: Path, video_stem: str, fps: float, size_wh):
        self.clips_root = clips_root
        self.video_stem = video_stem
        self.fps = fps
        self.size_wh = size_wh  # (w, h)

        self.current_speed_folder = "unknown"
        self.segment_index = 0
        self.writer = None
        self.segment_start_frame = 0
        self.current_path = None
        self.records = []

        self._open_new_writer(self.current_speed_folder, start_frame=0)

    def _segment_path(self, speed_folder: str, idx: int) -> Path:
        out_dir = self.clips_root / speed_folder
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{self.video_stem}_seg_{idx:04d}.mp4"

    def _open_new_writer(self, speed_folder: str, start_frame: int):
        path = self._segment_path(speed_folder, self.segment_index)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, self.size_wh)
        self.segment_start_frame = start_frame
        self.current_path = path

    def _close_current(self, end_frame_inclusive: int):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.records.append({
                "segment_index": self.segment_index,
                "speed_folder": self.current_speed_folder,
                "file": str(self.current_path),
                "start_frame": self.segment_start_frame,
                "end_frame": end_frame_inclusive
            })

    def switch_speed_folder(self, new_speed_folder: str, frame_idx: int):
        if new_speed_folder == self.current_speed_folder:
            return
        self._close_current(frame_idx - 1)
        self.segment_index += 1
        self.current_speed_folder = new_speed_folder
        self._open_new_writer(new_speed_folder, start_frame=frame_idx)

    def write(self, frame):
        if self.writer is not None:
            self.writer.write(frame)

    def finalize(self, last_frame_idx: int):
        self._close_current(last_frame_idx)


def pick_authoritative_event(events: List[SignEvent]) -> Optional[SignEvent]:
    if not events:
        return None
    return max(events, key=lambda e: e.score)


def find_videos(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    exts = {".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV", ".avi", ".AVI"}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix in exts])


def process_video(model: YOLO, video_path: Path, clips_root: Path, args) -> Optional[dict]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    segment_writer = SegmentWriter(clips_root=clips_root, video_stem=video_path.stem, fps=fps, size_wh=(w, h))

    # Speed state
    current_speed: Optional[str] = None        # None = unknown
    pre_30_speed: Optional[str] = None         # speed before entering current 30 zone
    last_transition_frame = -10**9

    # Track management
    active_tracks: Dict[int, TrackState] = {}

    transitions = []
    frame_idx = -1

    results = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker=args.tracker,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        verbose=False
    )

    for r in results:
        frame_idx += 1
        frame = r.orig_img

        # everyone gets one miss by default this frame
        for ts in active_tracks.values():
            ts.misses += 1

        # update active tracks from this frame detections
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxys = r.boxes.xyxy.cpu().tolist()

            for tid, cid, conf, xyxy in zip(ids, clss, confs, xyxys):
                x1, y1, x2, y2 = xyxy
                area = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, w * h))

                if tid not in active_tracks:
                    active_tracks[tid] = TrackState(track_id=tid)
                active_tracks[tid].add_obs(frame_idx=frame_idx, cls_id=int(cid), conf=float(conf), area=area)

        # finalize tracks that disappeared long enough
        finalized_events: List[SignEvent] = []
        remove_ids = []

        for tid, ts in active_tracks.items():
            if ts.misses > args.max_gap:
                if ts.frames_seen >= args.min_frames and ts.mean_conf >= args.min_mean_conf:
                    voted_cls = ts.voted_class()
                    # combined score for "most likely sign" selection
                    score = (0.5 * ts.mean_conf) + (0.3 * ts.max_area) + (0.2 * min(1.0, ts.frames_seen / 30.0))
                    finalized_events.append(SignEvent(
                        track_id=tid,
                        start_frame=ts.first_frame,
                        end_frame=ts.last_frame,
                        voted_cls=voted_cls,
                        mean_conf=ts.mean_conf,
                        frames_seen=ts.frames_seen,
                        max_area=ts.max_area,
                        score=score
                    ))
                remove_ids.append(tid)

        for tid in remove_ids:
            active_tracks.pop(tid, None)

        # Apply at most one transition when sign(s) leave view
        if finalized_events and (frame_idx - last_transition_frame) >= args.cooldown:
            winner = pick_authoritative_event(finalized_events)
            if winner is not None:
                cls_name = str(model.names[int(winner.voted_cls)]) if isinstance(model.names, dict) else str(model.names[int(winner.voted_cls)])

                prev_speed = current_speed
                transition_type = "ignored"

                if cls_name == "end_30":
                    # End zone 30 only if currently in 30
                    if current_speed == "30":
                        current_speed = pre_30_speed
                        pre_30_speed = None
                        transition_type = "end_30_return"
                    else:
                        transition_type = "end_30_ignored"

                elif cls_name in SPEED_CLASSES:
                    if cls_name == "30":
                        if current_speed == "30":
                            # repeated 30 sign -> keep 30, do not overwrite pre_30_speed
                            transition_type = "repeat_30_keep"
                        else:
                            # entering 30 zone
                            pre_30_speed = current_speed
                            current_speed = "30"
                            transition_type = "set_30"
                    else:
                        # non-30 speed signs set speed directly
                        current_speed = cls_name
                        # override clears pending 30-context
                        pre_30_speed = None
                        transition_type = "set_speed_non30"

                new_folder = current_speed if current_speed is not None else "unknown"
                segment_writer.switch_speed_folder(new_folder, frame_idx)
                last_transition_frame = frame_idx

                transitions.append({
                    "frame": frame_idx,
                    "winner_track_id": winner.track_id,
                    "winner_class": cls_name,
                    "winner_score": winner.score,
                    "event": {
                        "start_frame": winner.start_frame,
                        "end_frame": winner.end_frame,
                        "mean_conf": winner.mean_conf,
                        "frames_seen": winner.frames_seen,
                        "max_area": winner.max_area
                    },
                    "transition_type": transition_type,
                    "prev_speed": prev_speed,
                    "new_speed": current_speed,
                    "pre_30_speed": pre_30_speed
                })

        # always write frame to current segment
        segment_writer.write(frame)

    segment_writer.finalize(frame_idx)

    return {
        "video": str(video_path),
        "total_frames": total_frames,
        "fps": fps,
        "segments": segment_writer.records,
        "transitions": transitions
    }


def main():
    parser = argparse.ArgumentParser("Extract speed-labeled clips from dataset/OLD videos")
    parser.add_argument("--model", required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--input-dir", default="dataset/OLD videos")
    parser.add_argument("--output-dir", default="dataset/CLIPS")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

    # event robustness
    parser.add_argument("--max-gap", type=int, default=10, help="missed frames before track finalization")
    parser.add_argument("--min-frames", type=int, default=6, help="minimum track length to accept sign event")
    parser.add_argument("--min-mean-conf", type=float, default=0.45, help="minimum mean confidence for accepted event")
    parser.add_argument("--cooldown", type=int, default=15, help="minimum frames between transitions")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for folder in OUT_FOLDERS:
        (output_dir / folder).mkdir(parents=True, exist_ok=True)

    videos = find_videos(input_dir)
    if not videos:
        print(f"[INFO] No videos found in: {input_dir}")
        return

    model = YOLO(args.model)

    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "model": args.model,
        "videos_processed": 0,
        "videos": []
    }

    for i, vp in enumerate(videos, start=1):
        print(f"[{i}/{len(videos)}] Processing {vp.name} ...")
        result = process_video(model, vp, output_dir, args)
        if result is not None:
            summary["videos_processed"] += 1
            summary["videos"].append(result)

    summary_path = output_dir / "clips_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Processed {summary['videos_processed']} video(s).")
    print(f"[DONE] Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()