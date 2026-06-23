#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
    max_conf: float = 0.0
    max_area: float = 0.0
    first_frame: int = -1
    last_frame: int = -1
    misses: int = 0
    frames_seen: int = 0

    # for optional center gating statistics
    center_hits: int = 0

    def add_obs(self, frame_idx: int, cls_id: int, conf: float, area: float, in_center_region: bool):
        if self.first_frame < 0:
            self.first_frame = frame_idx
        self.last_frame = frame_idx
        self.frames_seen += 1

        self.conf_sum += conf
        self.conf_count += 1
        self.max_conf = max(self.max_conf, conf)
        self.max_area = max(self.max_area, area)

        self.cls_votes[cls_id] = self.cls_votes.get(cls_id, 0.0) + conf
        self.misses = 0

        if in_center_region:
            self.center_hits += 1

    @property
    def mean_conf(self) -> float:
        return self.conf_sum / max(1, self.conf_count)

    @property
    def total_vote(self) -> float:
        return sum(self.cls_votes.values()) if self.cls_votes else 0.0

    def voted_class(self) -> int:
        return max(self.cls_votes.items(), key=lambda kv: kv[1])[0]

    def dominance(self) -> float:
        if not self.cls_votes:
            return 0.0
        winner_vote = max(self.cls_votes.values())
        total = self.total_vote
        return (winner_vote / total) if total > 0 else 0.0

    def center_ratio(self) -> float:
        return self.center_hits / max(1, self.frames_seen)


@dataclass
class SignEvent:
    track_id: int
    start_frame: int
    end_frame: int
    voted_cls: int
    mean_conf: float
    max_conf: float
    frames_seen: int
    max_area: float
    dominance: float
    center_ratio: float
    score: float


class SegmentWriter:
    def __init__(self, clips_root: Path, video_stem: str, fps: float, size_wh: Tuple[int, int]):
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


def in_center_region(x1, y1, x2, y2, w, h, center_x_margin=0.45, center_y_margin=0.425) -> bool:
    """
    center_x_margin=0.45 means keep x-center in [0.05, 0.95]
    center_y_margin=0.425 means keep y-center in [0.075, 0.925]
    """
    cx = 0.5 * (x1 + x2) / max(1.0, w)
    cy = 0.5 * (y1 + y2) / max(1.0, h)

    x_min = 0.5 - center_x_margin
    x_max = 0.5 + center_x_margin
    y_min = 0.5 - center_y_margin
    y_max = 0.5 + center_y_margin

    return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)


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
    current_speed: Optional[str] = None   # None = unknown
    pre_30_speed: Optional[str] = None    # remembered speed before entering 30
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

        # everyone missed by default each frame
        for ts in active_tracks.values():
            ts.misses += 1

        # update tracks from detections
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            ids = r.boxes.id.int().cpu().tolist()
            clss = r.boxes.cls.int().cpu().tolist()
            confs = r.boxes.conf.cpu().tolist()
            xyxys = r.boxes.xyxy.cpu().tolist()

            for tid, cid, conf, xyxy in zip(ids, clss, confs, xyxys):
                x1, y1, x2, y2 = xyxy
                area = max(0.0, (x2 - x1) * (y2 - y1)) / float(max(1, w * h))

                center_ok = in_center_region(
                    x1, y1, x2, y2, w, h,
                    center_x_margin=args.center_x_margin,
                    center_y_margin=args.center_y_margin
                )

                if tid not in active_tracks:
                    active_tracks[tid] = TrackState(track_id=tid)

                active_tracks[tid].add_obs(
                    frame_idx=frame_idx,
                    cls_id=int(cid),
                    conf=float(conf),
                    area=area,
                    in_center_region=center_ok
                )

        # finalize disappeared tracks
        finalized_events: List[SignEvent] = []
        remove_ids = []

        for tid, ts in active_tracks.items():
            if ts.misses > args.max_gap:
                # --- anti-ghost gating ---
                passes = (
                    ts.frames_seen >= args.min_frames
                    and ts.mean_conf >= args.min_mean_conf
                    and ts.max_conf >= args.min_peak_conf
                    and ts.max_area >= args.min_max_area
                    and ts.dominance() >= args.min_dominance
                )

                if args.use_center_gate:
                    passes = passes and (ts.center_ratio() >= args.min_center_ratio)

                if passes:
                    voted_cls = ts.voted_class()
                    # weighted score for winner selection
                    score = (
                        0.40 * ts.mean_conf
                        + 0.20 * ts.max_conf
                        + 0.20 * ts.max_area
                        + 0.20 * ts.dominance()
                    )

                    finalized_events.append(SignEvent(
                        track_id=tid,
                        start_frame=ts.first_frame,
                        end_frame=ts.last_frame,
                        voted_cls=voted_cls,
                        mean_conf=ts.mean_conf,
                        max_conf=ts.max_conf,
                        frames_seen=ts.frames_seen,
                        max_area=ts.max_area,
                        dominance=ts.dominance(),
                        center_ratio=ts.center_ratio(),
                        score=score
                    ))

                remove_ids.append(tid)

        for tid in remove_ids:
            active_tracks.pop(tid, None)

        # apply at most one transition per cooldown window
        if finalized_events and (frame_idx - last_transition_frame) >= args.cooldown:
            winner = pick_authoritative_event(finalized_events)
            if winner is not None:
                cls_name = str(model.names[int(winner.voted_cls)]) if isinstance(model.names, dict) else str(model.names[int(winner.voted_cls)])

                prev_speed = current_speed
                transition_type = "ignored"

                # simplified 30/end_30 logic
                if cls_name == "end_30":
                    if current_speed == "30":
                        current_speed = pre_30_speed
                        pre_30_speed = None
                        transition_type = "end_30_return"
                    else:
                        transition_type = "end_30_ignored"

                elif cls_name in SPEED_CLASSES:
                    if cls_name == "30":
                        if current_speed == "30":
                            transition_type = "repeat_30_keep"
                        else:
                            pre_30_speed = current_speed
                            current_speed = "30"
                            transition_type = "set_30"
                    else:
                        current_speed = cls_name
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
                        "max_conf": winner.max_conf,
                        "frames_seen": winner.frames_seen,
                        "max_area": winner.max_area,
                        "dominance": winner.dominance,
                        "center_ratio": winner.center_ratio
                    },
                    "transition_type": transition_type,
                    "prev_speed": prev_speed,
                    "new_speed": current_speed,
                    "pre_30_speed": pre_30_speed
                })

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
    parser = argparse.ArgumentParser("Extract speed-labeled clips from dataset/OLD videos with anti-ghost voting")

    parser.add_argument("--model", required=True, help="Path to trained weights (best.pt)")
    parser.add_argument("--input-dir", default="dataset/OLD videos")
    parser.add_argument("--output-dir", default="dataset/CLIPS")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)

    # temporal robustness
    parser.add_argument("--max-gap", type=int, default=10, help="missed frames before track finalization")
    parser.add_argument("--cooldown", type=int, default=20, help="minimum frames between accepted transitions")

    # anti-ghost gates
    parser.add_argument("--min-frames", type=int, default=8, help="minimum track length")
    parser.add_argument("--min-mean-conf", type=float, default=0.55, help="minimum mean confidence")
    parser.add_argument("--min-peak-conf", type=float, default=0.70, help="minimum peak confidence in track")
    parser.add_argument("--min-max-area", type=float, default=0.0015, help="minimum max normalized bbox area")
    parser.add_argument("--min-dominance", type=float, default=0.75, help="winner vote / total vote threshold")

    # optional center-region gating
    parser.add_argument("--use-center-gate", action="store_true", help="enable center-region reliability gate")
    parser.add_argument("--min-center-ratio", type=float, default=0.6, help="fraction of obs inside center region")
    parser.add_argument("--center-x-margin", type=float, default=0.45, help="center x half-width in normalized coords")
    parser.add_argument("--center-y-margin", type=float, default=0.425, help="center y half-height in normalized coords")

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
        "params": vars(args),
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