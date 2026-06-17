import argparse
import configparser
from pathlib import Path

import cv2
import supervision as sv
from rfdetr import RFDETRMedium


DEFAULT_CHECKPOINT = Path("checkpoint_best_ema.pth")
DEFAULT_FRAMES_DIR = Path("mot17_yolov8/test/MOT17-01-DPM/img1")
DEFAULT_OUTPUT_VIDEO = Path("MOT17-01-DPM_rfdetr.mp4")


def read_sequence_fps(frames_dir: Path, default_fps: float = 30.0) -> float:
    seqinfo_path = frames_dir.parent / "seqinfo.ini"
    if not seqinfo_path.exists():
        return default_fps

    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    return config.getfloat("Sequence", "frameRate", fallback=default_fps)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference on MOT17 frames and export an annotated video."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit number of frames for a quick test run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame_paths = sorted(args.frames_dir.glob("*.jpg"))
    if args.max_frames is not None:
        frame_paths = frame_paths[: args.max_frames]

    if not frame_paths:
        raise FileNotFoundError(f"No JPG frames found in: {args.frames_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    fps = args.fps if args.fps is not None else read_sequence_fps(args.frames_dir)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Could not open video writer for: {args.output}")

    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = RFDETRMedium().from_checkpoint(str(checkpoint_path))
    box_annotator = sv.BoxAnnotator()

    try:
        for frame_index, frame_path in enumerate(frame_paths, start=1):
            detections = model.predict(str(frame_path), threshold=args.threshold)
            annotated_frame = box_annotator.annotate(
                scene=detections.metadata["source_image"].copy(),
                detections=detections,
            )

            if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                annotated_frame = cv2.resize(annotated_frame, (width, height))

            video_writer.write(annotated_frame)

            if frame_index == 1 or frame_index % 25 == 0 or frame_index == len(frame_paths):
                print(f"Processed {frame_index}/{len(frame_paths)} frames")
    finally:
        video_writer.release()

    print(f"Saved video: {args.output}")


if __name__ == "__main__":
    main()
