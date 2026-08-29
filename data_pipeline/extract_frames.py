"""
extract_frames.py

Fixed-sampling-rate frame extractor for the deepfake detection data pipeline.

Given a video (or a folder of videos), extracts frames at a fixed sampling
rate (frames per second), and saves them into per-video subfolders so the
downstream MTCNN face-alignment step (Week 1, next task) can consume them
directly.

Usage:
    # Single video
    python extract_frames.py --input path/to/video.mp4 --output_dir data_pipeline/frames --fps 1

    # Folder of videos (batch mode)
    python extract_frames.py --input path/to/videos_folder --output_dir data_pipeline/frames --fps 1
"""

import argparse
import logging
from pathlib import Path

import cv2
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def extract_frames_from_video(video_path: Path, output_dir: Path, target_fps: float) -> int:
    """
    Extract frames from a single video at a fixed sampling rate.

    Args:
        video_path: path to the input video file.
        output_dir: directory where this video's frames will be saved
                     (a subfolder named after the video is created inside it).
        target_fps: how many frames per second of video to extract.
                     e.g. target_fps=1 pulls 1 frame per second of footage,
                     regardless of the video's native frame rate.

    Returns:
        Number of frames successfully written.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video, skipping: {video_path}")
        return 0

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    if not native_fps or native_fps <= 0:
        logger.warning(f"Could not read native FPS for {video_path}, assuming 30.")
        native_fps = 30.0

    # How many native frames to advance between each extracted frame.
    frame_interval = max(int(round(native_fps / target_fps)), 1)

    video_name = video_path.stem
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index % frame_interval == 0:
            frame_filename = video_output_dir / f"frame_{saved_count:05d}.jpg"
            cv2.imwrite(str(frame_filename), frame)
            saved_count += 1

        frame_index += 1

    cap.release()
    return saved_count


def collect_video_paths(input_path: Path) -> list[Path]:
    """Return a list of video file paths from a single file or a directory."""
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not videos:
            raise ValueError(f"No video files found in directory: {input_path}")
        return videos

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def main():
    parser = argparse.ArgumentParser(description="Fixed-rate frame extractor for deepfake detection pipeline.")
    parser.add_argument("--input", required=True, help="Path to a video file or a folder of videos.")
    parser.add_argument("--output_dir", required=True, help="Directory to save extracted frames into.")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames to extract per second of video (default: 1.0).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional: only process the first N videos found (useful for quick pipeline "
                              "tests on a new dataset before running the full batch).")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    video_paths = collect_video_paths(input_path)
    logger.info(f"Found {len(video_paths)} video(s) total.")

    if args.limit is not None:
        video_paths = video_paths[:args.limit]
        logger.info(f"--limit {args.limit} set: processing only {len(video_paths)} video(s).")

    total_frames = 0
    failed_videos = []

    progress_bar = tqdm(video_paths, desc="Extracting frames", unit="video")
    for video_path in progress_bar:
        try:
            count = extract_frames_from_video(video_path, output_dir, args.fps)
            if count == 0:
                failed_videos.append(video_path.name)
            total_frames += count
            progress_bar.set_postfix({"frames_so_far": total_frames})
        except Exception as e:
            logger.error(f"Failed on {video_path.name}: {e}")
            failed_videos.append(video_path.name)

    logger.info(f"Done. Total frames extracted: {total_frames} from {len(video_paths)} video(s).")
    if failed_videos:
        logger.warning(f"Videos with issues ({len(failed_videos)}): {failed_videos}")


if __name__ == "__main__":
    main()
