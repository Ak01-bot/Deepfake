"""
mtcnn_align.py

Face detection, filtering, and alignment step for the deepfake detection
data pipeline.

Takes the frame folders produced by extract_frames.py (one subfolder per
video, containing raw extracted frames), runs MTCNN on each frame,
filters out low-confidence and spurious detections, crops/aligns/resizes
the face region, and saves clean face crops ready for CNN input.

Frames where no valid face is detected are discarded (not saved) rather
than causing an error, so a batch run over hundreds of videos doesn't
stop on a single bad frame.

Usage:
    python mtcnn_align.py --input data_pipeline/frames --output_dir data_pipeline/faces
"""

import argparse
import logging
from pathlib import Path

import torch
from facenet_pytorch import MTCNN
from PIL import Image
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Detections below this confidence are treated as noise (background
# textures, shadows, etc. that MTCNN occasionally flags as low-confidence
# "faces"). Validated against real MTCNN output during pipeline testing.
CONFIDENCE_THRESHOLD = 0.95

# Final size CNN backbones (XceptionNet/EfficientNet/ResNet via timm)
# expect for input.
OUTPUT_SIZE = (224, 224)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_best_face(boxes, probs) -> tuple:
    """
    Given all detections MTCNN returned for one frame, decide which one
    (if any) is the real face.

    Filtering logic:
        1. Discard any detection below CONFIDENCE_THRESHOLD.
        2. If more than one detection survives, take the one with the
           largest bounding box area (the real subject's face is
           typically the dominant face in a FaceForensics/Celeb-DF frame;
           small surviving boxes are more likely to still be noise).

    Args:
        boxes: array of [x1, y1, x2, y2] boxes from MTCNN, or None.
        probs: array of confidence scores matching boxes, or None.

    Returns:
        The single best box as a tuple (x1, y1, x2, y2), or None if no
        detection passed the confidence threshold.
    """
    if boxes is None or probs is None:
        return None

    candidates = [
        (box, prob) for box, prob in zip(boxes, probs)
        if prob is not None and prob >= CONFIDENCE_THRESHOLD
    ]

    if not candidates:
        return None

    def box_area(box):
        x1, y1, x2, y2 = box
        return abs((x2 - x1) * (y2 - y1))

    best_box, _ = max(candidates, key=lambda c: box_area(c[0]))
    return tuple(best_box)


def align_face_in_image(image_path: Path, mtcnn: MTCNN) -> Image.Image:
    """
    Detect, filter, crop, and resize the face in a single image.

    Args:
        image_path: path to the input frame image.
        mtcnn: an initialized MTCNN detector instance.

    Returns:
        A cropped, resized PIL Image of the face, or None if no valid
        face was found in this frame.
    """
    img = Image.open(image_path).convert("RGB")
    boxes, probs = mtcnn.detect(img)

    best_box = select_best_face(boxes, probs)
    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    # Clamp to image bounds in case MTCNN returns a slightly out-of-frame box
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(img.width, x2), min(img.height, y2)

    face_crop = img.crop((x1, y1, x2, y2))
    face_resized = face_crop.resize(OUTPUT_SIZE)
    return face_resized


def process_video_folder(video_frames_dir: Path, output_dir: Path, mtcnn: MTCNN) -> dict:
    """
    Run face alignment over every frame in one video's frame folder.

    Args:
        video_frames_dir: folder containing one video's extracted frames
                            (output of extract_frames.py).
        output_dir: base output directory; a subfolder named after the
                     video will be created inside it.
        mtcnn: an initialized MTCNN detector instance.

    Returns:
        Dict summarizing how many frames were kept vs. discarded.
    """
    video_name = video_frames_dir.name
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(
        p for p in video_frames_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    kept = 0
    discarded = 0

    for frame_path in frame_paths:
        try:
            face_image = align_face_in_image(frame_path, mtcnn)
        except Exception as e:
            logger.warning(f"Error processing {frame_path.name}: {e}")
            discarded += 1
            continue

        if face_image is None:
            discarded += 1
            continue

        output_path = video_output_dir / frame_path.name
        face_image.save(output_path)
        kept += 1

    return {"video_name": video_name, "kept": kept, "discarded": discarded}


def main():
    parser = argparse.ArgumentParser(description="MTCNN face detection and alignment for deepfake detection pipeline.")
    parser.add_argument("--input", required=True, help="Path to the frames folder (output of extract_frames.py). "
                                                          "Should contain one subfolder per video.")
    parser.add_argument("--output_dir", required=True, help="Directory to save aligned face crops into.")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info(f"Using device: {device}")
    mtcnn = MTCNN(keep_all=True, device=device)

    video_folders = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not video_folders:
        raise ValueError(f"No video subfolders found in {input_dir}. "
                          f"Did you run extract_frames.py on this path first?")

    logger.info(f"Found {len(video_folders)} video folder(s) to process.")

    total_kept = 0
    total_discarded = 0

    progress_bar = tqdm(video_folders, desc="Aligning faces", unit="video")
    for video_folder in progress_bar:
        result = process_video_folder(video_folder, output_dir, mtcnn)
        total_kept += result["kept"]
        total_discarded += result["discarded"]
        progress_bar.set_postfix({"kept": total_kept, "discarded": total_discarded})

    logger.info(
        f"Done. Total faces kept: {total_kept}, total frames discarded (no valid face): {total_discarded}"
    )


if __name__ == "__main__":
    main()
