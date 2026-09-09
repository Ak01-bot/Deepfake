"""
extract_embeddings_holdout.py

Extracts and caches embeddings from the faceshifter_holdout set
specifically — the manipulation technique held out from all training,
reserved for the Week 7 generalization test.

This is a separate script from extract_embeddings.py on purpose:
extract_embeddings.py deliberately EXCLUDES this folder (it loops
over CATEGORY_LABELS, which never includes faceshifter_holdout) so
there's no way to accidentally pull holdout data into training. This
script is the one and only place faceshifter_holdout data gets
processed, and it's used solely for the final generalization test.

Every video here is labeled fake (1) — faceshifter_holdout contains
only FaceShifter-manipulated videos, no real footage.

Usage:
    python extract_embeddings_holdout.py --holdout_root faces/faceshifter_holdout \\
        --checkpoint checkpoints/xception_best.pth --output_dir embeddings_holdout
"""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from tqdm import tqdm

from dataset import IMAGE_EXTENSIONS, get_eval_transform


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_extractor(checkpoint_path: str, backbone_name: str, device: torch.device):
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def extract_embeddings_for_video(video_dir: Path, model, transform, device: torch.device) -> np.ndarray:
    frame_paths = sorted(
        p for p in video_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frame_paths:
        return np.empty((0,))

    embeddings = []
    with torch.no_grad():
        for frame_path in frame_paths:
            image = Image.open(frame_path).convert("RGB")
            tensor = transform(image).unsqueeze(0).to(device)
            embedding = model(tensor).squeeze(0).cpu().numpy()
            embeddings.append(embedding)

    return np.stack(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from the faceshifter_holdout generalization test set.")
    parser.add_argument("--holdout_root", required=True, help="Path directly to the faceshifter_holdout folder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--backbone", default="xception")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    holdout_root = Path(args.holdout_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    model = load_feature_extractor(args.checkpoint, args.backbone, device)
    transform = get_eval_transform()

    video_dirs = sorted(p for p in holdout_root.iterdir() if p.is_dir())
    print(f"Found {len(video_dirs)} holdout videos to process.")
    print("Every video here is labeled FAKE — this is the FaceShifter technique, never used in training.")

    for video_dir in tqdm(video_dirs, desc="faceshifter_holdout", unit="video"):
        output_path = output_dir / f"{video_dir.name}.npy"
        if output_path.exists():
            continue
        embeddings = extract_embeddings_for_video(video_dir, model, transform, device)
        np.save(output_path, embeddings)

    print(f"\nDone. Holdout embeddings cached to: {output_dir}")


if __name__ == "__main__":
    main()
