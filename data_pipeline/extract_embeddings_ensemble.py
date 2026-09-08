"""
extract_embeddings_ensemble.py

Same purpose as extract_embeddings.py, but extracts embeddings from
TWO trained CNN backbones (Xception + ResNet50 — the top two from the
architecture comparison) per frame, and concatenates them into one
richer feature vector per frame.

Why this might help: Xception and ResNet50 have different internal
architectures (Xception uses depthwise separable convolutions, ResNet
uses residual connections), so they can learn to notice somewhat
different visual cues even when trained on the same task. Combining
both feature sets gives the LSTM/RNN branches more information to
work with than either backbone alone.

Output format: same as extract_embeddings.py — one .npy file per
video, but each frame's embedding is now (xception_dim + resnet_dim)
long instead of just xception_dim. The downstream train_temporal.py /
evaluate_temporal.py scripts need NO changes to use this — they
auto-detect embedding_dim from the cached file shape.

Usage:
    python extract_embeddings_ensemble.py --faces_root faces \\
        --xception_checkpoint checkpoints/xception_best.pth \\
        --resnet_checkpoint checkpoints/resnet50_best.pth \\
        --output_dir embeddings_ensemble
"""

import argparse
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from tqdm import tqdm

from dataset import CATEGORY_LABELS, IMAGE_EXTENSIONS, get_eval_transform


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_feature_extractor(checkpoint_path: str, backbone_name: str, device: torch.device):
    """Load a trained backbone and strip its classification head, same as extract_embeddings.py."""
    model = timm.create_model(backbone_name, pretrained=False, num_classes=0)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def extract_ensemble_embeddings_for_video(
    video_dir: Path, model_a, model_b, transform, device: torch.device
) -> np.ndarray:
    """
    Run every frame through BOTH backbones and concatenate their
    outputs per frame, producing a (num_frames, dim_a + dim_b) array.
    """
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

            embedding_a = model_a(tensor).squeeze(0).cpu().numpy()
            embedding_b = model_b(tensor).squeeze(0).cpu().numpy()

            combined = np.concatenate([embedding_a, embedding_b])
            embeddings.append(combined)

    return np.stack(embeddings)


def main():
    parser = argparse.ArgumentParser(description="Extract and cache ENSEMBLE (Xception+ResNet) embeddings.")
    parser.add_argument("--faces_root", required=True)
    parser.add_argument("--xception_checkpoint", required=True)
    parser.add_argument("--resnet_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    faces_root = Path(args.faces_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    print("Loading Xception feature extractor...")
    model_xception = load_feature_extractor(args.xception_checkpoint, "xception", device)

    print("Loading ResNet50 feature extractor...")
    model_resnet = load_feature_extractor(args.resnet_checkpoint, "resnet50", device)

    transform = get_eval_transform()

    for category in CATEGORY_LABELS:
        category_dir = faces_root / category
        if not category_dir.exists():
            print(f"Warning: category folder not found, skipping: {category_dir}")
            continue

        category_output_dir = output_dir / category
        category_output_dir.mkdir(parents=True, exist_ok=True)

        video_dirs = sorted(p for p in category_dir.iterdir() if p.is_dir())
        print(f"\n{category}: extracting ensemble embeddings for {len(video_dirs)} videos...")

        for video_dir in tqdm(video_dirs, desc=f"  {category}", unit="video"):
            output_path = category_output_dir / f"{video_dir.name}.npy"
            if output_path.exists():
                continue

            embeddings = extract_ensemble_embeddings_for_video(
                video_dir, model_xception, model_resnet, transform, device
            )
            np.save(output_path, embeddings)

    print("\nDone. Ensemble embeddings cached to:", output_dir)


if __name__ == "__main__":
    main()
