"""
dataset.py

PyTorch Dataset for the deepfake detection CNN, plus train/val/test
split logic over the aligned face crops produced by mtcnn_align.py.

Expected folder layout (one folder per category, as produced by the
pipeline so far):

    faces/
        original/<video_id>/frame_00000.jpg, ...
        deepfakes/<video_id>/frame_00000.jpg, ...
        face2face/<video_id>/...
        faceswap/<video_id>/...
        neuraltextures/<video_id>/...
        faceshifter_holdout/<video_id>/...   <- NOT included here on purpose,
                                                  this is the Week 7 generalization
                                                  holdout and must never appear
                                                  in train/val/test splits.

Labels: "original" -> 0 (real), every manipulated category -> 1 (fake).

Splitting is done at the VIDEO level, not the frame level. Frames from
the same video are highly correlated (near-duplicate faces a fraction
of a second apart), so splitting at the frame level would leak near-
identical images across train/val/test and inflate validation/test
accuracy artificially. Splitting by video_id avoids that.
"""

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Category -> label. 0 = real, 1 = fake.
# faceshifter_holdout is intentionally excluded — it must never be split
# into train/val/test, it is reserved entirely for the Week 7
# generalization test on an unseen manipulation technique.
CATEGORY_LABELS = {
    "original": 0,
    "deepfakes": 1,
    "face2face": 1,
    "faceswap": 1,
    "neuraltextures": 1,
}

# Standard ImageNet normalization stats, since all three backbones
# (Xception/EfficientNet/ResNet) were pretrained on ImageNet.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transform():
    """Light augmentation for training: helps the model generalize
    rather than memorize specific frames.

    NOTE: a stronger augmentation variant (RandomResizedCrop, wider
    rotation, stronger color jitter, RandomErasing) was tested as
    polishing Step 3 and found to REDUCE downstream temporal model
    accuracy (test Acc 88.44% -> 86.22%, F1 93.09% -> 91.55%, AUC
    94.62% -> 92.00%). Reverted back to this lighter version, which
    produced the project's best result to date.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transform():
    """No augmentation for validation/test — we want a stable,
    reproducible evaluation, not randomized inputs."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class FaceDataset(Dataset):
    """
    A simple (image_path, label) dataset over aligned face crops.
    """

    def __init__(self, samples: list, transform=None):
        """
        Args:
            samples: list of (image_path: Path, label: int) tuples.
            transform: torchvision transform to apply to each image.
        """
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _collect_video_ids(category_dir: Path) -> list:
    """Return every video-id subfolder inside one category's face folder."""
    if not category_dir.exists():
        return []
    return sorted(p.name for p in category_dir.iterdir() if p.is_dir())


def _frames_for_video(category_dir: Path, video_id: str, label: int) -> list:
    """Return all (frame_path, label) pairs for one video."""
    video_dir = category_dir / video_id
    return [
        (p, label) for p in sorted(video_dir.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def build_dataset_splits(
    faces_root: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """
    Build train/val/test sample lists by splitting VIDEOS (not frames)
    within each category, then collecting all frames for the videos
    assigned to each split.

    Args:
        faces_root: path to the "faces" folder containing the category
                     subfolders (output of mtcnn_align.py). Should NOT
                     point at faceshifter_holdout — that folder is
                     handled separately, see module docstring.
        val_ratio: fraction of videos (per category) assigned to validation.
        test_ratio: fraction of videos (per category) assigned to test.
        seed: random seed, for reproducible splits across runs.

    Returns:
        (train_samples, val_samples, test_samples) — each a list of
        (image_path, label) tuples.
    """
    faces_root = Path(faces_root)
    rng = random.Random(seed)

    train_samples, val_samples, test_samples = [], [], []

    for category, label in CATEGORY_LABELS.items():
        category_dir = faces_root / category
        video_ids = _collect_video_ids(category_dir)

        if not video_ids:
            print(f"Warning: no videos found for category '{category}' at {category_dir}")
            continue

        rng.shuffle(video_ids)

        n_total = len(video_ids)
        n_val = max(1, int(n_total * val_ratio))
        n_test = max(1, int(n_total * test_ratio))
        n_train = n_total - n_val - n_test

        train_ids = video_ids[:n_train]
        val_ids = video_ids[n_train:n_train + n_val]
        test_ids = video_ids[n_train + n_val:]

        for video_id in train_ids:
            train_samples.extend(_frames_for_video(category_dir, video_id, label))
        for video_id in val_ids:
            val_samples.extend(_frames_for_video(category_dir, video_id, label))
        for video_id in test_ids:
            test_samples.extend(_frames_for_video(category_dir, video_id, label))

        print(f"{category}: {n_train} train videos, {len(val_ids)} val videos, "
              f"{len(test_ids)} test videos")

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    rng.shuffle(test_samples)

    print(f"\nTotal frames -> train: {len(train_samples)}, "
          f"val: {len(val_samples)}, test: {len(test_samples)}")

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    # Quick sanity check: run this directly to confirm splits build
    # correctly before wiring it into the training loop.
    import sys

    faces_root = sys.argv[1] if len(sys.argv) > 1 else "faces"
    train_samples, val_samples, test_samples = build_dataset_splits(faces_root)

    if train_samples:
        train_ds = FaceDataset(train_samples, transform=get_train_transform())
        image, label = train_ds[0]
        print(f"\nSample check -> image shape: {tuple(image.shape)}, label: {label}")