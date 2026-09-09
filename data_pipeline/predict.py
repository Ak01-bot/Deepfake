"""
predict.py

End-to-end inference: given a raw video file, runs the FULL pipeline
(frame extraction -> MTCNN face alignment -> Xception embedding
extraction -> LSTM+RNN attention fusion model) and returns a single
real/fake verdict with a confidence score.

This is the function the Streamlit dashboard should call — it hides
all the pipeline plumbing (temp folders, frame counts, model loading)
behind one simple interface: give it a video path, get back a verdict.

Usage (command line, for testing):
    python predict.py --video path/to/video.mp4 \\
        --xception_checkpoint checkpoints/xception_best.pth \\
        --temporal_checkpoint checkpoints/temporal_attention.pth

Usage (importing into Streamlit or other Python code):
    from predict import predict_video

    result = predict_video(
        video_path="uploaded_video.mp4",
        xception_checkpoint="checkpoints/xception_best.pth",
        temporal_checkpoint="checkpoints/temporal_attention.pth",
    )
    print(result)
    # {
    #     "verdict": "fake",
    #     "confidence": 0.94,
    #     "real_probability": 0.06,
    #     "fake_probability": 0.94,
    #     "num_frames_used": 18,
    #     "num_frames_discarded": 1,
    # }
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from dataset import get_eval_transform
from extract_frames import collect_video_paths, extract_frames_from_video
from mtcnn_align import get_device as get_mtcnn_device
from mtcnn_align import align_face_in_image
from facenet_pytorch import MTCNN
from temporal_model import TemporalFusionAttentionModel

# Must match training: frame sampling rate used for extract_frames.py
DEFAULT_FPS = 1.0


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_xception_feature_extractor(checkpoint_path: str, device: torch.device):
    """Same pattern as extract_embeddings.py: load the trained CNN, strip its classifier head."""
    model = timm.create_model("xception", pretrained=False, num_classes=0)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


def load_temporal_model(checkpoint_path: str, embedding_dim: int, device: torch.device):
    """
    Loads the temporal fusion model with ATTENTION pooling — this must
    match how temporal_attention.pth was actually trained
    (train_temporal.py --use_attention). If you ever swap in a
    non-attention checkpoint here, use TemporalFusionModel instead.
    """
    model = TemporalFusionAttentionModel(
        embedding_dim=embedding_dim,
        lstm_hidden_size=256,
        lstm_num_layers=2,
        rnn_hidden_size=256,
        rnn_num_layers=1,
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model


def predict_video(
    video_path: str,
    xception_checkpoint: str,
    temporal_checkpoint: str,
    fps: float = DEFAULT_FPS,
) -> dict:
    """
    Run the full pipeline on a single video and return a verdict.

    Args:
        video_path: path to the input video file.
        xception_checkpoint: path to the trained Xception .pth file.
        temporal_checkpoint: path to the trained temporal_attention.pth file.
        fps: frame sampling rate — should match what the model was
              trained on (1.0, per the project's confirmed-best setting).

    Returns:
        Dict with "verdict" ("real" or "fake"), "confidence" (0-1,
        confidence in the returned verdict), "real_probability",
        "fake_probability", and frame counts for transparency.

    Raises:
        ValueError if no face could be detected in ANY frame of the video
        (nothing to classify) — the caller (e.g. Streamlit) should catch
        this and show a clear "no face detected" message to the user.
    """
    device = get_device()
    temp_dir = Path(tempfile.mkdtemp(prefix="deepfake_predict_"))

    try:
        # --- Step 1: extract frames ---
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        video_paths = collect_video_paths(Path(video_path))
        total_frames = 0
        for vp in video_paths:  # normally just one video
            total_frames += extract_frames_from_video(vp, frames_dir, fps)

        if total_frames == 0:
            raise ValueError("Could not extract any frames from this video — is the file valid?")

        # extract_frames_from_video writes into frames_dir/<video_stem>/
        video_frame_dir = frames_dir / Path(video_path).stem
        frame_paths = sorted(video_frame_dir.iterdir())

        # --- Step 2: MTCNN face detection + alignment ---
        mtcnn_device = get_mtcnn_device()
        mtcnn = MTCNN(keep_all=True, device=mtcnn_device)

        aligned_faces = []
        discarded = 0
        aligned_faces_dir = temp_dir / "aligned_faces"
        aligned_faces_dir.mkdir()

        for frame_path in frame_paths:
            face_image = align_face_in_image(frame_path, mtcnn)
            if face_image is None:
                discarded += 1
                continue

            # IMPORTANT: save to JPEG and reload, rather than using the
            # in-memory PIL image directly. This matches the training
            # pipeline exactly — mtcnn_align.py saves aligned faces to
            # disk as JPEG, and the training Dataset loads them back
            # from disk, so every training image went through one JPEG
            # compression pass. Skipping this step here caused a real
            # train/inference mismatch (JPEG compression alters pixel
            # values slightly), which was traced to a live video
            # ("original/437") being misclassified by predict.py despite
            # the exact same frames, loaded from the JPEG files the
            # training pipeline actually produced, classifying correctly.
            temp_face_path = aligned_faces_dir / frame_path.name
            face_image.save(temp_face_path)
            face_image = Image.open(temp_face_path).convert("RGB")

            aligned_faces.append(face_image)

        if not aligned_faces:
            raise ValueError(
                "No face could be detected in any frame of this video. "
                "Cannot classify — please try a different video with a clearer, front-facing subject."
            )

        # --- Step 3: Xception embeddings ---
        xception_model = load_xception_feature_extractor(xception_checkpoint, device)
        transform = get_eval_transform()

        embeddings = []
        with torch.no_grad():
            for face_image in aligned_faces:
                tensor = transform(face_image).unsqueeze(0).to(device)
                embedding = xception_model(tensor).squeeze(0).cpu().numpy()
                embeddings.append(embedding)

        embedding_sequence = np.stack(embeddings)  # (num_frames, embedding_dim)
        embedding_dim = embedding_sequence.shape[1]

        # --- Step 4: temporal fusion model ---
        temporal_model = load_temporal_model(temporal_checkpoint, embedding_dim, device)

        sequence_tensor = torch.from_numpy(embedding_sequence).float().unsqueeze(0).to(device)  # (1, seq_len, dim)
        seq_len_tensor = torch.tensor([embedding_sequence.shape[0]])

        with torch.no_grad():
            logits = temporal_model(sequence_tensor, seq_len_tensor)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        real_probability = float(probabilities[0])
        fake_probability = float(probabilities[1])
        verdict = "fake" if fake_probability > real_probability else "real"
        confidence = max(real_probability, fake_probability)

        return {
            "verdict": verdict,
            "confidence": round(confidence, 4),
            "real_probability": round(real_probability, 4),
            "fake_probability": round(fake_probability, 4),
            "num_frames_used": len(aligned_faces),
            "num_frames_discarded": discarded,
        }

    finally:
        # Always clean up the temp directory, even if something raised an error above
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run the full deepfake detection pipeline on a single video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--xception_checkpoint", required=True)
    parser.add_argument("--temporal_checkpoint", required=True)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    args = parser.parse_args()

    result = predict_video(
        video_path=args.video,
        xception_checkpoint=args.xception_checkpoint,
        temporal_checkpoint=args.temporal_checkpoint,
        fps=args.fps,
    )

    print("\n" + "=" * 40)
    print(f"Verdict:              {result['verdict'].upper()}")
    print(f"Confidence:           {result['confidence']:.2%}")
    print(f"Real probability:     {result['real_probability']:.2%}")
    print(f"Fake probability:     {result['fake_probability']:.2%}")
    print(f"Frames used:          {result['num_frames_used']}")
    print(f"Frames discarded:     {result['num_frames_discarded']}")
    print("=" * 40)


if __name__ == "__main__":
    main()
