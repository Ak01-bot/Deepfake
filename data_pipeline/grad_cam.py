"""
grad_cam.py

Grad-CAM (Gradient-weighted Class Activation Mapping) for the trained
Xception classifier — generates a heatmap over an input face image
showing which regions most influenced the real/fake prediction.

Why this applies to the CNN stage specifically, not the full
CNN+LSTM+RNN pipeline: Grad-CAM works by looking at gradients flowing
into a convolutional layer's spatial feature maps — it needs that
spatial (height x width) structure to produce a heatmap. LSTM/RNN
layers operate on flat feature vectors with no spatial structure, so
Grad-CAM doesn't naturally extend past the CNN. This is standard
practice: Grad-CAM explains what the CNN "sees" in a given frame, the
same explainability layer used in the source project's own framing
("a visual indication of which frames or facial regions most
influenced the decision").

This uses the Xception checkpoint's own classification head (trained
with num_classes=2), not the stripped feature-extractor version used
for embedding caching — Grad-CAM needs real class logits to compute
gradients against.

Usage:
    python grad_cam.py --image path/to/face.jpg --checkpoint checkpoints/xception_best.pth --output heatmap.jpg
"""

import argparse

import cv2
import numpy as np
import timm
import torch
import torch.nn.functional as F
from PIL import Image

from dataset import get_eval_transform


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """
    Automatically locate the last convolutional layer in the model —
    this is the layer Grad-CAM hooks into, since it holds the last set
    of spatially-meaningful feature maps before the model flattens
    everything down for classification.
    """
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found in this model — is this really a CNN backbone?")
    return last_conv


class GradCAM:
    """
    Standard Grad-CAM: hooks a target conv layer to capture its
    activations (forward pass) and gradients (backward pass), then
    combines them into a class-discriminative heatmap.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, target_class: int = None) -> np.ndarray:
        """
        Args:
            input_tensor: preprocessed image tensor, shape (1, 3, H, W).
            target_class: which class's score to explain. If None,
                            uses whichever class the model predicted
                            (i.e. explains the model's own decision).

        Returns:
            2D numpy array (values 0-1), the same spatial size as the
            hooked layer's feature maps — the caller resizes this to
            the original image size for overlay.
        """
        self.model.zero_grad()
        output = self.model(input_tensor)  # (1, num_classes)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        class_score = output[0, target_class]
        class_score.backward()

        # Global-average-pool the gradients per channel -> importance weight per channel
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation channels, then ReLU (Grad-CAM only cares
        # about features that positively influenced the target class)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, h, w)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam = cam / cam.max()  # normalize to 0-1

        return cam, target_class


def overlay_heatmap(original_image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Resizes the Grad-CAM heatmap to match the original image and
    blends it on top using a standard "jet" colormap (blue = low
    influence, red = high influence — the conventional Grad-CAM look).
    """
    original_np = np.array(original_image.convert("RGB"))
    height, width = original_np.shape[:2]

    cam_resized = cv2.resize(cam, (width, height))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap + (1 - alpha) * original_np).astype(np.uint8)
    return overlay


def run_grad_cam(image_path: str, checkpoint_path: str, output_path: str, backbone: str = "xception"):
    device = get_device()

    model = timm.create_model(backbone, pretrained=False, num_classes=2)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = find_last_conv_layer(model)
    grad_cam = GradCAM(model, target_layer)

    original_image = Image.open(image_path).convert("RGB")
    transform = get_eval_transform()
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    input_tensor.requires_grad = False  # gradients flow into activations via the hook, not the input itself

    cam, predicted_class = grad_cam.generate(input_tensor)

    class_names = {0: "real", 1: "fake"}
    print(f"Predicted class: {class_names.get(predicted_class, predicted_class)}")

    overlay = overlay_heatmap(original_image, cam)
    Image.fromarray(overlay).save(output_path)
    print(f"Grad-CAM heatmap saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate a Grad-CAM heatmap for a single face image.")
    parser.add_argument("--image", required=True, help="Path to a single aligned face image.")
    parser.add_argument("--checkpoint", required=True, help="Path to the trained CNN classifier checkpoint.")
    parser.add_argument("--backbone", default="xception")
    parser.add_argument("--output", required=True, help="Where to save the heatmap overlay image.")
    args = parser.parse_args()

    run_grad_cam(args.image, args.checkpoint, args.output, args.backbone)


if __name__ == "__main__":
    main()
