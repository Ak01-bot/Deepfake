"""
diagnose_xception_load.py

Standalone diagnostic — does NOT modify predict.py or any project file.
Checks how much of xception_best.pth actually loads into the current
timm "xception" model, given the exact loading pattern predict.py uses.

Run from data_pipeline/:
    python3 ../diagnose_xception_load.py
"""

import timm
import torch

CHECKPOINT_PATH = "../checkpoints/xception_best.pth"
print(f"timm version: {timm.__version__}")
print(f"torch version: {torch.__version__}")
print()

model = timm.create_model("xception", pretrained=False, num_classes=0)
model_keys = set(model.state_dict().keys())
model_total_params = sum(p.numel() for p in model.state_dict().values())

checkpoint_state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
# Handle the case where the checkpoint is wrapped (e.g. {"state_dict": ...})
if isinstance(checkpoint_state_dict, dict) and "state_dict" in checkpoint_state_dict:
    checkpoint_state_dict = checkpoint_state_dict["state_dict"]

checkpoint_keys = set(checkpoint_state_dict.keys())

result = model.load_state_dict(checkpoint_state_dict, strict=False)

matched_keys = model_keys & checkpoint_keys
matched_params = sum(model.state_dict()[k].numel() for k in matched_keys)

print("=" * 60)
print(f"Model expects {len(model_keys)} parameter tensors ({model_total_params:,} total values)")
print(f"Checkpoint contains {len(checkpoint_keys)} parameter tensors")
print(f"Matched (successfully loaded) keys: {len(matched_keys)}")
print(f"Missing keys (model wanted, checkpoint didn't have): {len(result.missing_keys)}")
print(f"Unexpected keys (checkpoint had, model didn't want): {len(result.unexpected_keys)}")
print(f"Fraction of model's parameters actually loaded from checkpoint: "
      f"{matched_params / model_total_params:.1%}")
print("=" * 60)

if result.missing_keys:
    print(f"\nFirst 10 missing keys:")
    for k in result.missing_keys[:10]:
        print(f"  {k}")

if result.unexpected_keys:
    print(f"\nFirst 10 unexpected keys (in checkpoint, not used):")
    for k in result.unexpected_keys[:10]:
        print(f"  {k}")

print()
if matched_params / model_total_params < 0.5:
    print("DIAGNOSIS: Less than half the network loaded. This is almost certainly")
    print("the bug — most of the model is running on random weights, not Charlie's")
    print("trained weights. Likely cause: timm version mismatch (see key name diff above).")
elif matched_params / model_total_params < 0.95:
    print("DIAGNOSIS: Most of the network loaded, but a meaningful chunk didn't.")
    print("Worth investigating the specific missing/unexpected keys above.")
else:
    print("DIAGNOSIS: Nearly everything loaded correctly — this checkpoint load")
    print("is NOT the source of the bug. Look elsewhere (see follow-up steps).")