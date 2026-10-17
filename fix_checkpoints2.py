import zipfile, shutil, tempfile
from pathlib import Path

def fix_checkpoint(zip_path, output_path):
    zip_path, output_path = Path(zip_path), Path(output_path)
    if not zip_path.exists():
        raise FileNotFoundError(f"Source zip not found: {zip_path}")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        pth_members = [n for n in names if n.endswith(".pth") and not n.endswith("/")]

        if len(pth_members) == 1:
            with zf.open(pth_members[0]) as src, open(output_path, "wb") as dst:
                shutil.copyfileobj(src, dst)
            print(f"[OK] Extracted nested .pth -> {output_path}")
            return

        tmp = Path(tempfile.mkdtemp(prefix="ckpt_fix_"))
        zf.extractall(tmp)

    entries = list(tmp.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp

    if output_path.exists():
        output_path.unlink()
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as out:
        for f in root.rglob("*"):
            if f.is_file():
                out.write(f, f"{output_path.stem}/{f.relative_to(root)}")
    shutil.rmtree(tmp)
    print(f"[OK] Re-packaged exploded checkpoint -> {output_path}")

fix_checkpoint("checkpoints/xception_best.pth.zip", "checkpoints/xception_best.pth")
fix_checkpoint("checkpoints/temporal_attention.pth.zip", "checkpoints/temporal_attention.pth")

Path("checkpoints/xception_best.pth.zip").unlink()
Path("checkpoints/temporal_attention.pth.zip").unlink()

contents = sorted(p.name for p in Path("checkpoints").iterdir())
print("Final checkpoints/ contents:", contents)
assert contents == ["temporal_attention.pth", "xception_best.pth"], f"UNEXPECTED CONTENTS: {contents}"
print("checkpoints/ is clean — exactly 2 files.")
