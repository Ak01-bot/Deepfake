import zipfile, shutil
from pathlib import Path

def fix_checkpoint(zip_path, output_path):
    zip_path, output_path = Path(zip_path), Path(output_path)
    try:
        zf = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile:
        shutil.copy(zip_path, output_path)
        print(f"Not a zip — copied directly -> {output_path}")
        return

    names = zf.namelist()
    pth_members = [n for n in names if n.endswith(".pth") and not n.endswith("/")]

    if len(pth_members) == 1:
        with zf.open(pth_members[0]) as src, open(output_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        print(f"Extracted nested .pth -> {output_path}")
        zf.close()
        return

    tmp = output_path.parent / (output_path.stem + "_tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    zf.extractall(tmp)
    zf.close()

    entries = list(tmp.iterdir())
    root = entries[0] if len(entries) == 1 and entries[0].is_dir() else tmp

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as out:
        for f in root.rglob("*"):
            if f.is_file():
                out.write(f, f"{output_path.stem}/{f.relative_to(root)}")
    shutil.rmtree(tmp)
    print(f"Re-packaged exploded checkpoint -> {output_path}")

fix_checkpoint("checkpoints/xception_best.pth.zip", "checkpoints/xception_best.pth")
fix_checkpoint("checkpoints/temporal_attention.pth.zip", "checkpoints/temporal_attention.pth")
