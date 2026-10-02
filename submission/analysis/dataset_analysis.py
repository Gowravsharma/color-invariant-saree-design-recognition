"""One-off dataset analysis: class balance, blur, duplicate/near-duplicate
sanity, and basic image statistics across both source datasets. Not part of
the training pipeline -- run once to inform the checkpoint questions in
need_to_look.txt (imbalance, hard images, blur/preprocessing)."""
import os, re, json
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

ROOT = Path(__file__).resolve().parents[2]
HANDLOOM_DIR = ROOT / "handloom_sarees"
FABRIC_DIR = ROOT / "kaggle"
RF_SUFFIX_RE = re.compile(r"_jpg\.rf\.[0-9a-f]+\.jpg$", re.IGNORECASE)


def index_kaggle_fabric(root: Path):
    groups = defaultdict(list)
    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.glob("*.jpg")):
                base = RF_SUFFIX_RE.sub("", f.name)
                groups[(cls_dir.name, base)].append(f)
    records = []
    for (cls, base), paths in groups.items():
        records.append(dict(identity=f"kaggle::{cls}::{base}", cls=cls,
                             source="kaggle", paths=sorted(paths)))
    return records


def index_handloom(root: Path):
    records = []
    for f in sorted(root.glob("*.jpg")):
        records.append(dict(identity=f"handloom::{f.stem}", cls="unlabeled",
                             source="handloom", paths=[f]))
    return records


def blur_score(path, max_side=512):
    img = Image.open(path).convert("L")
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float64)
    lap = laplace(arr)
    return float(lap.var())


def size_stats(path):
    with Image.open(path) as im:
        return im.size  # (w, h)


def main():
    records = index_kaggle_fabric(FABRIC_DIR) + index_handloom(HANDLOOM_DIR)
    print(f"total unique identities: {len(records)}")

    by_cls = defaultdict(int)
    for r in records:
        by_cls[r["cls"]] += 1
    print("class balance:", dict(by_cls))
    counts = list(by_cls.values())
    print(f"imbalance ratio (max/min class count): {max(counts)/min(counts):.2f}")

    rows = []
    for r in records:
        p = r["paths"][0]
        try:
            w, h = size_stats(p)
            b = blur_score(p)
        except Exception as e:
            print("FAILED", p, e)
            continue
        rows.append(dict(identity=r["identity"], cls=r["cls"], source=r["source"],
                          path=str(p), width=w, height=h, blur_var=b))

    blur_vals = np.array([r["blur_var"] for r in rows])
    print(f"\nblur (Laplacian variance) stats over {len(rows)} images:")
    print(f"  min={blur_vals.min():.1f}  p5={np.percentile(blur_vals,5):.1f}  "
          f"p10={np.percentile(blur_vals,10):.1f}  median={np.median(blur_vals):.1f}  "
          f"p90={np.percentile(blur_vals,90):.1f}  max={blur_vals.max():.1f}")

    rows_sorted = sorted(rows, key=lambda r: r["blur_var"])
    print("\n20 blurriest images:")
    for r in rows_sorted[:20]:
        print(f"  {r['blur_var']:8.1f}  {r['source']:9s} {r['cls']:10s} {r['path']}")

    print("\n10 sharpest images (for contrast):")
    for r in rows_sorted[-10:]:
        print(f"  {r['blur_var']:8.1f}  {r['source']:9s} {r['cls']:10s} {r['path']}")

    p10 = np.percentile(blur_vals, 10)
    by_cls_blurry = defaultdict(int)
    for r in rows:
        if r["blur_var"] <= p10:
            by_cls_blurry[r["cls"]] += 1
    print(f"\nbottom-10%-blur count by class (n={sum(by_cls_blurry.values())}):", dict(by_cls_blurry))

    sizes = np.array([(r["width"], r["height"]) for r in rows])
    print(f"\nimage size stats: width min/med/max = {sizes[:,0].min()}/{int(np.median(sizes[:,0]))}/{sizes[:,0].max()}"
          f"  height min/med/max = {sizes[:,1].min()}/{int(np.median(sizes[:,1]))}/{sizes[:,1].max()}")

    out_path = Path(__file__).parent / "blur_report.json"
    with open(out_path, "w") as f:
        json.dump(rows_sorted, f, indent=2)
    print(f"\nfull per-image report written to {out_path}")


if __name__ == "__main__":
    main()
