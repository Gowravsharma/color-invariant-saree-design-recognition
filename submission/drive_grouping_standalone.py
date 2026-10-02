"""Run this YOURSELF (the agent does not execute this against real Drive
images, per method1.md rule 7). It is Step B pulled out of the main
notebook so you can do the hand-labeling locally, right now, before ever
touching Kaggle -- then bundle the finished drive_groups.csv together with
the images into one dataset upload.

Usage (from a terminal, in this project folder):
    python submission/drive_grouping_standalone.py

Output (written next to this script's working directory, in
"drive_grouping_output/"):
    drive_contact_sheets/cluster_000.png, cluster_001.png, ...
        -- open these yourself in File Explorer / any image viewer
    drive_groups_template.csv
        -- one row per Drive image, with an empty "group_id" column

What to do with the output:
    1. Open each cluster_XXX.png contact sheet and decide which photos in
       it (if any) show the SAME saree design (allowing for color shift,
       blur, rotation/flip, crop/continuation of the same repeating
       pattern -- see the labeling rules below).
    2. Open drive_groups_template.csv in Excel/Sheets and fill in a
       group_id (any consistent label, e.g. 1, 2, 3...) for every row.
       Rules (method1.md SS5):
         - Same pattern, color shift, blur, rotation/flip -> SAME group_id.
         - Different crop/continuation of the same repeating pattern ->
           SAME group_id (mark variant=crop if easy to tell).
         - Blank/unusable image -> is_blank=1.
         - Pattern appears only once -> its own unique group_id (singleton;
           stays in the gallery as a distractor, never becomes a query).
         - Unsure whether two images are the same design -> keep them in
           DIFFERENT group_ids (conservative default).
    3. Save the filled-in file as "drive_groups.csv".
    4. On kaggle.com, go to your private "handloom" dataset -> add
       drive_groups.csv into it -> save as a New Version. Now the images
       AND the finished labels are bundled together from the start, so a
       single Kaggle run will find everything immediately.
"""
import os
import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

DRIVE_DIR = Path(__file__).resolve().parents[1] / "handloom_sarees"
OUT_DIR = Path(__file__).resolve().parent / "drive_grouping_output"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_DINOV2_CACHE = {}
def get_frozen_dinov2():
    if "model" not in _DINOV2_CACHE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        _DINOV2_CACHE["model"] = m.eval().to(DEVICE)
    return _DINOV2_CACHE["model"]

def embed_with_pretrained_dinov2(paths, rotations=(0, 90, 180, 270)):
    backbone = get_frozen_dinov2()
    tfm_mean, tfm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    all_embeds = []
    with torch.no_grad():
        for p in paths:
            img = Image.open(p).convert("L").convert("RGB").resize((224, 224), Image.BILINEAR)
            rot_embeds = []
            for rot in rotations:
                r = img.rotate(rot) if rot else img
                arr = np.asarray(r, dtype=np.float32) / 255.0
                arr = (arr - np.array(tfm_mean)) / np.array(tfm_std)
                t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)
                feats = backbone.forward_features(t)
                emb = F.normalize(feats["x_norm_clstoken"], dim=1)
                rot_embeds.append(emb.cpu().numpy()[0])
            all_embeds.append(np.stack(rot_embeds))
    return np.stack(all_embeds)

def rotation_invariant_distance_matrix(embeds):
    n = embeds.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = (embeds[i] @ embeds[j].T).max()
    return 1.0 - sim

def save_contact_sheet(paths, out_path, thumb=96, cols=8):
    n = len(paths)
    rows = math.ceil(n / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (30, 30, 30))
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB").resize((thumb, thumb))
        except Exception:
            continue
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet.save(out_path)

def run_drive_grouping_helper(drive_dir: Path, out_dir: Path, distance_threshold=0.35):
    from sklearn.cluster import AgglomerativeClustering
    paths = sorted(drive_dir.glob("*.jpg"))
    print(f"embedding {len(paths)} Drive images with pretrained DINOv2 (no fine-tuning)...")
    embeds = embed_with_pretrained_dinov2(paths)
    dist = rotation_invariant_distance_matrix(embeds)
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        metric="precomputed", linkage="average",
    ).fit(dist)
    labels = clustering.labels_
    contact_dir = out_dir / "drive_contact_sheets"
    contact_dir.mkdir(exist_ok=True, parents=True)
    by_cluster = defaultdict(list)
    for p, lab in zip(paths, labels):
        by_cluster[int(lab)].append(p)
    for cid, cpaths in by_cluster.items():
        save_contact_sheet(cpaths, contact_dir / f"cluster_{cid:03d}.png")
    template_path = out_dir / "drive_groups_template.csv"
    with open(template_path, "w", encoding="utf-8") as f:
        f.write("filename,cluster_id,group_id,is_blank,variant\n")
        for p, lab in zip(paths, labels):
            f.write(f"{p.name},{lab},,,\n")
    print(f"wrote {len(by_cluster)} contact sheets to {contact_dir}")
    print(f"wrote template for hand-labeling: {template_path}")
    print("Open the contact sheets yourself, fill in group_id/is_blank/variant by hand, "
          "save as drive_groups.csv.")

if __name__ == "__main__":
    if not DRIVE_DIR.is_dir():
        print(f"ERROR: {DRIVE_DIR} not found.")
        sys.exit(1)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    run_drive_grouping_helper(DRIVE_DIR, OUT_DIR)
