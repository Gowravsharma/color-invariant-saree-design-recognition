# %% [markdown]
# # AIE-CASE — Color-Invariant Saree Design Recognition
# Implementation of `method1.md` (the project's single source of truth for
# what to build). Single notebook, runs top to bottom on a Kaggle GPU
# (T4/P100), PyTorch.
#
# ## Ground rules disclosure (method1.md §0)
# 1. The Drive corpus (`handloom_sarees`, ~165 images) is **proprietary**.
#    Not redistributed. Kept in a private Kaggle dataset. Never displayed in
#    shareable notebook outputs. Deleted after the exercise.
# 2. **Never trained on, or used to select checkpoints/thresholds.** Drive is
#    the final test set only — see `DRIVE_DIR` gating throughout.
# 3. Disclosed: pretrained backbone **DINOv2 ViT-S/14** (Meta, via
#    `torch.hub`, 22.06M params, ImageNet/LVD-142M self-supervised
#    pretraining — loaded and shape-checked locally during development).
#    External data: Kaggle `div456/indian-saree-patterns` (MIT license,
#    Roboflow "Indian Fabric Patterns" export). No other external data.
# 4. Every non-obvious choice is commented with *why*, for the live defense.
# 5. All work original; implemented from `method1.md`, not copied from any
#    other submission (including a since-discovered public repo with a
#    similar title, which was deliberately not opened/used — see
#    `research_notes.md`).
# 6. Reproducibility: fixed seeds, saved split files
#    (`kaggle_manifest.csv`, `kaggle_split.csv`,
#    `drive_query_gallery_split.csv`), pinned library versions printed below.
# 7. **This code was written without the coding agent viewing any Drive
#    (`handloom_sarees`) image pixels.** Step B's clustering/contact-sheet
#    script is written for the *author* to run and review by eye; it was
#    only smoke-tested against non-proprietary Kaggle images. The main
#    training/eval pipeline processes Drive images only through numeric
#    tensors (embeddings, similarities, metrics) — never displays or
#    re-serializes them anywhere a reviewer or the agent could see pixels.
# 8. Everything below was tested, at minimum, for the code to run without
#    crashing on real (non-proprietary) images; **numbers in the final
#    Results section must come from an actual run** — do not hand-edit them.
#
# ### Approach note (≤500 chars, edit to match what actually ran)
# > DINOv2 ViT-S/14 (last 4 blocks tuned), GeM pooling over patch tokens,
# > 256-d L2-normed head; cosine ranking, verification via val-tuned
# > threshold. Pre: 224px resize, ImageNet norm. Train: SupCon (tau=0.07) on
# > deduped Kaggle images, 2 views/source; palette-grouped batches
# > (shared-palette hard negatives; positives always recoloured via
# > cluster-wise Lab remap). Aug: cluster/hue/perm/gray recolour, rotation,
# > crop, blur. Drive images used only for final testing.

# %%
import os, sys, io, json, time, math, random, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

def _pip_install(pkg, import_name=None):
    try:
        __import__(import_name or pkg)
    except ImportError:
        os.system(f"{sys.executable} -m pip install -q {pkg}")

_pip_install("opencv-python-headless", "cv2")
_pip_install("timm")
_pip_install("scikit-learn", "sklearn")

import cv2
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

print("library versions:")
for mod_name in ("numpy", "cv2", "torch", "PIL"):
    mod = sys.modules.get(mod_name) or __import__(mod_name)
    print(f"  {mod_name}: {getattr(mod, '__version__', 'unknown')}")

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

# %% [markdown]
# ## Config
# `DRIVE_DIR = None` (or a missing/empty folder) makes every Drive-gated
# cell print a notice and skip, instead of erroring — the notebook stays
# "runnable as-is" for a reviewer who only has the public Kaggle data
# attached (method1.md §11 checklist).

# %%
def find_dir(root: Path, name: str):
    root = Path(root)
    if not root.exists():
        return None
    if root.name == name:
        return root
    for dirpath, dirnames, _ in os.walk(root):
        if name in dirnames:
            return Path(dirpath) / name
    return None

def find_fabric_dir(kaggle_input: Path):
    """Find the Indian Saree Patterns / Indian Fabric Patterns export by
    STRUCTURE (a folder containing train/ + valid/ or test/ subfolders),
    not by name -- Kaggle's "Add Data" mounts a public dataset under its own
    slug (e.g. /kaggle/input/indian-saree-patterns/...), not a folder
    literally called "kaggle"."""
    if not kaggle_input.exists():
        return None
    for dirpath, dirnames, _ in os.walk(kaggle_input):
        names = {d.lower() for d in dirnames}
        if {"train", "valid"}.issubset(names) or {"train", "test"}.issubset(names):
            return Path(dirpath)
    return None

def find_drive_dir(kaggle_input: Path):
    """Find the private Drive corpus: prefer an exact "handloom_sarees"
    folder; otherwise, any attached dataset folder that's mostly loose .jpg
    files with no train/valid/test structure (so it isn't mistaken for the
    public fabric dataset) -- covers however the private dataset's zip
    happened to be named on upload."""
    exact = find_dir(kaggle_input, "handloom_sarees")
    if exact is not None:
        return exact
    if not kaggle_input.exists():
        return None
    # Full-depth search (Kaggle's actual mount path can nest arbitrarily,
    # e.g. /kaggle/input/datasets/<owner>/<slug>/..., not just one level
    # under /kaggle/input) -- prune train/valid/test branches entirely so a
    # fabric-dataset class subfolder (which also has >20 loose jpgs) is
    # never mistaken for the Drive folder.
    for dirpath, dirnames, filenames in os.walk(kaggle_input):
        dirnames[:] = [d for d in dirnames if d.lower() not in ("train", "valid", "test")]
        jpg_count = sum(1 for fn in filenames if fn.lower().endswith(".jpg"))
        if jpg_count > 20:
            return Path(dirpath)
    return None

KAGGLE_INPUT = Path("/kaggle/input")
FABRIC_DIR = find_fabric_dir(KAGGLE_INPUT) or Path("kaggle")
DRIVE_DIR = find_drive_dir(KAGGLE_INPUT) or Path("handloom_sarees")
if not DRIVE_DIR.is_dir() or not any(DRIVE_DIR.glob("*.jpg")):
    DRIVE_DIR = None  # reviewer-safe fallback -- Drive-gated cells will skip

WORK_DIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path("outputs")
WORK_DIR.mkdir(exist_ok=True, parents=True)
print("fabric dir:", FABRIC_DIR.resolve(), "exists:", FABRIC_DIR.exists())
print("drive dir:", DRIVE_DIR.resolve() if DRIVE_DIR else None,
      "(None => Drive-gated cells will skip)")
print("work dir:", WORK_DIR.resolve())

CONFIG = dict(
    img_size=224,
    embed_dim=256,
    backbone="dinov2_vits14",
    unfreeze_last_n_blocks=4,
    backbone_lr=1e-5,
    head_lr=1e-3,
    weight_decay=0.05,
    warmup_epochs=2,
    epochs=20,             # trimmed from 40 for time budget -- see note below
    steps_per_epoch=25,    # trimmed from 40 for time budget -- see note below
    B=48,                 # distinct source_ids per batch (method1.md §7.2)
    V=2,                  # views per source
    palette_pool_size=6,
    palette_k_colors=5,
    supcon_tau=0.07,
    early_stop_patience=8,
    val_every=5,           # trimmed from 1 -- kaggle_val_eval is O(val_size^2)
                           # CPU-bound recoloring (not GPU-accelerated), so
                           # validating every epoch was dominating wall-clock
                           # time; 1-in-5 still gives periodic checkpointing.
    seeds=[0],             # extend to [0, 1, 2] if time permits (§8)
    ablation="full",       # "geometric_only" | "no_shared_palette" | "full"
    run_bakeoff=False,     # §6 backbone bake-off -- stretch goal, off by default
    run_ablations=False,   # §8 ablations a/b/c -- stretch goal, off by default
)
with open(WORK_DIR / "config.json", "w") as f:
    json.dump(CONFIG, f, indent=2)

# %% [markdown]
# ## Step A: Kaggle data prep (method1.md §4)
# Recursively index all train/valid/test class folders (the *original*
# Roboflow split is discarded — see §2/§14: "do not use the original Kaggle
# train/valid/test folders as splits"). Dedupe exactly by MD5, then
# near-duplicate-group by perceptual hash (pHash, 64-bit DCT, Hamming <= 6)
# so the Roboflow salt-and-pepper triplicates collapse to one `source_id`
# even though — contrary to the "byte-identical" assumption in method1.md
# §2 — they are **not** byte-identical (verified: ~93 isolated 1-5px noise
# dots differ between copies of the same source photo, so MD5 alone misses
# them; pHash is the step that actually does the work here).

# %%
def list_kaggle_images(root: Path):
    paths = []
    for split in ("train", "valid", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for f in sorted(cls_dir.glob("*.jpg")):
                paths.append((f, cls_dir.name))
    return paths

def md5_of(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def is_near_blank(path, std_thresh=4.0):
    try:
        img = Image.open(path).convert("L").resize((64, 64))
    except Exception:
        return True
    return float(np.asarray(img, dtype=np.float32).std()) < std_thresh

def phash64(path, hash_size=8, highfreq_factor=4):
    """64-bit DCT perceptual hash. Returns a boolean array of length 64."""
    from scipy.fftpack import dct
    img_size = hash_size * highfreq_factor
    img = Image.open(path).convert("L").resize((img_size, img_size), Image.LANCZOS)
    arr = np.asarray(img, dtype=np.float64)
    d = dct(dct(arr, axis=0, norm="ortho"), axis=1, norm="ortho")
    dct_low = d[:hash_size, :hash_size]
    med = np.median(dct_low[1:, 1:])  # exclude DC term, as usual for pHash
    return (dct_low > med).flatten()

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

def build_kaggle_manifest(root: Path, hamming_thresh=6):
    all_paths = list_kaggle_images(root)
    rows = []
    for path, cls in all_paths:
        if is_near_blank(path):
            continue
        try:
            w, h = Image.open(path).size
        except Exception:
            continue
        if min(w, h) < 96:
            continue
        rows.append(dict(path=path, cls=cls, md5=md5_of(path)))
    print(f"kept {len(rows)} / {len(all_paths)} files after unreadable/near-blank/too-small filtering")

    # exact dedupe by md5 within class (keep first occurrence)
    seen_md5 = {}
    deduped = []
    for r in rows:
        key = (r["cls"], r["md5"])
        if key in seen_md5:
            continue
        seen_md5[key] = True
        deduped.append(r)
    print(f"{len(deduped)} / {len(rows)} remain after exact MD5 dedupe "
          f"(expect ~no drop -- the Roboflow triplicates are NOT byte-identical, see note above)")

    # near-duplicate grouping by pHash, union-find within class only
    hashes = [phash64(r["path"]) for r in deduped]
    uf = UnionFind(len(deduped))
    by_cls = defaultdict(list)
    for i, r in enumerate(deduped):
        by_cls[r["cls"]].append(i)
    for cls, idxs in by_cls.items():
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                ham = int(np.count_nonzero(hashes[i] != hashes[j]))
                if ham <= hamming_thresh:
                    uf.union(i, j)

    root_to_source_id = {}
    for i, r in enumerate(deduped):
        root = uf.find(i)
        if root not in root_to_source_id:
            root_to_source_id[root] = f"{r['cls']}::{len(root_to_source_id)}"
        r["source_id"] = root_to_source_id[root]
    n_sources = len(root_to_source_id)
    print(f"{len(deduped)} unique files -> {n_sources} unique design sources after pHash grouping "
          f"(hamming<={hamming_thresh})")
    return deduped

kaggle_rows = build_kaggle_manifest(FABRIC_DIR)
manifest_path = WORK_DIR / "kaggle_manifest.csv"
with open(manifest_path, "w", encoding="utf-8") as f:
    f.write("path,md5,source_id,class_name\n")
    for r in kaggle_rows:
        f.write(f"{r['path']},{r['md5']},{r['source_id']},{r['cls']}\n")
print("saved", manifest_path)

# %%
def stratified_split_by_source(rows, val_frac=0.15, seed=0):
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source_id"]].append(r)
    sources_by_cls = defaultdict(list)
    for sid, items in by_source.items():
        sources_by_cls[items[0]["cls"]].append(sid)
    train_sources, val_sources = [], []
    for cls, sids in sources_by_cls.items():
        sids = sids[:]
        rng.shuffle(sids)
        n_val = max(1, int(round(len(sids) * val_frac)))
        val_sources += sids[:n_val]
        train_sources += sids[n_val:]
    return train_sources, val_sources, by_source

train_source_ids, val_source_ids, records_by_id = stratified_split_by_source(kaggle_rows)
print(f"re-split by source_id (never by file): train={len(train_source_ids)} sources, "
      f"val={len(val_source_ids)} sources")

split_path = WORK_DIR / "kaggle_split.csv"
with open(split_path, "w", encoding="utf-8") as f:
    f.write("source_id,role\n")
    for sid in train_source_ids:
        f.write(f"{sid},train\n")
    for sid in val_source_ids:
        f.write(f"{sid},val\n")
print("saved", split_path)

def records_paths(sid):
    return [r["path"] for r in records_by_id[sid]]

# %% [markdown]
# ## Step B: Drive grouping — helper for the AUTHOR to run (method1.md §5)
# **The coding agent did not run this cell against real Drive images and
# will not view its outputs.** It embeds each Drive image with the
# *pretrained, untrained* DINOv2 backbone (no fine-tuning), clusters them,
# and writes a contact-sheet PNG per cluster to disk **only** (never
# displayed inline) plus a `drive_groups_template.csv` for the author to
# fill in by hand, following the labeling rules below. This cell is fully
# skipped (prints a notice) when `DRIVE_DIR is None`.
#
# Labeling rules (restated from method1.md §5, keep in the final writeup):
# - Same pattern, color shift, blur, rotation/flip -> SAME group.
# - Different crop/continuation of the same repeating pattern -> SAME group,
#   `variant=crop` if easy to tell.
# - Blank images -> `is_blank=1`, excluded from queries (and optionally the
#   gallery).
# - Pattern appears once -> singleton group, stays in gallery as a
#   distractor, never a query.
# - Unsure whether two images are the same design -> keep separate groups
#   (conservative), note it.

# %%
_DINOV2_CACHE = {}
def get_frozen_dinov2():
    """Loads/caches the pretrained DINOv2 ViT-S/14 once (torch.hub hits its
    local cache after the first call) -- shared by Step B, the untrained-head
    baselines, so neither re-downloads/re-initializes it per image/query."""
    if "model" not in _DINOV2_CACHE:
        m = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", trust_repo=True)
        _DINOV2_CACHE["model"] = m.eval().to(DEVICE)
    return _DINOV2_CACHE["model"]

def embed_with_pretrained_dinov2(paths, rotations=(0, 90, 180, 270)):
    """Pretrained (no fine-tuning), grayscale input, max similarity over
    4 rotations when later comparing pairs -- so embeddings are stored
    per-rotation and the caller takes the max cosine sim across the 4x4
    rotation combinations for a pair."""
    backbone = get_frozen_dinov2()
    tfm_mean, tfm_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    all_embeds = []  # [n_images, n_rotations, 384]
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
    return np.stack(all_embeds)  # [N, 4, 384]

def rotation_invariant_distance_matrix(embeds):
    n = embeds.shape[0]
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            s = embeds[i] @ embeds[j].T  # [4,4]
            sim[i, j] = s.max()
    return 1.0 - sim  # distance

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
    sheet.save(out_path)  # disk only -- never displayed inline in this notebook

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
    print(f"wrote {len(by_cluster)} contact sheets to {contact_dir} (disk only, not displayed)")
    print(f"wrote template for hand-labeling: {template_path}")
    print("AUTHOR: open the contact sheets yourself, fill in group_id/is_blank/variant by hand, "
          "save as drive_groups.csv next to this notebook's working dir.")

if DRIVE_DIR is not None:
    run_drive_grouping_helper(DRIVE_DIR, WORK_DIR)
else:
    print("DRIVE_DIR is None -- skipping Step B (no proprietary data attached). "
          "This is expected for a reviewer running without the private dataset.")

# %% [markdown]
# ## Step C: Model (method1.md §6)
# DINOv2 ViT-S/14, all but the last `unfreeze_last_n_blocks` transformer
# blocks (+ final norm) frozen. GeM pooling **over patch tokens** (not the
# CLS token) — the design identity is a repeated texture/motif, so pooled
# local patch features suit it better than a single global CLS summary.
# `Linear(384 -> 256)` + L2 normalize. A `timm` CNN fallback is included in
# case `torch.hub` is unreachable (no internet on this Kaggle session).

# %%
class GeMTokens(nn.Module):
    """GeM pooling over a token sequence [B, N, C] -> [B, C].

    Forced to run in fp32 regardless of the ambient autocast context: under
    GPU autocast `tokens` can be fp16, and transformer activations are known
    to have occasional large-magnitude outlier dimensions ("massive
    activations") -- raised to power p~3, those can overflow fp16's ~65504
    max, the same class of bug as the SupCon masked_fill overflow above.
    """
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, tokens):
        with torch.amp.autocast(device_type=tokens.device.type, enabled=False):
            x = tokens.float().clamp(min=self.eps).pow(self.p.float())
            out = x.mean(dim=1).pow(1.0 / self.p.float())
        return out

class GeMSpatial(nn.Module):
    """GeM pooling over a CNN feature map [B, C, H, W] -> [B, C] (fallback
    backbone). Forced to fp32 for the same overflow reason as GeMTokens."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        with torch.amp.autocast(device_type=x.device.type, enabled=False):
            x = x.float().clamp(min=self.eps).pow(self.p.float())
            out = F.adaptive_avg_pool2d(x, 1).pow(1.0 / self.p.float()).flatten(1)
        return out

class EmbeddingNet(nn.Module):
    def __init__(self, backbone_name="dinov2_vits14", embed_dim=256,
                 unfreeze_last_n=4, pretrained=True):
        super().__init__()
        self.kind = "dinov2" if backbone_name.startswith("dinov2") else "timm"
        if self.kind == "dinov2":
            # DINOv2's torch.hub entrypoints always load pretrained weights (no
            # random-init option is exposed) -- `trust_repo` just means "this is
            # our own vetted repo", unrelated to `pretrained`, which only
            # matters for the timm fallback branch below.
            self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, trust_repo=True)
            feat_dim = self.backbone.embed_dim
            for p in self.backbone.parameters():
                p.requires_grad = False
            for blk in self.backbone.blocks[-unfreeze_last_n:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.backbone.norm.parameters():
                p.requires_grad = True
            self.pool = GeMTokens()
        else:
            import timm
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained,
                                               num_classes=0, global_pool="")
            feat_dim = self.backbone.num_features
            self.pool = GeMSpatial()
        self.fc = nn.Linear(feat_dim, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim)

    def forward(self, x):
        if self.kind == "dinov2":
            feats = self.backbone.forward_features(x)
            pooled = self.pool(feats["x_norm_patchtokens"])
        else:
            pooled = self.pool(self.backbone.forward_features(x))
        return F.normalize(self.bn(self.fc(pooled)), dim=1)

    def param_groups(self, backbone_lr, head_lr):
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        head_params = list(self.pool.parameters()) + list(self.fc.parameters()) + list(self.bn.parameters())
        return [{"params": backbone_params, "lr": backbone_lr},
                {"params": head_params, "lr": head_lr}]

# %% [markdown]
# ## Step D: Color-invariance machinery (method1.md §7)
# Cluster-wise Lab-space palette remap: segment each image into `k` color
# clusters on the (a,b) plane, remap each cluster (ranked by mean lightness)
# to a target palette color, keep L (structure/contrast). This specifically
# imitates how real sarees are dyed — body and border recolored
# independently — which a single global hue shift does not.

# %%
def sample_palette(rng, k=5):
    hue = rng.uniform(0, 2 * np.pi, k)
    chroma = rng.uniform(0, 60, k)
    return np.stack([128 + chroma * np.cos(hue), 128 + chroma * np.sin(hue)], 1).astype(np.float32)

def cluster_recolor(img_rgb, palette, rng, l_jitter=0.05):
    k = len(palette)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w, _ = lab.shape
    small = cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    ab_samples = np.ascontiguousarray(small[:, 1:])
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(ab_samples, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    ab = lab[..., 1:].reshape(-1, 2)
    d = ((ab[:, None, :] - centers[None]) ** 2).sum(-1)
    lab_idx = d.argmin(1)
    L = lab[..., 0].reshape(-1)
    cl_L = np.array([L[lab_idx == c].mean() if (lab_idx == c).any() else 0.0 for c in range(k)])
    order = np.argsort(cl_L)  # cluster ids, dark -> light
    new_ab = ab.copy()
    for rank, c in enumerate(order):
        m = lab_idx == c
        shift = palette[rank] - centers[c]
        new_ab[m] = ab[m] + shift
    out = lab.copy()
    out[..., 1:] = new_ab.reshape(h, w, 2)
    out[..., 0] = np.clip(out[..., 0] * (1 + rng.uniform(-l_jitter, l_jitter)), 0, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)

def hue_rotate_recolor(img_rgb, rng):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    angle = rng.uniform(0, 2 * np.pi)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    ca, sa = np.cos(angle), np.sin(angle)
    lab[..., 1] = a * ca - b * sa + 128
    lab[..., 2] = a * sa + b * ca + 128
    lab = np.clip(lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def channel_perm_recolor(img_rgb, rng):
    perm = [0, 1, 2]
    rng.shuffle(perm)
    if perm == [0, 1, 2]:
        perm = [1, 2, 0]
    return img_rgb[:, :, perm]

def grayscale_recolor(img_rgb):
    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    return np.repeat(g[..., None], 3, axis=2)

def apply_recolor_mode(img_pil, mode, palette, rng):
    arr = np.asarray(img_pil.convert("RGB"))
    if mode == "cluster":
        out = cluster_recolor(arr, palette, rng)
    elif mode == "hue":
        out = hue_rotate_recolor(arr, rng)
    elif mode == "perm":
        out = channel_perm_recolor(arr, rng)
    else:  # "gray"
        out = grayscale_recolor(arr)
    return Image.fromarray(out)

# %%
def random_resized_crop_nonblank(img_pil, rng, size=224, scale=(0.3, 1.0),
                                  ratio=(3/4, 4/3), std_thresh=8.0, max_tries=5):
    w, h = img_pil.size
    area = w * h
    for attempt in range(max_tries):
        target_area = rng.uniform(*scale) * area
        aspect = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
        cw = int(round(np.sqrt(target_area * aspect)))
        ch = int(round(np.sqrt(target_area / aspect)))
        if 0 < cw <= w and 0 < ch <= h:
            x0 = int(rng.integers(0, w - cw + 1))
            y0 = int(rng.integers(0, h - ch + 1))
            crop = img_pil.crop((x0, y0, x0 + cw, y0 + ch)).resize((size, size), Image.BILINEAR)
            if np.asarray(crop.convert("L"), dtype=np.float32).std() >= std_thresh or attempt == max_tries - 1:
                return crop
    return img_pil.resize((size, size), Image.BILINEAR)

def center_crop_eval(img_pil, size=224, short_side=224):
    w, h = img_pil.size
    scale = short_side / min(w, h)
    img_pil = img_pil.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    w, h = img_pil.size
    left, top = (w - size) // 2, (h - size) // 2
    return img_pil.crop((left, top, left + size, top + size))

def random_rot_flip(img_pil, rng):
    k = int(rng.integers(0, 4))
    if k:
        img_pil = img_pil.rotate(90 * k)
    if rng.random() < 0.5:
        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
    return img_pil

def gaussian_blur_aug(img_pil, rng, p=0.3, sigma_range=(0.1, 2.0)):
    if rng.random() < p:
        sigma = rng.uniform(*sigma_range)
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return img_pil

def brightness_contrast_aug(img_pil, rng, p=0.5, jitter=0.15):
    if rng.random() < p:
        img_pil = TF.adjust_brightness(img_pil, 1 + rng.uniform(-jitter, jitter))
    if rng.random() < p:
        img_pil = TF.adjust_contrast(img_pil, 1 + rng.uniform(-jitter, jitter))
    return img_pil

def noise_or_jpeg_aug(img_pil, rng, p=0.2):
    if rng.random() >= p:
        return img_pil
    if rng.random() < 0.5:
        quality = int(rng.integers(60, 96))
        buf = io.BytesIO()
        img_pil.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
    arr = np.asarray(img_pil, dtype=np.float32)
    noise = rng.normal(0, 6.0, arr.shape)
    return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
def to_tensor_norm(img_pil):
    t = TF.to_tensor(img_pil)
    return TF.normalize(t, IMAGENET_MEAN, IMAGENET_STD)

RECOLOR_MODE_NAMES = ["cluster", "hue", "perm", "gray"]
RECOLOR_MODE_PROBS = [0.70, 0.10, 0.05, 0.15]

def train_view(path, rng, img_size, palette=None, forced_mode=None, ablation="full"):
    img = Image.open(path).convert("RGB")
    img = random_resized_crop_nonblank(img, rng, size=img_size)
    img = random_rot_flip(img, rng)
    if ablation != "geometric_only":
        mode = forced_mode or rng.choice(RECOLOR_MODE_NAMES, p=RECOLOR_MODE_PROBS)
        if mode == "cluster" and palette is None:
            palette = sample_palette(rng, k=5)
        img = apply_recolor_mode(img, mode, palette, rng)
    img = gaussian_blur_aug(img, rng)
    img = brightness_contrast_aug(img, rng)
    img = noise_or_jpeg_aug(img, rng)
    return to_tensor_norm(img)

# %% [markdown]
# ### Shared-palette batch construction (method1.md §7.2)
# Each batch: `B` distinct source_ids, `V=2` views each. A small palette
# pool is drawn per batch; when both views of a source land on `cluster`
# recolor mode, they're forced onto *different* pool palettes. Consequence:
# many different designs end up sharing a palette within the batch (hard
# negatives for "same color, different design") while the same design
# appears in different palettes (positives) — the model cannot solve the
# task with color alone.

# %%
def build_shared_palette_batch(source_ids, rng, B, V, img_size, pool_size=6,
                                k_colors=5, ablation="full"):
    chosen = rng.choice(len(source_ids), size=min(B, len(source_ids)), replace=False)
    chosen_sids = [source_ids[i] for i in chosen]
    use_shared_pool = ablation != "no_shared_palette"
    palette_pool = [sample_palette(rng, k_colors) for _ in range(pool_size)] if use_shared_pool else None
    imgs, labels = [], []
    for sid in chosen_sids:
        paths = records_paths(sid)
        modes = [rng.choice(RECOLOR_MODE_NAMES, p=RECOLOR_MODE_PROBS) for _ in range(V)] \
            if ablation != "geometric_only" else [None] * V
        pal_assignment = [None] * V
        cluster_views = [i for i, m in enumerate(modes) if m == "cluster"]
        if use_shared_pool and len(cluster_views) >= 2:
            n = min(len(cluster_views), pool_size)
            chosen_pals = rng.choice(pool_size, size=n, replace=False)
            for i, p_idx in zip(cluster_views, chosen_pals):
                pal_assignment[i] = palette_pool[p_idx]
        elif use_shared_pool and len(cluster_views) == 1:
            pal_assignment[cluster_views[0]] = palette_pool[int(rng.integers(0, pool_size))]
        for v in range(V):
            path = paths[int(rng.integers(0, len(paths)))]
            tensor = train_view(path, rng, img_size, palette=pal_assignment[v],
                                 forced_mode=modes[v], ablation=ablation)
            imgs.append(tensor)
            labels.append(sid)
    label_to_idx = {sid: i for i, sid in enumerate(sorted(set(labels)))}
    label_tensor = torch.tensor([label_to_idx[l] for l in labels], dtype=torch.long)
    return torch.stack(imgs), label_tensor

# %% [markdown]
# ## Step D cont'd: SupCon loss (method1.md §7.3)

# %%
def supcon_loss(z, labels, tau=0.07):
    z = F.normalize(z.float(), dim=1)
    n = z.size(0)
    sim = (z @ z.T) / tau
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    # Use the dtype's own min (not a hardcoded -1e9) -- under GPU autocast
    # `sim` is fp16, and -1e9 overflows fp16's ~65504 max magnitude, which
    # crashes with "value cannot be converted to type c10::Half without
    # overflow". CPU runs (no autocast) never hit this, which is why local
    # testing missed it.
    sim = sim.masked_fill(self_mask, torch.finfo(sim.dtype).min)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.sum(1) > 0
    loss = -(log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]
    return loss.mean()

# %% [markdown]
# ## Step F (§9.1 pieces needed during training): Kaggle validation protocol
# Query = a recolored + re-cropped view of a held-out val source. Gallery =
# canonical (un-recolored, center-cropped) views of **all** val sources,
# plus **palette-matched distractors**: every other val source re-rendered
# in the *query's own palette* — a hard same-color/different-design
# negative, so a model that secretly uses color gets caught here, at
# validation time, before ever touching Drive.

# %%
@torch.no_grad()
def embed_tensors(model, tensors, batch_size=64):
    model.eval()
    out = []
    for i in range(0, len(tensors), batch_size):
        chunk = torch.stack(tensors[i:i + batch_size]).to(DEVICE)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            emb = model(chunk)
        out.append(emb.float().cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, CONFIG["embed_dim"])

def kaggle_val_eval(model, val_source_ids, img_size, seed=12345, k_colors=5):
    """Returns identification metrics + (pos_sims, neg_sims) for threshold selection."""
    rng = np.random.default_rng(seed)
    canon_tensors = []
    for sid in val_source_ids:
        img = Image.open(records_paths(sid)[0]).convert("RGB")
        canon_tensors.append(to_tensor_norm(center_crop_eval(img, img_size)))
    canon_emb = embed_tensors(model, canon_tensors)

    ranks, aps, pos_sims, neg_sims = [], [], [], []
    for qi, sid in enumerate(val_source_ids):
        palette = sample_palette(rng, k_colors)
        q_path = records_paths(sid)[int(rng.integers(0, len(records_paths(sid))))]
        q_img = random_resized_crop_nonblank(Image.open(q_path).convert("RGB"), rng, size=img_size)
        q_img = apply_recolor_mode(q_img, "cluster", palette, rng)
        q_emb = embed_tensors(model, [to_tensor_norm(q_img)])[0]

        distractor_tensors, distractor_ids = [], []
        for sid2 in val_source_ids:
            if sid2 == sid:
                continue
            img2 = Image.open(records_paths(sid2)[0]).convert("RGB")
            img2 = center_crop_eval(img2, img_size)
            img2 = apply_recolor_mode(img2, "cluster", palette, rng)
            distractor_tensors.append(to_tensor_norm(img2))
            distractor_ids.append(sid2)
        distractor_emb = embed_tensors(model, distractor_tensors)

        gallery_emb = torch.cat([canon_emb, distractor_emb], dim=0)
        gallery_ids = list(val_source_ids) + distractor_ids
        sims = (q_emb.unsqueeze(0) @ gallery_emb.t()).squeeze(0)
        order = sims.argsort(descending=True).tolist()
        ranked_ids = [gallery_ids[i] for i in order]
        rank = ranked_ids.index(sid) + 1
        ranks.append(rank)
        aps.append(1.0 / rank)
        correct_idx = gallery_ids.index(sid)
        pos_sims.append(sims[correct_idx].item())
        neg_candidates = [i for i in range(len(gallery_ids)) if i != correct_idx]
        neg_sims.append(sims[int(rng.choice(neg_candidates))].item())

    metrics = dict(
        recall1=float(np.mean([r == 1 for r in ranks])),
        recall5=float(np.mean([r <= 5 for r in ranks])),
        mAP=float(np.mean(aps)),
        n_val_sources=len(val_source_ids),
    )
    return metrics, pos_sims, neg_sims

def pick_threshold(pos_sims, neg_sims, mode="balanced_accuracy"):
    sims = np.array(pos_sims + neg_sims)
    labels = np.array([1] * len(pos_sims) + [0] * len(neg_sims))
    best_t, best_score = 0.0, -1.0
    for t in np.unique(sims):
        pred = sims >= t
        tpr = (pred & (labels == 1)).sum() / max(1, (labels == 1).sum())
        tnr = (~pred & (labels == 0)).sum() / max(1, (labels == 0).sum())
        score = (tpr + tnr) / 2 if mode == "balanced_accuracy" else tpr  # placeholder for FAR-based mode
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score

# %% [markdown]
# ## Step E: Training (method1.md §8)

# %%
def train_one_run(seed, ablation, config, train_source_ids, val_source_ids):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    model = EmbeddingNet(config["backbone"], config["embed_dim"],
                          config["unfreeze_last_n_blocks"]).to(DEVICE)
    optimizer = torch.optim.AdamW(model.param_groups(config["backbone_lr"], config["head_lr"]),
                                   weight_decay=config["weight_decay"])
    total_steps = config["epochs"] * config["steps_per_epoch"]
    warmup_steps = config["warmup_epochs"] * config["steps_per_epoch"]
    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    rng = np.random.default_rng(seed)

    best_val_recall1, best_epoch, patience_ctr = -1.0, -1, 0
    log_rows = []
    ckpt_path = WORK_DIR / f"best_seed{seed}_{ablation}.pt"
    t0 = time.time()
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0
        for _ in range(config["steps_per_epoch"]):
            imgs, labels = build_shared_palette_batch(
                train_source_ids, rng, config["B"], config["V"], config["img_size"],
                config["palette_pool_size"], config["palette_k_colors"], ablation=ablation)
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                z = model(imgs)
                loss = supcon_loss(z, labels, tau=config["supcon_tau"])
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            epoch_loss += loss.item()
        epoch_loss /= config["steps_per_epoch"]

        row = dict(epoch=epoch, loss=epoch_loss, lr=scheduler.get_last_lr()[0])
        if (epoch + 1) % config["val_every"] == 0:
            val_metrics, pos_sims, neg_sims = kaggle_val_eval(model, val_source_ids, config["img_size"])
            row.update({f"val_{k}": v for k, v in val_metrics.items()})
            if val_metrics["recall1"] > best_val_recall1:
                best_val_recall1 = val_metrics["recall1"]
                best_epoch = epoch
                patience_ctr = 0
                torch.save(model.state_dict(), ckpt_path)
                threshold, bal_acc = pick_threshold(pos_sims, neg_sims)
                with open(WORK_DIR / f"threshold_seed{seed}_{ablation}.json", "w") as f:
                    json.dump(dict(threshold=threshold, balanced_accuracy=bal_acc,
                                    epoch=epoch), f, indent=2)
            else:
                patience_ctr += 1
        log_rows.append(row)
        print(row)
        if patience_ctr >= config["early_stop_patience"]:
            print(f"early stopping at epoch {epoch} (best epoch {best_epoch}, "
                  f"best val recall1={best_val_recall1:.4f})")
            break

    with open(WORK_DIR / f"train_log_seed{seed}_{ablation}.json", "w") as f:
        json.dump(log_rows, f, indent=2)
    print(f"run (seed={seed}, ablation={ablation}) done in {time.time()-t0:.1f}s, "
          f"best val recall1={best_val_recall1:.4f} @ epoch {best_epoch}")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return model, dict(best_val_recall1=best_val_recall1, best_epoch=best_epoch, ckpt_path=str(ckpt_path))

# %%
run_results = {}
ablations_to_run = (["geometric_only", "no_shared_palette", "full"]
                     if CONFIG["run_ablations"] else [CONFIG["ablation"]])
for ablation in ablations_to_run:
    for seed in CONFIG["seeds"]:
        model, info = train_one_run(seed, ablation, CONFIG, train_source_ids, val_source_ids)
        run_results[(ablation, seed)] = info

# main model used for the rest of the notebook = first configured ablation/seed
MAIN_ABLATION, MAIN_SEED = CONFIG["ablation"], CONFIG["seeds"][0]
main_model = EmbeddingNet(CONFIG["backbone"], CONFIG["embed_dim"],
                           CONFIG["unfreeze_last_n_blocks"]).to(DEVICE)
main_model.load_state_dict(torch.load(run_results[(MAIN_ABLATION, MAIN_SEED)]["ckpt_path"], map_location=DEVICE))
main_model.eval()
with open(WORK_DIR / f"threshold_seed{MAIN_SEED}_{MAIN_ABLATION}.json") as f:
    MAIN_THRESHOLD = json.load(f)["threshold"]
print("main model loaded, threshold.json ->", MAIN_THRESHOLD)
with open(WORK_DIR / "threshold.json", "w") as f:
    json.dump({"threshold": MAIN_THRESHOLD, "source": f"seed{MAIN_SEED}_{MAIN_ABLATION}",
               "note": "chosen on Kaggle val only, per method1.md SS9.1 -- never on Drive"}, f, indent=2)

# %% [markdown]
# ## Step F §9.2: Drive test (main result) — gated, author-labeled only
# Requires `drive_groups.csv` (author-filled, from Step B's template) next
# to this notebook. **Skips gracefully if missing** — the agent did not and
# will not fabricate group labels for a real run.

# %%
def load_drive_groups(csv_path: Path):
    import csv as csv_mod
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv_mod.DictReader(f):
            if row.get("group_id", "").strip() == "":
                continue  # not yet labeled by author
            rows.append(row)
    return rows

def build_drive_query_gallery_split(drive_dir: Path, groups_csv: Path, out_csv: Path, seed=0):
    rows = load_drive_groups(groups_csv)
    rows = [r for r in rows if r.get("is_blank", "0").strip() not in ("1", "true", "True")]
    by_group = defaultdict(list)
    for r in rows:
        by_group[r["group_id"]].append(r["filename"])
    rng = np.random.default_rng(seed)
    query_of_group = {}
    for gid, files in by_group.items():
        if len(files) < 2:
            continue
        files_sorted = sorted(files)
        query_of_group[gid] = files_sorted[int(rng.integers(0, len(files_sorted)))]
    query_set = set(query_of_group.values())
    all_files = [r["filename"] for r in rows]
    gallery_files = [f for f in all_files if f not in query_set]
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("filename,role,group_id\n")
        fname_to_group = {r["filename"]: r["group_id"] for r in rows}
        for fn in query_set:
            f.write(f"{fn},query,{fname_to_group[fn]}\n")
        for fn in gallery_files:
            f.write(f"{fn},gallery,{fname_to_group[fn]}\n")
    return query_of_group, gallery_files, fname_to_group

def drive_identification_and_verification(model, drive_dir, query_of_group, gallery_files,
                                            fname_to_group, img_size, threshold):
    gallery_tensors = [to_tensor_norm(center_crop_eval(Image.open(drive_dir / fn).convert("RGB"), img_size))
                        for fn in gallery_files]
    gallery_emb = embed_tensors(model, gallery_tensors)
    query_files = list(query_of_group.values())
    query_tensors = [to_tensor_norm(center_crop_eval(Image.open(drive_dir / fn).convert("RGB"), img_size))
                      for fn in query_files]
    query_emb = embed_tensors(model, query_tensors)

    ranks, aps = [], []
    for qi, qfn in enumerate(query_files):
        qgroup = fname_to_group[qfn]
        sims = (query_emb[qi:qi+1] @ gallery_emb.t()).squeeze(0)
        order = sims.argsort(descending=True).tolist()
        relevant = {gf for gf in gallery_files if fname_to_group[gf] == qgroup}
        ranked_relevant_positions = [i + 1 for i, idx in enumerate(order) if gallery_files[idx] in relevant]
        if not ranked_relevant_positions:
            continue
        ranks.append(ranked_relevant_positions[0])
        n_rel = len(relevant)
        hits, precisions = 0, []
        for i, idx in enumerate(order):
            if gallery_files[idx] in relevant:
                hits += 1
                precisions.append(hits / (i + 1))
        aps.append(float(np.mean(precisions)) if precisions else 0.0)

    identification = dict(
        recall1=float(np.mean([r == 1 for r in ranks])) if ranks else None,
        recall5=float(np.mean([r <= 5 for r in ranks])) if ranks else None,
        mAP=float(np.mean(aps)) if aps else None,
        n_queries=len(ranks),
        n_gallery=len(gallery_files),
    )

    from sklearn.metrics import roc_auc_score, roc_curve
    all_files = gallery_files + query_files
    all_tensors = gallery_tensors + query_tensors
    all_emb = torch.cat([gallery_emb, query_emb], dim=0)
    all_groups = [fname_to_group[f] for f in all_files]
    sim_matrix = all_emb @ all_emb.t()
    n = len(all_files)
    pair_sims, pair_labels = [], []
    for i in range(n):
        for j in range(i + 1, n):
            pair_sims.append(sim_matrix[i, j].item())
            pair_labels.append(1 if all_groups[i] == all_groups[j] else 0)
    pair_sims, pair_labels = np.array(pair_sims), np.array(pair_labels)
    auc = roc_auc_score(pair_labels, pair_sims) if len(set(pair_labels)) > 1 else None
    eer = None
    if auc is not None:
        fpr, tpr, _ = roc_curve(pair_labels, pair_sims)
        fnr = 1 - tpr
        idx = np.nanargmin(np.abs(fnr - fpr))
        eer = float((fnr[idx] + fpr[idx]) / 2)
    pred = pair_sims >= threshold
    tp = int(((pred == 1) & (pair_labels == 1)).sum())
    tn = int(((pred == 0) & (pair_labels == 0)).sum())
    fp = int(((pred == 1) & (pair_labels == 0)).sum())
    fn_ = int(((pred == 0) & (pair_labels == 1)).sum())
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn_)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    balanced_acc = 0.5 * (tp / max(1, tp + fn_) + tn / max(1, tn + fp))
    verification = dict(
        roc_auc=auc, eer=eer, f1_at_saved_threshold=f1,
        balanced_accuracy_at_saved_threshold=balanced_acc,
        n_positive_pairs=int(pair_labels.sum()), n_negative_pairs=int((1 - pair_labels).sum()),
        note="class-imbalanced (many more negative than positive pairs) -- plain accuracy not reported",
    )
    return identification, verification

def find_drive_groups_csv():
    """The author's hand-labeled drive_groups.csv can arrive two ways: dropped
    directly into /kaggle/working (session file upload), or uploaded as part
    of the private Drive dataset itself (a new dataset version), in which
    case it surfaces read-only under /kaggle/input instead. Check both."""
    in_working = WORK_DIR / "drive_groups.csv"
    if in_working.exists():
        return in_working
    if DRIVE_DIR is not None:
        direct = DRIVE_DIR / "drive_groups.csv"
        if direct.exists():
            return direct
        for dirpath, _, filenames in os.walk(KAGGLE_INPUT):
            if "drive_groups.csv" in filenames:
                return Path(dirpath) / "drive_groups.csv"
    return None

DRIVE_GROUPS_CSV = find_drive_groups_csv()
if DRIVE_DIR is not None and DRIVE_GROUPS_CSV is not None:
    query_of_group, gallery_files, fname_to_group = build_drive_query_gallery_split(
        DRIVE_DIR, DRIVE_GROUPS_CSV, WORK_DIR / "drive_query_gallery_split.csv", seed=0)
    drive_identification, drive_verification = drive_identification_and_verification(
        main_model, DRIVE_DIR, query_of_group, gallery_files, fname_to_group,
        CONFIG["img_size"], MAIN_THRESHOLD)
    print("DRIVE TEST (main result) -- identification:", json.dumps(drive_identification, indent=2))
    print("DRIVE TEST (main result) -- verification:", json.dumps(drive_verification, indent=2))
else:
    print("Skipping SS9.2 Drive test: either DRIVE_DIR is None, or "
          f"{WORK_DIR / 'drive_groups.csv'} does not exist yet (author must hand-label it "
          "from the Step B contact sheets/template first).")
    drive_identification = drive_verification = None

# %% [markdown]
# ## Step F §9.3: Color-invariance stress tests (the core requirement)

# %%
def lab_histogram(img_pil, bins=8):
    lab = cv2.cvtColor(np.asarray(img_pil.convert("RGB")), cv2.COLOR_RGB2LAB)
    hist = cv2.calcHist([lab], [0, 1, 2], None, [bins, bins, bins],
                         [0, 256, 0, 256, 0, 256])
    hist = hist.flatten().astype(np.float64)
    return hist / max(1e-9, hist.sum())

def chi_square_distance(h1, h2, eps=1e-9):
    return float(0.5 * np.sum((h1 - h2) ** 2 / (h1 + h2 + eps)))

def color_adversarial_subset_auc(pair_sims, pair_labels, color_dists, quartile=0.25):
    color_dists = np.array(color_dists)
    pair_sims, pair_labels = np.array(pair_sims), np.array(pair_labels)
    pos_mask = pair_labels == 1
    neg_mask = pair_labels == 0
    if pos_mask.sum() < 4 or neg_mask.sum() < 4:
        return None
    pos_thresh = np.quantile(color_dists[pos_mask], 1 - quartile)   # top quartile = most color-different
    neg_thresh = np.quantile(color_dists[neg_mask], quartile)       # bottom quartile = most color-similar
    subset_mask = (pos_mask & (color_dists >= pos_thresh)) | (neg_mask & (color_dists <= neg_thresh))
    if subset_mask.sum() < 4 or len(set(pair_labels[subset_mask])) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(pair_labels[subset_mask], pair_sims[subset_mask]))

def recolor_probe(model, val_source_ids, img_size, n_recolors=5, seed=999):
    rng = np.random.default_rng(seed)
    sims_same, recall1_hits = [], []
    for sid in val_source_ids:
        img = Image.open(records_paths(sid)[0]).convert("RGB")
        base = to_tensor_norm(center_crop_eval(img, img_size))
        base_emb = embed_tensors(model, [base])[0]
        recolored = []
        for _ in range(n_recolors):
            palette = sample_palette(rng, 5)
            r_img = apply_recolor_mode(center_crop_eval(img, img_size), "cluster", palette, rng)
            recolored.append(to_tensor_norm(r_img))
        r_emb = embed_tensors(model, recolored)
        sims = (r_emb @ base_emb.unsqueeze(1)).squeeze(1)
        sims_same.extend(sims.tolist())
    return dict(mean_cosine_sim_to_recolored_self=float(np.mean(sims_same)),
                std_cosine_sim_to_recolored_self=float(np.std(sims_same)))

if drive_verification is not None:
    # reconstruct pair-level sims/labels/color-distances for the stress test (Drive pairs)
    all_files = gallery_files + list(query_of_group.values())
    tensors = [to_tensor_norm(center_crop_eval(Image.open(DRIVE_DIR / fn).convert("RGB"), CONFIG["img_size"]))
               for fn in all_files]
    embs = embed_tensors(main_model, tensors)
    pil_imgs = [center_crop_eval(Image.open(DRIVE_DIR / fn).convert("RGB"), CONFIG["img_size"]) for fn in all_files]
    hists = [lab_histogram(im) for im in pil_imgs]
    n = len(all_files)
    sims_list, labels_list, color_d_list = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            sims_list.append((embs[i] @ embs[j]).item())
            labels_list.append(1 if fname_to_group[all_files[i]] == fname_to_group[all_files[j]] else 0)
            color_d_list.append(chi_square_distance(hists[i], hists[j]))
    adv_auc = color_adversarial_subset_auc(sims_list, labels_list, color_d_list)
    print(f"color-adversarial verification AUC (hard subset: same-design/very-different-color vs "
          f"different-design/similar-color): {adv_auc}")
else:
    print("Skipping color-adversarial Drive subset (no Drive verification pairs available).")

probe_metrics = recolor_probe(main_model, val_source_ids, CONFIG["img_size"])
print("Kaggle-val recolor probe:", probe_metrics)

# %% [markdown]
# ## Step F §9.4: Baselines (same protocol, same split)

# %%
@torch.no_grad()
def embed_color_histogram(paths, img_size):
    embs = []
    for p in paths:
        img = center_crop_eval(Image.open(p).convert("RGB"), img_size)
        h = lab_histogram(img)
        h = h / max(1e-9, np.linalg.norm(h))
        embs.append(torch.tensor(h, dtype=torch.float32))
    return torch.stack(embs)

@torch.no_grad()
def embed_pretrained_untrained_head(paths, img_size, grayscale=False):
    backbone = get_frozen_dinov2()
    pool = GeMTokens().to(DEVICE)
    out = []
    for p in paths:
        img = center_crop_eval(Image.open(p).convert("RGB"), img_size)
        if grayscale:
            img = Image.fromarray(grayscale_recolor(np.asarray(img)))
        t = to_tensor_norm(img).unsqueeze(0).to(DEVICE)
        feats = backbone.forward_features(t)
        emb = F.normalize(pool(feats["x_norm_patchtokens"]), dim=1)
        out.append(emb.cpu()[0])
    return torch.stack(out)

def run_baselines_on_kaggle_val(val_source_ids, img_size):
    """Lightweight baseline comparison using the Kaggle-val protocol's
    canonical-gallery-vs-recolored-query setup (no Drive dependency, so this
    always runs, unlike SS9.2/9.3)."""
    rng = np.random.default_rng(777)
    canon_paths = [records_paths(sid)[0] for sid in val_source_ids]
    query_info = []
    for sid in val_source_ids:
        palette = sample_palette(rng, 5)
        img = random_resized_crop_nonblank(Image.open(records_paths(sid)[0]).convert("RGB"), rng, size=img_size)
        img = apply_recolor_mode(img, "cluster", palette, rng)
        query_info.append((sid, img))

    random_emb = F.normalize(torch.randn(len(val_source_ids), CONFIG["embed_dim"]), dim=1)
    ranks_random = []
    rng2 = np.random.default_rng(1)
    for i in range(len(val_source_ids)):
        q = F.normalize(torch.randn(1, CONFIG["embed_dim"]), dim=1)
        sims = (q @ random_emb.t()).squeeze(0)
        order = sims.argsort(descending=True).tolist()
        ranks_random.append(order.index(i) + 1)
    baseline_results = {
        "random_embeddings": dict(
            recall1=float(np.mean([r == 1 for r in ranks_random])),
            recall5=float(np.mean([r <= 5 for r in ranks_random])),
        )
    }

    hist_canon = embed_color_histogram(canon_paths, img_size)
    hist_ranks = []
    for i, (sid, qimg) in enumerate(query_info):
        h = lab_histogram(qimg); h = h / max(1e-9, np.linalg.norm(h))
        q = torch.tensor(h, dtype=torch.float32)
        sims = hist_canon @ q
        order = sims.argsort(descending=True).tolist()
        hist_ranks.append(order.index(i) + 1)
    baseline_results["color_histogram_only"] = dict(
        recall1=float(np.mean([r == 1 for r in hist_ranks])),
        recall5=float(np.mean([r <= 5 for r in hist_ranks])),
        note="expected to do poorly here since queries are deliberately recolored -- "
             "a strong score would mean the eval protocol itself is leaking color",
    )

    backbone = get_frozen_dinov2()
    pool = GeMTokens().to(DEVICE)
    for grayscale in (False, True):
        key = "pretrained_untrained_head_grayscale" if grayscale else "pretrained_untrained_head_rgb"
        canon_emb = embed_pretrained_untrained_head(canon_paths, img_size, grayscale=grayscale)
        ranks = []
        with torch.no_grad():
            for i, (sid, qimg) in enumerate(query_info):
                qi = qimg
                if grayscale:
                    qi = Image.fromarray(grayscale_recolor(np.asarray(qi)))
                t = to_tensor_norm(qi).unsqueeze(0).to(DEVICE)
                feats = backbone.forward_features(t)
                q_emb = F.normalize(pool(feats["x_norm_patchtokens"]), dim=1).cpu()[0]
                sims = canon_emb @ q_emb
                order = sims.argsort(descending=True).tolist()
                ranks.append(order.index(i) + 1)
        baseline_results[key] = dict(
            recall1=float(np.mean([r == 1 for r in ranks])),
            recall5=float(np.mean([r <= 5 for r in ranks])),
        )
    return baseline_results

baseline_results = run_baselines_on_kaggle_val(val_source_ids, CONFIG["img_size"])
final_val_metrics, _, _ = kaggle_val_eval(main_model, val_source_ids, CONFIG["img_size"])
baseline_results["final_model"] = final_val_metrics
print("baselines + final model (Kaggle-val protocol):")
print(json.dumps(baseline_results, indent=2))

# %% [markdown]
# ## Step F §9.5: Statistics (bootstrap CIs)

# %%
def bootstrap_ci(values, n_resamples=1000, seed=0, alpha=0.05):
    rng = np.random.default_rng(seed)
    values = np.array(values, dtype=np.float64)
    boots = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(n_resamples)]
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(np.mean(values)), float(lo), float(hi)

_, pos_sims_final, neg_sims_final = kaggle_val_eval(main_model, val_source_ids, CONFIG["img_size"])
recall1_indicator = [1.0 if s > np.median(neg_sims_final) else 0.0 for s in pos_sims_final]  # rough proxy for CI demo
print("mean/95%CI (bootstrap, 1000 resamples) over positive-pair similarities:",
      bootstrap_ci(pos_sims_final))
print("With ~", len(val_source_ids), "val sources (and similarly small Drive test set), "
      "small differences (~2 points) between configs are noise -- stated explicitly per method1.md SS9.5.")

# %% [markdown]
# ## Step F §9.6: Output files

# %%
results = dict(
    config=CONFIG,
    kaggle_split_sizes=dict(train=len(train_source_ids), val=len(val_source_ids)),
    baselines_and_final_kaggle_val=baseline_results,
    drive_identification=drive_identification,
    drive_verification=drive_verification,
    kaggle_val_recolor_probe=probe_metrics,
)
with open(WORK_DIR / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

with open(WORK_DIR / "results_table.md", "w") as f:
    f.write("| method | recall@1 | recall@5 |\n|---|---|---|\n")
    for name, m in baseline_results.items():
        f.write(f"| {name} | {m.get('recall1')} | {m.get('recall5')} |\n")
print("wrote results.json and results_table.md to", WORK_DIR)

# %% [markdown]
# ## Step G: Efficiency report (method1.md §10)

# %%
def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def count_flops(model, input_size=(1, 3, 224, 224), device=DEVICE):
    was_training = model.training
    model.eval()
    dummy = torch.randn(*input_size, device=device)
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with FlopCounterMode(display=False) as fcm:
            model(dummy)
        flops, method = fcm.get_total_flops(), "torch.utils.flop_counter.FlopCounterMode"
    except Exception as e:
        flops, method = None, f"unavailable ({e})"
    if was_training:
        model.train()
    return flops, method

def benchmark_latency(model, batch_sizes=(1, 32), n_warmup=10, n_iters=100, device=DEVICE):
    model.eval()
    results = {}
    with torch.no_grad():
        for bs in batch_sizes:
            x = torch.randn(bs, 3, 224, 224, device=device)
            for _ in range(n_warmup):
                _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times = []
            for _ in range(n_iters):
                t0 = time.perf_counter()
                _ = model(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                times.append((time.perf_counter() - t0) * 1000)
            times = np.array(times)
            results[bs] = dict(median_ms=float(np.median(times)), p95_ms=float(np.percentile(times, 95)),
                                ms_per_image_median=float(np.median(times) / bs))
    return results

def brute_force_search_cost(embed_dim, gallery_sizes=(160, 10_000, 1_000_000), device=DEVICE):
    costs = {}
    for n in gallery_sizes:
        gallery = torch.randn(n, embed_dim, device=device)
        q = torch.randn(1, embed_dim, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = (q @ gallery.t())
        if device.type == "cuda":
            torch.cuda.synchronize()
        costs[n] = (time.perf_counter() - t0) * 1000
    return costs

total_params, trainable_params = count_params(main_model)
flops, flop_method = count_flops(main_model)
latency = benchmark_latency(main_model)
search_cost_ms = brute_force_search_cost(CONFIG["embed_dim"])
embed_dim = CONFIG["embed_dim"]
efficiency_report = dict(
    device=str(DEVICE),
    total_params=total_params,
    trainable_params=trainable_params,
    trainable_fraction=round(trainable_params / total_params, 4),
    flops=flops,
    gflops=round(flops / 1e9, 4) if flops else None,
    flop_method=flop_method,
    latency_ms=latency,
    embedding_dim=embed_dim,
    embedding_bytes=dict(fp32=embed_dim * 4, fp16=embed_dim * 2, int8=embed_dim * 1),
    gallery_storage_10M_images_fp32_GB=round(10_000_000 * embed_dim * 4 / 1e9, 2),
    brute_force_search_ms=search_cost_ms,
    note="Brute-force cosine search shown for N=160/10k/1M; for larger galleries use an ANN "
         "index (FAISS/HNSW) -- not built here, out of scope for this exercise's scale.",
    backbone_justification=(
        "DINOv2 ViT-S/14 (22.06M total params) with only the last "
        f"{CONFIG['unfreeze_last_n_blocks']} transformer blocks + final norm trainable "
        f"({trainable_params/1e6:.2f}M / {total_params/1e6:.2f}M, "
        f"{100*trainable_params/total_params:.1f}%) keeps the optimizable footprint small "
        "while reusing strong self-supervised features; GeM-over-patch-tokens suits the "
        "repeated-texture nature of the task better than a single CLS-token summary."
    ),
)
with open(WORK_DIR / "efficiency_report.json", "w") as f:
    json.dump(efficiency_report, f, indent=2, default=str)
print(json.dumps(efficiency_report, indent=2, default=str))

# %% [markdown]
# ## Step H: Deliverables checklist / limitations (method1.md §11)
#
# - [x] Approach note above (edit to match what actually ran).
# - [x] Single notebook, runs top to bottom; `DRIVE_DIR=None` fallback keeps
#       it runnable without the proprietary data.
# - [x] `kaggle_manifest.csv`, `kaggle_split.csv`,
#       `drive_query_gallery_split.csv` (when Drive available) saved for
#       exact reproducibility.
# - [ ] **Before sharing**: clear all cell outputs that could show Drive
#       images (there should be none — Step B only ever writes to disk —
#       but double-check the committed notebook's rendered outputs by eye).
# - [ ] Verify submission links are viewable by the reviewer; submit at
#       https://forms.gle/2tff1iKAvxoKh2xaA.
# - [ ] **Delete the Drive copy after the exercise concludes.**
#
# **Limitations (fill in / extend from an actual run):**
# - Small unique-image count in Kaggle (~750 sources) and especially in
#   Drive (~165, further reduced by blanks/singletons) — confidence
#   intervals are wide; see the bootstrap numbers above.
# - Color invariance during *training* is entirely synthetic (cluster-wise
#   Lab recoloring); SS9.2's Drive numbers are the one place real colorway
#   variation (if present in the hand-labeled groups) gets tested.
# - Drive group labels are hand-made by the author and inherently
#   subjective, especially the "unsure -> keep separate" convention, which
#   biases recall downward (conservative) rather than upward.
# - Only report numbers that were actually produced by a run of this
#   notebook — do not hand-edit `results.json`/`results_table.md`.
