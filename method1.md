# AGENT_SPEC: Color-Invariant Saree Design Recognition (AIE-CASE)

Hand this whole file to the coding agent. It is the single source of truth for what to build.
Environment: Kaggle Notebook (free GPU), PyTorch. Deadline: 11:59 AM IST, 2 Oct 2026.

---

## 0. Ground rules (read first, never violate)

1. **The Drive corpus (160 images) is proprietary.** Never redistribute it. Keep it in a PRIVATE Kaggle dataset. Never display Drive images in notebook outputs that reviewers or others can see (numbers and plots without images are fine). Delete the copy after the exercise.
2. **Never train on Drive images. Never tune the threshold, the checkpoint or any hyperparameter on them.** They are the final test set only.
3. **Disclose everything**: pretrained backbone (DINOv2 ViT-S/14, Meta, via torch.hub), datasets, any external data (none planned), every cleaning/relabeling/re-split step.
4. The author must be able to defend every choice in a live follow-up. Prefer simple, explainable choices. Comment the *why* in code.
5. Do not copy anyone else's solution. All work must be the author's own.
6. Reproducibility: fixed seeds, saved split files, pinned library versions printed at the top of the notebook.
7. **Proprietary data must never enter any AI tool's context.** The agent must NOT receive Drive images: no uploads, no screenshots, no images displayed in notebook outputs that an assistant can read. The agent writes code; the author runs it on Kaggle against the private Drive dataset and does the labeling by hand. The agent only ever sees numbers and file names, never pixels from Drive.
8. Everything in this file is a plan, not a verified result. The code snippets are untested reference implementations. Test them on Kaggle images before relying on them, and report only numbers that were actually produced by a run.

## 1. Task recap (from the PDF)

- Input: RGB saree image. Gallery: known designs, same design may appear in several colorways.
- **Identification**: rank gallery by similarity to the query, return best match(es).
- **Verification**: given two images, decide yes/no whether they carry the same design.
- **Core requirement**: same motif in different palettes MUST match; different motifs MUST NOT match even in identical palettes.
- Deliverables: (1) approach note, max 500 characters; (2) working end-to-end PyTorch code (training + inference), runnable as-is; (3) evaluation protocol with results and a clearly documented gallery/query split; (4) efficiency report (bonus): params, FLOPs and/or latency, embedding size.
- Assessed on: problem formulation, rigor of evaluation, code quality and reproducibility, clarity of reasoning, efficiency/novelty.

## 2. Data facts and open items (VERIFY, do not assume)

Known:
- **Kaggle `div456/indian-saree-patterns`**: 4 class folders (Banarasi, Bandhani, Ikat, Pichwai) under train/valid/test. Roughly 250-350 images per class. The author reports images appear **three times, byte-identical**. The class names are textile/craft types, so they are NOT design IDs (inference, check by eye on ~20 images).
- **Drive corpus**: 160 images, NO labels. The author looked at them: they contain repeating patterns with slight color shifts, blur, orientation changes, continuations/crops, and some blank images.

To check in the first 20 minutes and report in the notebook:
1. Count files per split/class; compute MD5 for all Kaggle files; report #unique vs total, and whether identical copies sit in different splits (train/valid/test).
2. Check image sizes, corrupted files, and near-blank images (low pixel std).
3. Look at the Drive filenames for any naming convention that hints at groups.

Decision: Kaggle = training + validation. Drive = final test only.

## 3. Pipeline overview

```
[A] Data prep -> [B] Drive grouping (manual) -> [C] Model -> [D] Recolor + batch sampler
   -> [E] Train -> [F] Evaluate (Kaggle val, then Drive test) -> [G] Efficiency -> [H] Write-up
```

Build in this order and get a **minimum viable submission early**: after step B, run the plain pretrained backbone through evaluation (step F) end to end. Then improve.

## 4. Step A: Kaggle data prep

1. Recursively list images from all train/valid/test class folders (ignore the original splits).
2. Drop unreadable files, images with short side < 96 px, and near-blank images.
3. **Exact dedupe**: MD5. Keep one file per hash.
4. **Near-duplicate grouping** (in case of crops/resizes): 64-bit DCT pHash; union-find files with Hamming distance <= 6 into one group (optionally also compare against the 8 rotations/flips). Each group gets a `source_id`.
5. Save `kaggle_manifest.csv`: `path, md5, source_id, class_name`.
6. **Re-split by `source_id`** (never by file): ~85% train / ~15% val, seed 0. Stratify by class_name if possible. Save `kaggle_split.csv`. Document the numbers in the notebook.
7. Cache: resize so the short side is 320 px, save as JPEG to `/kaggle/working/cache/` for fast loading.

Expected scale: only a few hundred unique images. Therefore fine-tune only a small part of the backbone, use strong augmentation, and watch for overfitting (monitor val retrieval, stop early).

## 5. Step B: Drive grouping (manual labeling, ~1 hour)

Goal: `drive_groups.csv` with `filename, group_id, is_blank`.

Helper script (agent builds it, author reviews):
1. Embed each Drive image with the **pretrained** DINOv2 (no training) on grayscale input, and also under 4 rotations (take the max similarity over rotations when comparing pairs).
2. Agglomerative clustering (cosine, average linkage) with a conservative threshold, so groups are pure but may be over-split.
3. Produce a contact sheet per cluster (a PNG written to disk only, for the author's private use; do NOT display it in a shareable notebook).
4. The author fixes groups by hand. Labeling rules (write these in the final evaluation section):
   - Same pattern with color shift, blur, rotation/flip: SAME group.
   - Different crop or continuation of the same repeating pattern: SAME group, but mark `variant=crop` if easy, so results can be broken down.
   - Blank images: `is_blank=1`, excluded from queries and (optionally) from the gallery. State the choice.
   - Pattern appears only once: singleton group; stays in the gallery as a distractor, never a query.
   - If unsure whether two images are the same design: keep them in different groups (conservative) and note it.

## 6. Step C: Model

- Backbone: `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')` (needs Internet ON in Kaggle; otherwise attach the weights as a Kaggle dataset). Fallback if unavailable: timm `convnext_tiny` or `resnet50` pretrained.
- **This backbone is a reasoned default, not a proven best.** Before the main training run, do a short bake-off: DINOv2 ViT-S/14, DINOv2 ViT-B/14, ConvNeXt-Tiny, ResNet-50, each with the same GeM + 256-d head, compared by the **Kaggle-validation protocol only (section 9.1), never on Drive** (choosing by Drive scores would contaminate the test). Keep ViT-S/14 unless another is clearly better per parameter. Record the comparison table; it is part of the efficiency and rigor story.
- **Freeze all but the last 4 transformer blocks** (and the final norm). Backbone LR 1e-5, head LR 1e-3.
- Pooling: **GeM over patch tokens** (p learnable, init 3). Rationale: the identity is a repeated texture/motif, so pooled local patch features suit it better than a single CLS token.
- Head: `Linear(384 -> 256)`, then L2 normalize. **Embedding dim = 256.**
- Input: 224x224, ImageNet mean/std.
- Inference: embed with `torch.no_grad()` and fp16; similarity = cosine = dot product of normalized vectors. Optional TTA (average embeddings over 4 rotations): evaluate as an ablation only, report its latency cost.

## 7. Step D: Color-invariance machinery (the core idea)

### 7.1 Recolor function (reference implementation, TEST IT on a few images first)

Operate in Lab. Keep L (structure/contrast of the motif), change only a,b. Cluster pixels into k colors, then map each cluster to a color of a target palette. Palette color j is paired with the cluster of j-th lightness rank, so one palette applied to different images yields a similar overall look.

```python
import numpy as np, cv2

def sample_palette(rng, k=5):
    """k target colors in OpenCV-Lab (a,b centered at 128). Includes near-neutral tones (gold/zari/white)."""
    hue = rng.uniform(0, 2*np.pi, k)
    chroma = rng.uniform(0, 60, k)
    return np.stack([128 + chroma*np.cos(hue), 128 + chroma*np.sin(hue)], 1).astype(np.float32)  # (k,2)

def recolor(img_rgb, palette, rng, gray_p=0.0, l_jitter=0.05):
    if rng.random() < gray_p:
        g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        return np.repeat(g[..., None], 3, 2)
    k = len(palette)
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    h, w, _ = lab.shape
    small = cv2.resize(lab, (64, 64), interpolation=cv2.INTER_AREA).reshape(-1, 3)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(small[:, 1:], k, None, crit, 3, cv2.KMEANS_PP_CENTERS)   # clusters on (a,b)
    ab = lab[..., 1:].reshape(-1, 2)
    d = ((ab[:, None, :] - centers[None]) ** 2).sum(-1)
    lab_idx = d.argmin(1)                                   # cluster per pixel
    # rank clusters by mean L, rank palette colors by their (random) order -> pair them
    L = lab[..., 0].reshape(-1)
    cl_L = np.array([L[lab_idx == c].mean() if (lab_idx == c).any() else 0 for c in range(k)])
    order = np.argsort(cl_L)                                # cluster ids from dark to light
    new_ab = ab.copy()
    for rank, c in enumerate(order):
        m = lab_idx == c
        shift = palette[rank] - centers[c]                  # move cluster centroid to palette color,
        new_ab[m] = ab[m] + shift                           # keep within-cluster variation
    out = lab.copy()
    out[..., 1:] = new_ab.reshape(h, w, 2)
    out[..., 0] = np.clip(out[..., 0] * (1 + rng.uniform(-l_jitter, l_jitter)), 0, 255)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)
```

Note: the palette's j-th entry should be assigned a lightness order. Simplest: sort the sampled palette by a random but fixed permutation per palette (store it as part of the palette). Keep it consistent within a batch.

Also add, with small probability, simple global hue rotation and channel permutation, and grayscale (p ~ 0.15) so the model also tolerates un-segmentable colorways.

Rationale to state in the write-up: a global hue shift is too easy because real colorways recolor regions (body vs. border) independently. Cluster-wise remapping mimics that.

### 7.2 Batch construction ("shared-palette batches")

- Each batch has B distinct source_ids (B = 48-64) and **V = 2 views per source**.
- Each batch draws a small **palette pool** (e.g. 6 palettes). Every view of every source gets a palette from this pool, with the constraint that **the two views of one source use different palettes**.
- Consequence: within a batch, many *different designs share the same palette* (they act as hard negatives for "same palette, different design"), and the *same design appears in different palettes* (positives). The model cannot solve the task with color.
- Views also get independent geometric augmentation: RandomResizedCrop(224, scale=(0.3, 1.0)) (this simulates "continuation/crop" variants of repeating patterns), random 90-degree rotations, flips, mild Gaussian blur (p=0.3), mild brightness/contrast jitter.
- Labels = source_id. Different crops of the same source share the label.

### 7.3 Loss

Supervised contrastive (SupCon), temperature 0.07, computed in fp32:

```python
import torch, torch.nn.functional as F
def supcon(z, labels, tau=0.07):
    z = F.normalize(z.float(), dim=1)
    n = z.size(0)
    sim = (z @ z.T) / tau
    self_mask = torch.eye(n, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, -1e9)
    pos = (labels[:, None] == labels[None, :]) & ~self_mask
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.sum(1) > 0
    loss = -(log_prob * pos).sum(1)[has_pos] / pos.sum(1)[has_pos]
    return loss.mean()
```

### 7.4 Augmentation spec (single source of truth; if anything above disagrees, this section wins)

Every augmentation exists because it imitates a variation actually seen in the Drive images (color shift, blur, orientation, crop/continuation, blank areas). Apply in this order, to training views only, starting from the cached 320px-short-side image:

| # | Transform | Setting | Imitates / why |
|---|---|---|---|
| 1 | RandomResizedCrop to 224x224 | scale (0.3, 1.0), ratio (3/4, 4/3). Reject a crop whose pixel std is below a small threshold (near-blank) and resample, max 5 tries. | Crops and continuations of repeating patterns; avoids training on blank patches |
| 2 | Random 90-degree rotation + horizontal flip | k uniform in {0,1,2,3}; flip p=0.5. Small-angle rotation is OFF by default (it adds padding artifacts). | Orientation changes |
| 3 | **Recolor (the key one)**: pick ONE mode per view | cluster-wise palette remap 70%, global hue rotation in the (a,b) plane 10%, RGB channel permutation 5%, grayscale 15% | Colorways. Palette remap imitates body/border dyed independently; the others cover cases that cluster remapping handles badly. |
| 4 | Gaussian blur | p=0.3, sigma in [0.1, 2.0] | Blur and focus differences |
| 5 | Brightness / contrast jitter | p=0.5, each within +/-15% | Lighting and exposure. Keep it mild; do NOT add strong saturation or hue jitter (the recolor step owns color). |
| 6 | Optional: Gaussian noise or JPEG re-compression | p=0.2, JPEG quality 60-95 | Capture and compression noise. Skip if short on time. |
| 7 | ImageNet mean/std normalization | fixed | Matches the pretrained backbone |

Rules that go with it:
- **Two views of the same source must differ in palette.** They must not both be grayscale, and for the palette-remap mode they must use different palettes from the batch pool (section 7.2). The three minor recolor modes (hue rotation, channel permutation, grayscale) are applied per view independently.
- Shared-palette batching (7.2) applies to the palette-remap mode.
- **Do NOT use by default:** random erasing/cutout (can erase the whole motif), perspective or elastic warps (distort motif geometry), crops smaller than 30% of the image (may contain only background).
- **Evaluation-time preprocessing = no augmentation.** Resize the short side to 224 and center-crop 224 (same for every method, baselines included), then normalize. Optional rotation TTA only as a separately reported ablation.
- **Validation palettes use a different RNG seed from training palettes**, so the Kaggle-val recolor test is not on the exact palettes seen in training.
- Run recolor inside DataLoader workers (k-means on a 64x64 thumbnail per view is cheap). Log the augmentation settings in the config JSON.
- Visually check about 20 augmented views per transform on Kaggle images before training. If recolored images look muddy or the motif is damaged, fix that before anything else.

## 8. Step E: Training

- AdamW, weight decay 0.05, cosine schedule with 2-epoch warmup, AMP (fp16).
- 30-60 epochs; "epoch" = one pass over all unique train sources. ~Few hundred images means this takes minutes per epoch at most.
- Every epoch: evaluate on the **Kaggle val split** (see 9.1). Keep the best checkpoint (by val Recall@1 under recolor queries). Early stop with patience 8.
- Save: best checkpoint, config JSON (all hyperparameters), training log CSV.
- Seeds: run the final config with 3 seeds (0, 1, 2) if time permits; report mean +/- std.
- Ablations (same budget, if time permits): (a) no recolor, only geometric aug; (b) recolor but NOT shared-palette batches (random palettes per view); (c) full method.

## 9. Step F: Evaluation protocol

All embeddings L2-normalized; similarity = cosine.

### 9.1 Kaggle validation (synthetic; for model selection and the verification threshold ONLY)

- Held-out val sources. Query = a recolored (different palette) + re-cropped view of a source. Gallery = the original views of ALL val sources, plus distractors: val sources recolored into the query's palette (palette-matched negatives).
- Metrics: Recall@1, Recall@5, mAP.
- Verification pairs: positives = (source, its recolored view); negatives = (source, different source), half of them palette-matched. Choose the **verification threshold** here (the threshold that maximizes balanced accuracy, or fixes FAR at 1%). Save it to `threshold.json`. This is the ONLY place the threshold is chosen.
- Label these results clearly as synthetic. Do not present them as the main result.

### 9.2 Drive test (main result; real colorways, hand-labeled groups)

**Fixed gallery/query split** (this is the "clearly documented split" the PDF asks for):
1. For every group (non-blank) with >= 2 images, choose ONE image as the query using `rng = np.random.default_rng(0)`.
2. Gallery = all other non-blank images (rest of the groups + singleton distractors). The query itself is never in the gallery.
3. Save `drive_query_gallery_split.csv` (filename, role) so the split is exactly reproducible.
4. Relevant items for a query = gallery images of the same group.

Identification metrics: Recall@1, Recall@5, mAP (use `average_precision_score` per query over the full gallery ranking). Also report a secondary **leave-one-out** version (every image with >= 1 same-group partner takes a turn as query) to use all data.

Verification: all image pairs among non-blank Drive images. Positive = same group, negative = different group. Metrics: **ROC-AUC, EER, TAR@FAR=1%** (threshold-free), plus balanced accuracy and F1 at the threshold from `threshold.json` (never tuned on Drive). Remember the heavy class imbalance (most pairs are negative); do not report plain accuracy.

### 9.3 Color-invariance stress tests (this is what tests the core requirement)

1. **Color-adversarial verification subset**: compute a color distance for each pair (e.g. chi-square distance between 3D Lab histograms with 8 bins per channel). Positives = same-group pairs in the TOP quartile of color distance (same design, very different palettes). Negatives = different-group pairs in the BOTTOM quartile (different design, similar palettes). Report AUC on this subset. A color-reliant model collapses toward 0.5.
2. **Similarity vs. color distance** plot (positives and negatives separately; scatter or binned means). No Drive images shown, only numbers.
3. **Recolor probe** on held-out Kaggle val images: mean cosine similarity between each image and several recolored versions of it, plus Recall@1 when the recolored image is the query.
4. (Optional) breakdown of positive pairs by variation type (color / rotation / blur / crop-continuation) if the author tagged them.

### 9.4 Baselines (same protocol, same split)

1. Random embeddings (chance floor).
2. Color histogram only (it should fail on the adversarial subset; if it does well there, the labels or the subset are wrong, investigate).
3. Pretrained DINOv2 + same GeM/head setup but untrained head (or CLS features), RGB input.
4. Same, with grayscale input.
5. Ablations from step E.
6. Final model.

### 9.5 Statistics

- Bootstrap over queries (1000 resamples) to give 95% CIs for Recall@1 and mAP; bootstrap over pairs for AUC.
- With 160 images, small differences (about 2 points) are noise; do not over-claim.
- Report which numbers come from synthetic vs. real data.

### 9.6 Output files

`results.json`, `results_table.md` (one table: methods x metrics), `plots/` (numbers/plots only; no Drive images), plus the saved split files.

## 10. Step G: Efficiency report

For the final model (image 1x3x224x224):
- Parameter count: total and trainable (`sum(p.numel())`).
- FLOPs: `torch.utils.flop_counter.FlopCounterMode` (report as MACs/FLOPs, say which).
- Latency: batch size 1 and 32, GPU and CPU, with 10 warmup + 100 timed runs, `torch.cuda.synchronize()`, report median and p95.
- Embedding size: 256 float32 = 1 KB per image; also report the fp16 and int8 sizes, and the storage for a 10M-image gallery (rough: 10M x 256 x 4 B = ~10 GB fp32).
- Search cost: brute-force matrix multiply `(1 x 256) @ (256 x N)` time for N = 160, 10k, 1M (random vectors) to show scalability; mention FAISS/HNSW for larger N, but do not build it.
- Justify choices: ViT-S backbone (22M params) with only the last 4 blocks trained, 256-d embedding. Compare against a tiny alternative if time allows (e.g. a ResNet-18/MobileNet baseline) to show the efficiency/accuracy trade-off.

## 11. Step H: Deliverables checklist

- [ ] `approach_note.txt`: max 500 characters (below, 447 characters, verified).
- [ ] Notebook (single, runs top to bottom), sections: Setup/versions, Data prep, Drive grouping (private), Model, Recolor demo (use Kaggle images only), Training, Evaluation, Baselines, Efficiency, Limitations, Disclosure.
- [ ] Config switch `DRIVE_DIR = None` -> notebook still runs end to end without the proprietary data (skips Drive cells with a clear message). This keeps it runnable as-is for the reviewer.
- [ ] Clear all outputs that show Drive images before sharing. Verify the links are viewable by the reviewer before the deadline. Submit at https://forms.gle/2tff1iKAvxoKh2xaA.
- [ ] Limitations paragraph: small unique-image count in Kaggle; synthetic recolor is a proxy; Drive labels are hand-made and subjective; only 160 test images, wide confidence intervals.
- [ ] After submission: delete the Drive copy.

### Approach note (447 characters)

```
DINOv2 ViT-S/14 (last 4 blocks tuned), GeM pooling, 256-d L2-normed head; cosine ranking, verification via val-tuned threshold. Pre: 224px resize, ImageNet norm. Train: SupCon/InfoNCE (tau=0.07) on deduped Kaggle images, 2 views each; palette-grouped batches (histogram-matched shared palettes give same-palette hard negatives; positives always recoloured); aug: k-means/hue/gray recolour, rotation, crop, blur. Drive images used only for testing.
```

Edit the wording to match what is actually implemented (for example, the note says "histogram-matched"; the reference code uses cluster-wise palette remapping, so change it to "cluster-recoloured" if that is what is built). The note must describe the final code, not the plan.

## 12. Kaggle practicalities

- Add data via "Add Data": the Kaggle saree dataset, and the private Drive dataset.
- Turn Internet ON only for `torch.hub`/pip, or attach DINOv2 weights as a Kaggle dataset to avoid it.
- GPU: T4/P100; batch of 128 images at 224 px with a ViT-S and 4 unfrozen blocks fits comfortably under AMP.
- Write outputs to `/kaggle/working`. Save checkpoints each epoch (sessions can die).
- Before submitting, run "Save & Run All (Commit)" once to prove it runs top to bottom, and note total runtime.

## 13. Time plan (from ~02:00 IST; deadline 11:59 IST)

| Time | Goal |
|---|---|
| 02:00-03:00 | Step A (dedupe/split) + notebook skeleton; baseline pretrained eval harness written |
| 03:00-04:30 | Step B Drive grouping (author labels); baseline results on Drive = first working submission |
| 04:30-07:00 | Recolor + sampler + training; Kaggle-val monitoring |
| 07:00-09:00 | Ablations, seeds, color-adversarial tests, plots |
| 09:00-10:30 | Efficiency report, write-up, approach note, cleanup, outputs cleared |
| 10:30-11:30 | Commit run, check links viewable, submit. Do not leave submission to the last minutes. |

Rest is allowed and wise; a tired author cannot defend decisions live.

## 14. Things the agent must NOT do

- Do not assume the Kaggle class folders are design IDs.
- Do not use the original Kaggle train/valid/test folders as splits.
- Do not train, select checkpoints or tune thresholds on Drive images.
- Do not print/display/redistribute Drive images in shareable outputs.
- Do not report synthetic-validation numbers as the main result.
- Do not claim results that were not actually run. If something is not finished, say so in the Limitations section.