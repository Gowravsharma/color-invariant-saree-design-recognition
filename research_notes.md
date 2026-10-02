# Research & Dataset Analysis Notes — AIE-CASE Color-Invariant Saree Design Recognition

Pre-solution research and data analysis, answering each item in `need_to_look.txt`
before any model code was written. Dated 2026-10-02.

---

## 1–3. Why is this an issue / existing solutions / gaps vs. our requirements

**Headline finding:** no public dataset or benchmark exists for "same saree
motif, different dye colorway" matching. Across every search below, nothing
solves this exact problem — the closest work either solves a *different*
invariance (illumination, not colorway) or a *different* identity target
(body shape, not surface motif). The missing piece is not an algorithm, it's
data: no one has a labeled "same design, different colorway" saree dataset,
including the two we were given.

### Closest analog #1 — classical color/illumination-invariant texture retrieval
Hand-engineered color-invariant descriptors (Gevers & Smeulders-style color
invariants, hue–saturation histograms) were explored through the 2000s.
Current literature on this explicitly says the approach is "becoming
obsolete in favor of features automatically learned... with deep learning."
More importantly, this body of work solves *illumination* invariance (same
dye, different light/angle), not *colorway* invariance (deliberately
different dye) — a different problem from ours.
- [Illumination Invariant Texture Retrieval](https://ieeexplore.ieee.org/document/1699519/)
- [combining color and shape invariant features for image retrieval](https://staff.fnwi.uva.nl/th.gevers/pub/GeversIP00.pdf)
- [Illumination color covariant locale-based visual object retrieval](https://www.sciencedirect.com/science/article/abs/pii/S0031320301001637)
- [Evaluating color texture descriptors under large variations of controlled lighting conditions](https://arxiv.org/pdf/1508.01108)

### Closest analog #2 — cloth-changing person re-identification
This field actively solves "make an embedding invariant to a salient,
variable surface property while preserving identity," via shape/silhouette
extraction, contour sketches, and causal/mutual-information disentanglement
to strip clothing-color bias (e.g. MAC-DIM maximizes mutual information
between color-appearance and shape features specifically to remove clothing
color bias). It is **inverted** from our case: person re-ID discards
clothing color *and* pattern entirely and keeps body shape; we must discard
only color and explicitly *keep* the pattern/motif. The invariance-training
technique (disentangle an identity-irrelevant nuisance factor via
augmentation/adversarial/causal methods) transfers conceptually; the feature
target does not.
- [Towards Robust Person Re-Identification: Learning Invariant Features Under Clothing Changes and Occlusions](https://uh-ir.tdl.org/items/51d6a8b4-fa00-4691-abce-24c3ac7c0323)
- [Clothes-Invariant Feature Learning by Causal Intervention for Clothes-Changing Person Re-identification](https://www.researchgate.net/publication/370656488_Clothes-Invariant_Feature_Learning_by_Causal_Intervention_for_Clothes-Changing_Person_Re-identification)
- [Learning Shape Representations for Person Re-Identification under Clothing Change](https://openaccess.thecvf.com/content/WACV2021/papers/Li_Learning_Shape_Representations_for_Person_Re-Identification_Under_Clothing_Change_WACV_2021_paper.pdf)
- [CLIP-Driven Cloth-Agnostic Feature Learning for Cloth-Changing Person Re-Identification](https://arxiv.org/pdf/2406.09198)

### Closest analog #3 — logo retrieval (strongest positive signal)
Production logo-retrieval systems successfully use metric learning
(triplet/proxy-based loss) to match the same logo across its official
colorway variants — direct validation that metric learning is the right
algorithm family for "same graphic identity, different color." Caveat:
logos are flat, crisp vector graphics; saree motifs are woven/printed into
deformable, textured fabric with folds, sheen, and repeat-pattern
periodicity — a materially harder visual domain.
- [Multi-Label Logo Recognition and Retrieval based on Weighted Fusion of Neural Features](https://arxiv.org/html/2205.05419v1)
- [Segment Augmentation and Differentiable Ranking for Logo Retrieval](https://arxiv.org/pdf/2209.02482)
- [Scalable Logo Recognition using Proxies](https://assets.amazon.science/60/f0/50c9d1d04a8cacecdc7c1c74f625/scalable-logo-recognition-using-proxies.pdf)

### Indian-textile-specific prior art
- **Content-based image retrieval of Indian traditional textile motifs using
  deep feature fusion** ([Nature Sci. Reports / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11063031/))
  — read in full. Uses InceptionV3 (2048-d) + InceptionResNetV2 (1536-d)
  features, concatenated to 3584-d, retrieved via Manhattan-distance top-20
  on their own **TIAD** dataset (22,547 images, 9 styles: Bagh, Bandhani,
  Batik, Chikankari, Ikat, Kalamkari, Kashida, Madhubani, Warli). **Confirmed
  by reading the full text: this paper does not address color/colorway
  invariance at all.** Clean, citable confirmation that our specific
  requirement is a genuine open gap, not something solved off-the-shelf —
  and that this prior work also doesn't train metric-learned embeddings or
  report identification/verification metrics (precision/recall on top-20
  only).
- **Fine-grained textile provenance recognition** ([SSRN abstract](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7027167),
  full text paywalled/403) — abstract snippet frames Indian saree
  classification as fine-grained recognition (similar motifs/pallus/borders
  across provenances, wide intra-class variation within a provenance) and
  names CNN/transformer, **metric and angular-margin learning**, and graph
  neural networks as the relevant approach families. Independent support for
  the angular-margin (ArcFace-style) direction, though it doesn't
  specifically discuss colorway invariance in the available snippet.
- Other hits (titles only, not read in depth): *HybridWeaveNet* (dual
  attention + EfficientNetV2 for handloom heritage fabric pattern
  recognition), *STD-net*/*Sareenet* (saree texture detection/classification
  for e-commerce) — all classification-oriented, not colorway-invariant
  identity matching.

### ⚠️ Integrity flag
A public GitHub repo titled **`color-invariant-saree-design-recognition`**
turned up, described as "dataset preparation, PyTorch embedding training,
and evaluation for a color-invariant saree design retrieval and
verification task" — almost certainly another candidate's submission to
this identical DeepLure brief, now public. **Not opened beyond the search
snippet; not used as a reference.** Given the brief's "all work must be your
own" rule, this should stay off-limits — don't view it either.

### Net takeaway for approach
Use metric learning (validated by the logo-retrieval analog), borrow the
*invariance-training methodology* from cloth-changing re-ID (train the
nuisance factor — color — out via strong, explicit augmentation rather than
hoping it's ignored), and accept upfront that there is no real
same-design/different-colorway ground truth anywhere in the provided data —
so color invariance must be trained and evaluated against a synthetic proxy
(see §9), disclosed as a limitation.

---

## 4. Dataset balance / imbalance / minority classes

After de-duplicating Kaggle's Roboflow salt-and-pepper triplicate
augmentation back down to unique source photographs (grouped by
`<class>/<base-filename-before-the-_jpg.rf.<hash> suffix>`):

| Class | Unique source photos |
|---|---|
| Banarasi | 233 |
| Pichwai | 183 |
| Ikat | 168 |
| Bandhani | 166 |
| **Kaggle total** | **750** |
| `handloom_sarees` (DeepLure, unlabeled) | 165 |
| **Combined total** | **915** |

Max/min class ratio = **1.41×** — mild imbalance, nothing requiring
reweighting or oversampling. `handloom_sarees` has **zero metadata** — no
class/weave label of any kind, so "balance" can't be assessed for it;
it's treated as its own pool of 165 design identities.

Verified no train/valid/test leakage in Roboflow's original split (checked
by grouping all three splits' files by deduplicated base name — zero source
photos span more than one split).

Note: "minority class" framing applies less than it would for a classifier,
since training is **identity-level** (each of the 915 unique photos is its
own class for the metric-learning head), not 4-way weave-style
classification. The 4 Kaggle classes only matter for (a) stratifying the
train/val/test identity split so each split has proportional weave-style
coverage, and (b) an optional sanity-check retrieval-by-style readout.

---

## 5. Hard images

- **Label-purity issue found in Kaggle's "Pichwai" class**: visually
  confirmed at least one source photo is a framed, circular **wall-decor
  painting** (lotus motif on canvas, studio-shot on plain gray background)
  rather than a saree-fabric close-up — same motif tradition (Pichwai is
  historically a painting style from Nathdwara, also used as a saree-print
  name), different product category entirely. Quantified via filename
  search (`wall`/`decor`/`frame`): ~4 files (≈1–2 unique source photos after
  dedup) out of 321 Pichwai files / 183 unique Pichwai identities — small
  numerically (<1%), but worth disclosing, since it's an outlier that could
  become a confusing hard-negative or skew what "Pichwai" means to the
  model.
- Outside that, a broad visual sample (handloom + all 4 Kaggle classes)
  found the large majority of both datasets to be clean, tight close-up
  fabric crops with no people, bodies, or background clutter — good news:
  no saree segmentation / person-detection preprocessing step is needed.
- Not yet done: a systematic hard-*pair* audit (e.g., visually similar
  motifs across different weave classes that could become hard negatives,
  such as fine Ikat striping vs. Bandhani dot grids). Recommended to do this
  as a **post-baseline error-analysis pass** — run a first trained model,
  inspect its highest-similarity cross-identity false matches — rather than
  guessing hard pairs upfront without a model to validate the guess against.

---

## 6. Blur (camera-movement preprocessing question)

Quantitative pass: Laplacian-of-grayscale variance (standard blur-detection
proxy), computed after resizing every image's longer side to 512px for a
consistent scale, over all 915 unique identities. Script:
`submission/analysis/dataset_analysis.py`; full per-image scores:
`submission/analysis/blur_report.json`.

**Distribution:** min 8.9 · p5 44.0 · p10 79.2 · median 394.2 · p90 1390.1 ·
max 5964.1 (dimensionless Laplacian-variance units).

**Headline finding:** the bottom 10%-sharpest bucket (92 images,
dataset-wide) is dominated by `handloom_sarees` — **57 of 165 handloom
images (35% of the entire DeepLure corpus)** fall into that bucket, versus
Kaggle's worst-affected classes contributing far smaller fractions of their
own class sizes (18/233 Banarasi, 9/166 Bandhani, 8/168 Ikat, 0/183
Pichwai in the bottom decile). **The DeepLure corpus is disproportionately
blurrier than the Kaggle set.**

**Visual check of the worst offenders** (manually inspected several of the
lowest-scoring handloom images) found this is a **mix of two different
things**, which the single Laplacian-variance number conflates:
1. **Genuine motion/focus blur** — fuzzy, double-edged stripe/weave
   boundaries consistent with handheld camera shake (e.g. `img_677180.jpg`).
2. **Plain/sparse-pattern fabric** — large flat near-white regions with only
   a few small embroidered motifs (e.g. `img_393000.jpg`,
   `img_889324.jpg`). These score "low texture" on a Laplacian-variance
   metric without actually being camera-blurred — the motifs present are
   still reasonably crisp.

So the raw 35% figure likely **overcounts** true motion blur. Recommended
next step: a quick manual pass over just the 57 flagged handloom images
(small, tractable by hand) to split genuine blur from plain fabric before
deciding what to do with each.

**Recommended handling once separated:**
- *True motion blur*: don't attempt deblurring (risks destroying the fine
  motif edges the task depends on, and isn't worth the time budget).
  Instead add blur as a **training-time augmentation** (light Gaussian blur
  / downsample-upsample), since a real deployment query photo will
  sometimes be blurry too — train for robustness rather than trying to fix
  the input.
- *Plain/sparse fabric*: not a preprocessing problem at all — it's
  legitimately less informative. Lower retrieval recall on these identities
  is an honest result to expect and report, not a bug to chase.

---

## 9. Evaluation & benchmarking strategy

Given there is no real-world colorway ground truth anywhere in the provided
data (§1–3), color invariance can only be trained and *measured* against a
**synthetic recoloring proxy** — this must be stated as an explicit,
disclosed limitation of the evaluation, not hidden.

- **Split**: open-set, identity-level (each of the 915 unique photos is a
  distinct "design identity"). Entire identities — never just images — are
  held out for val/test, stratified by weave class for Kaggle and
  independently for the unlabeled handloom pool (70/15/15). This is the
  standard face-recognition-style open-set protocol: test identities are
  never seen during training or checkpoint selection.
- **Identification**: Rank-1, Rank-5 accuracy and mAP against a gallery of
  one canonical-color image per test identity, queried with synthetically
  recolored renders. With exactly one relevant gallery item per identity,
  this mAP is mathematically equivalent to **Mean Reciprocal Rank** — worth
  being precise about in the write-up so it isn't mistaken for standard
  multi-relevant-item mAP.
- **Verification**: ROC-AUC, EER, and TAR@FAR(1%)/TAR@FAR(0.1%) over
  same-identity vs. different-identity embedding-similarity pairs drawn from
  the recolored query pool.
- **Color-invariance stress test** (makes the brief's core requirement
  directly measurable): sweep synthetic recoloring strength from 0 (trivial,
  same-color sanity check) to 1 (full recolor) on queries only, gallery
  held at canonical color, and report Rank-1/mAP at each level. A genuinely
  invariant embedding should degrade gracefully rather than collapse the
  moment any recoloring is applied.
- **Caveat restated**: all of the above measures invariance to *synthetic*
  HSV/channel recoloring — a reasonable but imperfect proxy for genuine
  textile dye-lot variation, since real fabric can shift local sheen/shadow
  in ways simple color-space transforms don't fully capture.

---

## Items left blank in `need_to_look.txt`
Items 7 and 8 were empty in the checkpoint file at the time of this
analysis — nothing to address there yet.
