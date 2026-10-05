# Method overview

This document describes the pieces of CSCS-AL implemented in this repository, and the
interface contract for the one piece that is *not* bundled here (the provenance-weighted
nnU-Net trainer).

## 1. Cold-start (`cscs_al_partial.cold_start`)

Given SSL embeddings and two label-free scalar signals per volume — typicality `T(x)`
(inverse mean k-NN distance in embedding space) and reconstruction difficulty `U_ssl(x)`
(mean masked-reconstruction variance under a frozen SSL encoder) — `CSCSColdStart`:

1. Computes the **Difficulty-Coverage Ratio**, `DCR = Spearman(U_ssl, T)` over the whole
   pool. This is a single dataset-level number: positive DCR means volumes the SSL encoder
   finds hard to reconstruct also tend to be typical (dense in embedding space); negative
   DCR means hard volumes tend to be peripheral/outliers.
2. Runs k-means++ with `k = K0` clusters on standardized embeddings.
3. Within each cluster, percentile-ranks `T` and `U_ssl` locally and picks the volume
   maximizing `T_local^(1-gamma0) * U_local^gamma0`, with `gamma0 = 0.5` always (the
   cold-start pacing is DCR-adjusted through the `gamma` formula, but at round 0 the
   annotation-progress factor is 0, so gamma0 reduces to the neutral 0.5 regardless of DCR).

`RandomColdStart` and `ProbCoverColdStart` are included as baselines.

## 2. Iterative acquisition (`cscs_al_partial.acquisition`)

At round `r >= 1`, once a task model `f_r` exists:

- `compute_useg` computes a **volume-level segmentation uncertainty**: mean voxelwise
  entropy over a dilated predicted-foreground mask (dilation: a full cube, Chebyshev
  distance, not a Euclidean ball — see the function docstring for the exact structuring
  element), falling back to the mean over the whole volume when the dilated foreground is
  smaller than `min_fg_voxels`. This is a mean over the (dilated) ROI, not a top-percentile
  subset of voxels within it -- an unrelated top-10%-of-voxels aggregation exists elsewhere
  in the original codebase (used by a simpler entropy-only baseline) and is not part of the
  CSCS-AL score shipped here.
- Typicality `T_r` is **not** recomputed against the shrinking residual pool: it is the
  same `T(x)` array computed once from embeddings before cold-start (KNN distances against
  the *full* original pool), simply indexed down to the volumes still in the residual pool
  at round `r` (you supply this by subsetting your own precomputed `T_ssl` array, e.g. as
  `run_iterative_round.py` does). This matches the actual acquisition code path that
  produced the paper's results, which reuses the frozen cold-start `T_ssl` array every
  round rather than rerunning KNN on a shrinking neighbor set each time.
- `compute_gamma_fixed_dcr` (or `compute_gamma_dynamic` for the per-round ablation
  variant) turns the fixed cold-start DCR plus how much of the pool is already labeled
  into a pacing weight `gamma_r in [0.3, 0.7]`.
- `compute_iterative_scores` combines the two signals: both are **percentile-rank**
  normalized (average ranks, divided by the residual pool size) before being combined as
  `T_ranked^(1-gamma_r) * U_seg_ranked^gamma_r`. This is a geometric mean of ranks, not a
  min-max-normalized combination — the distinction matters for how outlier-robust the
  resulting ranking is.
- `ConvergenceChecker` decides when to stop querying. Its default `mode="supervised"`
  (`update()`) stops when round-to-round *validation Dice* changes by less than a
  threshold for a set number of consecutive rounds (comparing consecutive rounds, not the
  running best), when the pool is exhausted, or at a hard round cap. This needs a held-out
  labeled validation set — see §6 for what to use when you don't have one.

## 3. Partial correction (`cscs_al_partial.partial_correction`)

For each volume queried at round `r >= 1`:

1. `rank_patches_by_entropy` partitions the volume into non-overlapping cubic patches and
   ranks them by mean voxelwise entropy of the model's current prediction.
2. `apply_partial_correction` reveals the reference label on the top-`b%` patches by
   **patch count** (not by cumulative voxel count — see the module docstring for what this
   means for the realized vs. nominal corrected fraction) and keeps the model's prediction
   elsewhere, producing a provenance map over four states:
   - `HUMAN_FULL` — cold-start volumes, fully expert-annotated.
   - `HUMAN_CORRECTED` — the revealed top-`b%` patches.
   - `MODEL_UNCHECKED` — everything else in a partially-corrected volume.
   - `EXCLUDED` — reserved for voxels you want to zero-weight for any other reason (e.g.
     unreliable edge patches); `apply_partial_correction` does not populate this state
     itself, it is provided for you to set based on your own criteria before building the
     weight map.
3. `build_weight_map` turns the provenance map (plus the model's per-voxel confidence) into
   a continuous `[0, 1]` weight: 1.0 for human-derived voxels, `alpha * g(confidence)` for
   `MODEL_UNCHECKED` (0 if `alpha=0`), 0 for `EXCLUDED`, with
   `g(c) = max(0, (c - tau_low) / (1 - tau_low))`.

## 4. Provenance-weighted training — interface contract (trainer released separately)

The weight map produced above is meant to be injected as a **second channel** alongside
the label map before nnU-Net preprocessing (label channel 0, weight channel 1, weight
encoded as an integer `round(w * 1000)` so it survives the same on-disk dtype as the
label). A custom nnU-Net v2 trainer then:

- Detects the extra channel on the first training batch.
- Splits label and weight *before* the loss (a plain Dice loss would otherwise try to
  one-hot-encode the 2-channel target and fail).
- Computes cross-entropy normalized by the *summed* per-voxel weight (so revealing more
  supervision in a volume does not, by itself, rescale the magnitude of the CE term), and
  restricts the Dice term to `HUMAN_FULL` / `HUMAN_CORRECTED` voxels only.
- Handles deep supervision by downsampling the label channel per scale as usual, while
  keeping the weight channel at full resolution only for the finest scale (coarser scales
  get a uniform weight of 1.0 — low-resolution supervision heads do not benefit from
  patch-level weighting).

This trainer (and its Ranger21-based optimizer/scheduler) is released as a separate
artifact; this repository is self-contained for everything upstream of it (deciding which
volumes to query and which patches to correct).

## 5. Evaluation (`cscs_al_partial.evaluation`)

Dice and 95th-percentile Hausdorff distance, with an explicit, documented convention for
how empty masks are handled and how physical voxel spacing must be passed in (same axis
order as the array — see the module docstring; an axis-order bug in an earlier internal
version silently swapped in-plane and through-plane spacing for anisotropic volumes, and
is fixed in this implementation).

## 6. Deployment scenarios: with vs. without ground truth

Every number reported in the paper comes from `mode="supervised"` (§2): a fixed, labeled
validation set exists for every dataset, and `ConvergenceChecker.update()` uses it to know
when to stop. That assumption does not hold when you point this method at a genuinely new,
unlabeled dataset — there is no Dice to compute yet. Pick one of the following depending on
what you actually have:

**A. You have volumes + reference masks, and want to *evaluate* the method**
(reproduce/benchmark, as in the paper). Use `mode="supervised"`. Nothing else changes.

**B. You have raw, unlabeled volumes and want to *use* the method to build an annotated
dataset from nothing.** Use `mode="stability"` or `mode="entropy"` (`ConvergenceChecker.
update_proxy()`). Both are label-free and reuse signals already computed for acquisition
— no extra inference pass, no annotation spent purely on monitoring:

  - `mode="stability"`: stop once the task model's own hard segmentations on the residual
    pool stop changing between rounds (`compute_prediction_agreement`, called by your loop
    with each round's `predictions_dir`). Recommended default — easier to reason about
    ("the model stopped changing its mind") and less sensitive to small-pool noise than a
    raw entropy trend.
  - `mode="entropy"`: stop once the mean `U_seg` you already compute for acquisition (§2)
    stops moving round-to-round. Slightly cheaper (no need to keep predictions around
    across rounds to compare), more prone to misleading local plateaus.

  Trade-off: you get an annotated dataset and a principled stopping round, but **no
  quantitative accuracy guarantee** — there is nothing to compute Dice against, by
  construction. You know *when* to stop, not *how good* the result is.

**C. Recommended starting point for most real deployments**: hand-label a small, fixed
"trust set" of 5-10 volumes up front, held out from the pool the AL loop actually queries,
and use `mode="supervised"` against that set. This is a small, one-time annotation cost —
far less than annotating the whole dataset — in exchange for getting the real ΔDice
stopping rule (A) back, on a dataset that otherwise starts fully unlabeled (B). This is
what we would suggest to someone asking "how do I start using CSCS-AL on my own data."

`ConvergenceChecker` itself is agnostic to which scenario you're in: construct it with
`mode="supervised"` (A, C) or `mode="stability"`/`"entropy"` (B), and call `update()` or
`update_proxy()` accordingly — see the class docstring in `acquisition/convergence.py`.
