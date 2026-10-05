"""
Sliding-Window Patch Correction (E2 / E2-OPT).

Given a selected volume, ranks non-overlapping 3D patches by segmentation
entropy and simulates human correction by revealing reference labels in
the top-b% most uncertain patches; the rest retain the model's prediction.

Provenance categories (from partial_correction.provenance):
    HUMAN_CORRECTED : patch in the corrected top-b%
    MODEL_UNCHECKED : patch not selected for correction
    HUMAN_FULL      : used for cold-start volumes (not produced here)

Note on the budget semantics: `budget` here is a *patch-count* fraction
(`round(budget * n_patches)`), not a voxel-weighted cumulative-sum budget.
Because patches near volume edges or containing little tissue can have a
different voxel count than interior patches, the realized fraction of
*voxels* corrected can differ from -- and in practice tends to exceed --
the nominal `budget`. If you need an exact voxel-budgeted variant, rank by
entropy as below but accumulate `n_voxels` until the cumulative sum
reaches `budget * total_voxels` instead of counting patches.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from .provenance import (
    HUMAN_CORRECTED,
    MODEL_UNCHECKED,
    make_provenance_map_from_masks,
)


# ══════════════════════════════════════════════════════════════════════
# PATCH RANKING
# ══════════════════════════════════════════════════════════════════════

def rank_patches_by_entropy(
    entropy_map: np.ndarray,
    patch_size: int = 32,
) -> List[Dict]:
    """
    Divide the volume into non-overlapping 3D patches and rank by mean entropy.

    Args:
        entropy_map : (D, H, W) float32, voxelwise entropy
        patch_size  : side length of the cubic patch (voxels)

    Returns:
        List of dicts, sorted by score descending (most uncertain first):
        [
          {"score": float, "slices": (slice_d, slice_h, slice_w),
           "n_voxels": int},
          ...
        ]
    """
    D, H, W = entropy_map.shape
    patches = []

    for d0 in range(0, D, patch_size):
        for h0 in range(0, H, patch_size):
            for w0 in range(0, W, patch_size):
                sd = slice(d0, min(d0 + patch_size, D))
                sh = slice(h0, min(h0 + patch_size, H))
                sw = slice(w0, min(w0 + patch_size, W))
                patch = entropy_map[sd, sh, sw]
                patches.append({
                    "score":   float(patch.mean()),
                    "slices":  (sd, sh, sw),
                    "n_voxels": patch.size,
                })

    patches.sort(key=lambda p: p["score"], reverse=True)
    return patches


def filter_patches_with_foreground(
    patch_ranking: List[Dict],
    softmax: np.ndarray,
    min_fg_voxels: int = 10,
    fg_labels: Optional[List[int]] = None,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split a patch ranking into foreground-containing and background-only
    patches, useful if you want to bias correction away from empty patches.

    Args:
        patch_ranking  : output of rank_patches_by_entropy (sorted desc)
        softmax        : (C, D, H, W) softmax probabilities
        min_fg_voxels  : minimum predicted foreground voxels in patch
        fg_labels      : foreground label indices (default: all > 0)

    Returns:
        (fg_patches, bg_patches) -- both sorted by score descending
    """
    pred = np.argmax(softmax, axis=0)  # (D, H, W)

    fg_patches, bg_patches = [], []
    for patch in patch_ranking:
        sd, sh, sw = patch["slices"]
        patch_pred = pred[sd, sh, sw]
        if fg_labels is not None:
            fg_count = int(np.isin(patch_pred, fg_labels).sum())
        else:
            fg_count = int((patch_pred > 0).sum())
        if fg_count >= min_fg_voxels:
            fg_patches.append({**patch, "n_fg_voxels": fg_count})
        else:
            bg_patches.append({**patch, "n_fg_voxels": fg_count})

    return fg_patches, bg_patches


# ══════════════════════════════════════════════════════════════════════
# PARTIAL CORRECTION
# ══════════════════════════════════════════════════════════════════════

def apply_partial_correction(
    prediction: np.ndarray,
    reference: np.ndarray,
    patch_ranking: List[Dict],
    budget: float = 0.20,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Simulate human correction of the top-K uncertain patches.

    The annotator corrects patches in ranked order (most uncertain first)
    until the budget fraction of *patches* has been corrected. Corrected
    voxels receive the reference label; uncorrected voxels keep the model
    prediction.

    Args:
        prediction    : (D, H, W) int array -- model's predicted label map
        reference     : (D, H, W) int array -- reference (ground-truth) label map
        patch_ranking : output of rank_patches_by_entropy or
                        filter_patches_with_foreground (sorted desc)
        budget        : fraction of patches to correct, in (0, 1]

    Returns:
        mixed_label      : (D, H, W) int -- corrected volume label map
        provenance_map   : (D, H, W) int8 -- per-voxel provenance category
        stats            : dict with correction cost metrics
    """
    mixed_label = prediction.copy()
    n_total = len(patch_ranking)
    n_to_correct = max(1, int(round(budget * n_total)))

    corrected_mask   = np.zeros(prediction.shape, dtype=bool)
    uncorrected_mask = np.zeros(prediction.shape, dtype=bool)

    for i, patch in enumerate(patch_ranking):
        sd, sh, sw = patch["slices"]
        if i < n_to_correct:
            mixed_label[sd, sh, sw] = reference[sd, sh, sw]
            corrected_mask[sd, sh, sw] = True
        else:
            uncorrected_mask[sd, sh, sw] = True

    provenance_map = make_provenance_map_from_masks(
        volume_shape=prediction.shape,
        human_full_mask=np.zeros(prediction.shape, dtype=bool),
        corrected_mask=corrected_mask,
        excluded_mask=np.zeros(prediction.shape, dtype=bool),
    )

    n_voxels_corrected = int(corrected_mask.sum())
    n_patches_with_fg = sum(
        1 for p in patch_ranking[:n_to_correct]
        if p.get("n_fg_voxels", 1) >= 1
    )
    slices_touched = set()
    for patch in patch_ranking[:n_to_correct]:
        sd = patch["slices"][0]
        slices_touched.update(range(sd.start, sd.stop))

    stats = {
        "n_patches_total":      n_total,
        "n_patches_corrected":  n_to_correct,
        "n_patches_with_fg":    n_patches_with_fg,
        "fg_patch_fraction":    round(n_patches_with_fg / max(n_to_correct, 1), 3),
        "n_voxels_corrected":   n_voxels_corrected,
        "n_slices_touched":     len(slices_touched),
        "pct_volume_corrected": round(100.0 * n_voxels_corrected / prediction.size, 2),
    }

    return mixed_label, provenance_map, stats
