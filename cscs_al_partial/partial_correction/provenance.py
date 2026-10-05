"""
Provenance-Weighted Loss Maps
===============================
Each voxel in the training set receives a continuous weight in [0, 1]
according to how its label was obtained.

Categories (stored as int8 in provenance_map):
    HUMAN_FULL      = 0  -- original expert annotation (cold-start volumes)
    HUMAN_CORRECTED = 1  -- patch corrected manually by the annotator (E2)
    MODEL_UNCHECKED = 2  -- model prediction, not verified by a human
    EXCLUDED        = 3  -- deliberately excluded from the loss

Weight formula:
    w(v) = h(provenance(v)) * g(confidence(v))

    h(HUMAN_FULL or HUMAN_CORRECTED) = 1.0
    h(MODEL_UNCHECKED)               = alpha   (0.0 = ignore MU entirely)
    h(EXCLUDED)                      = 0.0

    g(confidence) = max(0, conf - tau_low) / (1 - tau_low)
                    (linear ramp; only applies when alpha > 0)

alpha=0.0: model predictions never contribute to the loss (E2 in the paper).
alpha=0.2: high-confidence model predictions contribute weakly, as a soft
           self-distillation signal (E2-OPT in the paper).
"""

import numpy as np

# Provenance category codes (int8)
HUMAN_FULL:      int = 0
HUMAN_CORRECTED: int = 1
MODEL_UNCHECKED: int = 2
EXCLUDED:        int = 3

HUMAN_CATEGORIES = (HUMAN_FULL, HUMAN_CORRECTED)


def build_weight_map(
    provenance_map: np.ndarray,
    confidence_map: np.ndarray,
    alpha: float = 0.0,
    tau_low: float = 0.70,
) -> np.ndarray:
    """
    Compute a continuous per-voxel weight map from provenance and confidence.

    Args:
        provenance_map  : integer array (same spatial shape as the label),
                          values in {HUMAN_FULL, HUMAN_CORRECTED,
                                     MODEL_UNCHECKED, EXCLUDED}
        confidence_map  : float array, same shape, values in [0, 1]
                          (typically max-softmax per voxel)
        alpha           : weight ceiling for MODEL_UNCHECKED voxels
        tau_low         : confidence below which MODEL_UNCHECKED gets w=0

    Returns:
        weight_map : float32 array, same shape, values in [0, 1]
    """
    weight_map = np.zeros(provenance_map.shape, dtype=np.float32)

    # Human-annotated voxels always get full weight
    human_mask = np.isin(provenance_map, list(HUMAN_CATEGORIES))
    weight_map[human_mask] = 1.0

    # Model-unchecked voxels: only contribute if alpha > 0
    if alpha > 0.0:
        model_mask = (provenance_map == MODEL_UNCHECKED)
        conf = np.clip(confidence_map, 0.0, 1.0)
        denom = max(1.0 - tau_low, 1e-8)
        g = np.maximum(0.0, (conf - tau_low) / denom)
        weight_map[model_mask] = alpha * g[model_mask]

    # EXCLUDED stays at 0 (already initialized)

    return np.clip(weight_map, 0.0, 1.0)


def make_provenance_map_from_masks(
    volume_shape: tuple,
    human_full_mask: np.ndarray,
    corrected_mask: np.ndarray,
    excluded_mask: np.ndarray,
) -> np.ndarray:
    """
    Build a provenance_map from boolean masks.

    Priority order (in case of overlap): EXCLUDED > HUMAN_FULL > CORRECTED.
    Everything not in any mask defaults to MODEL_UNCHECKED.

    Args:
        volume_shape    : (D, H, W)
        human_full_mask : bool array (D, H, W) -- original full annotation
        corrected_mask  : bool array (D, H, W) -- manually corrected patches
        excluded_mask   : bool array (D, H, W) -- excluded voxels

    Returns:
        provenance_map : int8 array (D, H, W)
    """
    pmap = np.full(volume_shape, MODEL_UNCHECKED, dtype=np.int8)
    pmap[corrected_mask]  = HUMAN_CORRECTED
    pmap[human_full_mask] = HUMAN_FULL
    pmap[excluded_mask]   = EXCLUDED
    return pmap


def summarize_provenance(
    provenance_map: np.ndarray,
    weight_map: np.ndarray,
) -> dict:
    """Return a dict of provenance statistics, useful for per-round logging."""
    total = provenance_map.size
    return {
        "n_voxels_human_full":      int((provenance_map == HUMAN_FULL).sum()),
        "n_voxels_human_corrected": int((provenance_map == HUMAN_CORRECTED).sum()),
        "n_voxels_model_unchecked": int((provenance_map == MODEL_UNCHECKED).sum()),
        "n_voxels_excluded":        int((provenance_map == EXCLUDED).sum()),
        "mean_weight":              round(float(weight_map.mean()), 4),
        "pct_weight_zero":          round(float((weight_map == 0).sum() / total * 100), 2),
        "pct_weight_one":           round(float((weight_map == 1).sum() / total * 100), 2),
    }
