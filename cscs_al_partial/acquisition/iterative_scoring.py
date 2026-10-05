"""
CSCS-AL — iterative acquisition scoring for active learning rounds r >= 1.

Round 0 uses the cold-start CSCS score (see cold_start/cscs.py).
Rounds r >= 1 replace SSL-derived uncertainty with segmentation entropy
computed from the current task model's softmax predictions on the residual
pool.

Score formula:
    S_r(x) = T_ranked(x)^(1-gamma_r) * U_seg_ranked(x)^(gamma_r)

Both signals are percentile-rank normalized to [eps, 1] within the current
residual pool before being combined -- not min-max normalized.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════
# RANK NORMALIZATION
# ══════════════════════════════════════════════════════════════════════

def rank_normalize(scores: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Percentile-rank normalize an array to [eps, 1].

    Uses average ranks for ties. Normalization is rank(x) / N_pool.
    The eps floor prevents exact-zero inputs to the geometric mean below.

    Args:
        scores : 1-D float array of length N_pool
        eps    : minimum value after normalization (default 1e-6)

    Returns:
        normalized : same shape as scores, values in [eps, 1]
    """
    if len(scores) == 0:
        return np.array([], dtype=np.float32)
    ranked = rankdata(scores, method="average")  # shape (N,), values in [1, N]
    normalized = ranked / len(scores)             # values in [1/N, 1]
    return np.clip(normalized, eps, 1.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════
# VOLUME-LEVEL SEGMENTATION UNCERTAINTY (U_seg)
# ══════════════════════════════════════════════════════════════════════

def compute_useg(
    npz_path: Path,
    fg_labels: Optional[List[int]] = None,
    dilation_voxels: int = 3,
    min_fg_voxels: int = 100,
) -> Dict:
    """
    Compute volume-level segmentation uncertainty from a softmax NPZ.

    Uses mean voxelwise entropy over a dilated foreground mask. The
    dilation structuring element is a full cube of side
    (2*dilation_voxels + 1) -- i.e. Chebyshev distance, not a Euclidean
    ball. Falls back to whole-volume entropy if the dilated foreground is
    too small to be a reliable ROI.

    Args:
        npz_path        : path to an nnU-Net --save_probabilities NPZ,
                           key 'probabilities', shape (C, D, H, W)
        fg_labels        : foreground label indices (default: all labels > 0)
        dilation_voxels  : morphological dilation radius around the
                            predicted foreground mask
        min_fg_voxels    : if the dilated foreground has fewer voxels than
                            this, fall back to the whole volume

    Returns a dict with:
        useg        : float, mean entropy over the ROI (the score used in S_r)
        fg_fraction : fraction of voxels predicted as foreground
        roi_used    : 'fg_dilated' | 'full_volume'
        entropy_map : np.ndarray (D, H, W), full voxelwise entropy
    """
    data = np.load(str(npz_path))
    key = 'probabilities' if 'probabilities' in data else list(data.keys())[0]
    softmax = data[key].astype(np.float32)  # (C, D, H, W)

    eps = 1e-7
    entropy_map = -np.sum(softmax * np.log(softmax + eps), axis=0)  # (D, H, W)
    pred = np.argmax(softmax, axis=0)                                 # (D, H, W)

    if fg_labels is not None:
        fg_mask = np.isin(pred, fg_labels)
    else:
        fg_mask = pred > 0

    if dilation_voxels > 0 and fg_mask.any():
        struct = np.ones((dilation_voxels * 2 + 1,) * 3, dtype=bool)
        fg_mask_dilated = binary_dilation(fg_mask, structure=struct)
    else:
        fg_mask_dilated = fg_mask

    if fg_mask_dilated.sum() >= min_fg_voxels:
        roi = fg_mask_dilated
        roi_used = "fg_dilated"
    else:
        roi = np.ones_like(fg_mask, dtype=bool)
        roi_used = "full_volume"
        logger.debug(f"compute_useg: FG too small ({fg_mask_dilated.sum()} voxels), "
                     f"fallback to full volume")

    useg = float(entropy_map[roi].mean())
    fg_fraction = float(fg_mask.sum() / fg_mask.size)

    return {
        "useg":        useg,
        "fg_fraction": fg_fraction,
        "roi_used":    roi_used,
        "entropy_map": entropy_map,
    }


# ══════════════════════════════════════════════════════════════════════
# ROUND-TO-ROUND PREDICTION AGREEMENT (label-free stopping proxy)
# ══════════════════════════════════════════════════════════════════════

def compute_prediction_agreement(
    prev_predictions_dir: Path,
    curr_predictions_dir: Path,
    common_ids: List[str],
    fg_labels: Optional[List[int]] = None,
) -> Optional[float]:
    """
    Mean foreground Dice-like overlap between two rounds' hard
    segmentations, for volumes with a softmax NPZ in both directories.

    Used as a label-free proxy for the metric-based stopping criterion
    (see ConvergenceChecker.update_proxy(mode="stability")): once the
    model's own predictions on the residual pool stop changing between
    rounds, further annotation is unlikely to move the needle, even with
    no ground truth to measure that against directly.

    Args:
        prev_predictions_dir : dir with {volume_id}.npz softmax from round r-1
        curr_predictions_dir : dir with {volume_id}.npz softmax from round r
        common_ids            : volume IDs to compare (only those found in
                                 both directories contribute to the mean)
        fg_labels              : foreground label indices (default: all labels > 0)

    Returns:
        mean agreement in [0, 100], or None if no volume in common_ids has
        a prediction in both directories (nothing to compare yet -- e.g.
        the very first round predictions were produced).
    """
    prev_dir = Path(prev_predictions_dir)
    curr_dir = Path(curr_predictions_dir)
    scores = []
    for vid in common_ids:
        prev_path = prev_dir / f"{vid}.npz"
        curr_path = curr_dir / f"{vid}.npz"
        if not prev_path.exists() or not curr_path.exists():
            continue
        prev_data = np.load(str(prev_path))
        curr_data = np.load(str(curr_path))
        prev_softmax = prev_data['probabilities'] if 'probabilities' in prev_data else prev_data[list(prev_data.keys())[0]]
        curr_softmax = curr_data['probabilities'] if 'probabilities' in curr_data else curr_data[list(curr_data.keys())[0]]
        prev_pred = np.argmax(prev_softmax, axis=0)
        curr_pred = np.argmax(curr_softmax, axis=0)

        if fg_labels is not None:
            prev_fg = np.isin(prev_pred, fg_labels)
            curr_fg = np.isin(curr_pred, fg_labels)
        else:
            prev_fg = prev_pred > 0
            curr_fg = curr_pred > 0

        union = int(prev_fg.sum()) + int(curr_fg.sum())
        if union == 0:
            scores.append(100.0)  # both predict empty foreground -> perfect agreement
            continue
        inter = int(np.logical_and(prev_fg, curr_fg).sum())
        scores.append(200.0 * inter / union)

    return float(np.mean(scores)) if scores else None


# ══════════════════════════════════════════════════════════════════════
# ITERATIVE ACQUISITION SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_iterative_scores(
    T_ssl: np.ndarray,
    U_seg: np.ndarray,
    gamma: float,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Compute CSCS-AL iterative acquisition scores for the residual pool.

        S_r(x) = T_ranked(x)^(1-gamma) * U_seg_ranked(x)^gamma

    Both T_ssl and U_seg are percentile-rank normalized before scoring.

    Args:
        T_ssl : (N,) SSL typicality scores for the residual pool
        U_seg : (N,) segmentation uncertainty scores for the residual pool
        gamma : pacing weight in [0, 1]; 0 -> typicality only, 1 -> U_seg only
        eps   : floor for rank normalization

    Returns:
        scores : (N,) float32, higher = more informative for selection
    """
    T_ranked = rank_normalize(T_ssl, eps=eps)
    U_ranked = rank_normalize(U_seg, eps=eps)
    return (T_ranked ** (1.0 - gamma) * U_ranked ** gamma).astype(np.float32)
