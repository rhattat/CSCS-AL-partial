"""
Segmentation evaluation metrics: Dice and 95th-percentile Hausdorff distance.

IMPORTANT -- physical spacing convention:
NIfTI images loaded via `nibabel.load(path).get_fdata()` return an array
whose axis order matches `img.header.get_zooms()` directly: axis 0 <-> zooms[0]
(x), axis 1 <-> zooms[1] (y), axis 2 <-> zooms[2] (z). Do NOT reverse the
zooms tuple before passing it to `scipy.ndimage.distance_transform_edt`'s
`sampling` argument -- the array is not in (z, y, x) order unless you have
explicitly transposed it yourself. The per-round evaluation code that
actually produced the paper's results reversed the spacing under the
mistaken assumption that the array was already (z, y, x); this was found
and fixed shortly before this repository was prepared (no-op on isotropic
data such as FeTA, real distortion on the strongly anisotropic datasets,
~0.8mm in-plane vs ~6mm through-plane). `dice_coefficient` and `hd95` below
are otherwise a direct, unmodified port of that same evaluation code
(including its empty-mask conventions: both masks empty -> Dice 1.0 / HD95
0.0; this is a real behavioral choice of the code that ran the experiments,
not an idealized convention written for this release).
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_coefficient(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """
    Binary Dice similarity coefficient.

    Returns 1.0 if both `pred_bin` and `gt_bin` are empty for this class
    (correctly-predicted absence). This is a deliberate convention, not a
    numerical accident -- state it explicitly if you report per-class Dice
    for classes that can be legitimately absent from a given volume.
    """
    inter = np.logical_and(pred_bin, gt_bin).sum()
    s = pred_bin.sum() + gt_bin.sum()
    return 1.0 if s == 0 else float(2.0 * inter / s)


def hd95(pred_bin: np.ndarray, gt_bin: np.ndarray, spacing: Sequence[float]) -> float:
    """
    95th-percentile symmetric surface distance, in the physical units of
    `spacing` (typically millimetres).

    `spacing` must be given in the same axis order as `pred_bin`/`gt_bin`
    themselves (see module docstring) -- e.g. if your arrays come straight
    from `nib.load(path).get_fdata()`, pass `img.header.get_zooms()[:3]`
    unmodified.

    Returns 0.0 if either mask is empty -- this is the convention of the
    code that produced the paper's reported numbers, not an idealized
    choice made for this release. It means a false-positive prediction
    against an empty reference (or a total miss against a non-empty
    reference) scores a "perfect" 0.0 rather than an undefined or heavily
    penalized value; state this explicitly wherever HD95 is reported
    downstream.
    """
    if not pred_bin.any() or not gt_bin.any():
        return 0.0

    pred_surf = pred_bin ^ binary_erosion(pred_bin)
    gt_surf   = gt_bin   ^ binary_erosion(gt_bin)

    dt_pred = distance_transform_edt(~pred_bin, sampling=spacing)
    dt_gt   = distance_transform_edt(~gt_bin,   sampling=spacing)

    d_p2g = dt_gt[pred_surf]
    d_g2p = dt_pred[gt_surf]
    all_d = np.concatenate([d_p2g, d_g2p])
    return float(np.percentile(all_d, 95))


def evaluate_volume(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: Sequence[float],
    labels: List[int],
    fg_labels: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Per-class Dice/HD95 for one volume, plus an unweighted macro-average
    over `fg_labels` (or all non-zero `labels` if not given).

    `spacing` follows the same convention as `hd95` above: same axis order
    as `pred`/`gt`, unreversed.

    Returns a flat dict: {"dice_<label>": ..., "hd95_<label>": ...,
    "dice_mean_fg": ..., "hd95_mean_fg": ...}.
    """
    fg_labels = fg_labels if fg_labels is not None else [l for l in labels if l != 0]
    out: Dict[str, float] = {}
    for lab in labels:
        pred_bin = (pred == lab)
        gt_bin = (gt == lab)
        out[f"dice_{lab}"] = dice_coefficient(pred_bin, gt_bin)
        out[f"hd95_{lab}"] = hd95(pred_bin, gt_bin, spacing)

    out["dice_mean_fg"] = float(np.mean([out[f"dice_{l}"] for l in fg_labels]))
    out["hd95_mean_fg"] = float(np.mean([out[f"hd95_{l}"] for l in fg_labels]))
    return out
