"""
Structured run logging.

Writes per-run CSV/YAML logs designed to match the outputs of the other
modules in this package directly -- e.g. `append_provenance_log` takes the
dict produced by `partial_correction.summarize_provenance`, and
`append_correction_log` takes the stats dict produced by
`partial_correction.apply_partial_correction`.

File layout (all under a single `run_dir`, one per dataset/strategy/seed):

    run_manifest.yaml     -- static metadata for this run (written once, at startup)
    round_log.csv         -- one row per round: selection scores, timing
    metrics.csv           -- one row per round: Dice/HD95 per class
    correction_log.csv    -- one row per round: patch correction cost (partial-correction runs only)
    provenance_log.csv    -- one row per round: weight map diagnostics (partial-correction runs only)
"""

import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml


# ══════════════════════════════════════════════════════════════════════
# MANIFEST
# ══════════════════════════════════════════════════════════════════════

def write_run_manifest(
    run_dir: str,
    dataset: str,
    seed_index: int,
    strategy: str,
    cscs_al: bool = False,
    partial_correction: bool = False,
    correction_budget: Optional[float] = None,
    patch_size: Optional[int] = None,
    provenance_alpha: Optional[float] = None,
    tau_low: Optional[float] = None,
    gamma_schedule: Optional[str] = None,
    extra: Optional[Dict] = None,
) -> None:
    """
    Write run_manifest.yaml at the start of a run. Call once per run.

    Args:
        cscs_al             : whether CSCS-AL iterative (round >= 1) scoring is active
        partial_correction  : whether sliding-window partial correction is active
        extra               : any additional fields to record (e.g. trainer name,
                               if you are pairing this with your own nnU-Net trainer)
    """
    manifest = {
        "dataset":            dataset,
        "seed_index":         seed_index,
        "strategy":           strategy,
        "cscs_al":            cscs_al,
        "partial_correction": partial_correction,
        "date":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit":         _get_git_hash(),
        "gamma_schedule":     gamma_schedule or "cscs_curriculum_default",
        "correction_budget":  correction_budget,
        "patch_size":         patch_size,
        "provenance_alpha":   provenance_alpha,
        "tau_low":            tau_low,
    }
    if extra:
        manifest.update(extra)
    out = Path(run_dir) / "run_manifest.yaml"
    with open(out, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)


def _get_git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════
# ROUND LOG
# ══════════════════════════════════════════════════════════════════════

_ROUND_LOG_FIELDS = [
    "run_id", "dataset", "seed", "round",
    "n_labeled", "n_unlabeled",
    "selected_id",
    "gamma_r",
    "T_ssl_ranked_selected",   # T_ranked value of the selected volume
    "U_seg_ranked_selected",   # U_seg_ranked value of the selected volume
    "S_r_selected",            # final CSCS-AL score of the selected volume
    "T_ssl_ranked_mean",       # mean over the residual pool (sanity check)
    "U_seg_ranked_mean",       # mean over the residual pool (sanity check)
    "selection_time_sec",
    "inference_time_sec",
    "training_time_sec",
    "total_round_time_sec",
]


def append_round_log(run_dir: str, row: Dict) -> None:
    _append_csv(Path(run_dir) / "round_log.csv", _ROUND_LOG_FIELDS, row)


# ══════════════════════════════════════════════════════════════════════
# METRICS LOG
# ══════════════════════════════════════════════════════════════════════

_METRICS_FIELDS_BASE = [
    "run_id", "dataset", "seed", "round", "n_labeled",
    "dice_mean_fg", "hd95_mean_fg",
]


def append_metrics_log(
    run_dir: str,
    run_id: str,
    dataset: str,
    seed: int,
    round_t: int,
    n_labeled: int,
    metrics: Dict,
    class_labels: Optional[Dict[int, str]] = None,
) -> None:
    """
    Write one metrics row (as produced by `evaluation.evaluate_volume`).
    Per-class dice/hd95 columns are added dynamically.

    Args:
        class_labels: optional {label_index: label_name} to name columns
                       (e.g. {1: 'organ_a', 2: 'organ_b'}); falls back to
                       whatever keys are already in `metrics` otherwise.
    """
    row = {
        "run_id":       run_id,
        "dataset":      dataset,
        "seed":         seed,
        "round":        round_t,
        "n_labeled":    n_labeled,
        "dice_mean_fg": round(metrics.get("dice_mean_fg", 0), 4),
        "hd95_mean_fg": round(metrics.get("hd95_mean_fg", 0), 3),
    }
    if class_labels:
        for cls_idx, cls_name in class_labels.items():
            row[f"dice_{cls_name}"] = round(metrics.get(f"dice_{cls_idx}", 0), 4)
            row[f"hd95_{cls_name}"] = round(metrics.get(f"hd95_{cls_idx}", 0), 3)
    else:
        for k, v in metrics.items():
            if k not in row:
                row[k] = v

    _append_csv(Path(run_dir) / "metrics.csv", list(row.keys()), row)


# ══════════════════════════════════════════════════════════════════════
# CORRECTION LOG (partial-correction runs only)
# ══════════════════════════════════════════════════════════════════════

_CORRECTION_LOG_FIELDS = [
    "run_id", "dataset", "seed", "round", "selected_id",
    "n_patches_total",
    "n_patches_corrected",
    "n_patches_with_fg",
    "fg_patch_fraction",
    "n_voxels_corrected",
    "n_slices_touched",
    "pct_volume_corrected",
    "cum_patches_corrected",
    "cum_voxels_corrected",
    "cum_pct_volume_corrected",
]


def append_correction_log(run_dir: str, row: Dict) -> None:
    """Row is the stats dict from `partial_correction.apply_partial_correction`,
    plus run/dataset/seed/round/selected_id identifiers and any cumulative
    fields you choose to track across rounds."""
    _append_csv(Path(run_dir) / "correction_log.csv", _CORRECTION_LOG_FIELDS, row)


# ══════════════════════════════════════════════════════════════════════
# PROVENANCE LOG (partial-correction runs only)
# ══════════════════════════════════════════════════════════════════════

_PROVENANCE_LOG_FIELDS = [
    "run_id", "dataset", "seed", "round", "selected_id",
    "n_voxels_human_full",
    "n_voxels_human_corrected",
    "n_voxels_model_unchecked",
    "n_voxels_excluded",
    "mean_weight",
    "pct_weight_zero",
    "pct_weight_one",
]


def append_provenance_log(run_dir: str, row: Dict) -> None:
    """Row is the dict from `partial_correction.summarize_provenance`, plus
    run/dataset/seed/round/selected_id identifiers."""
    _append_csv(Path(run_dir) / "provenance_log.csv", _PROVENANCE_LOG_FIELDS, row)


# ══════════════════════════════════════════════════════════════════════
# INTERNAL HELPER
# ══════════════════════════════════════════════════════════════════════

def _append_csv(path: Path, fields: List[str], row: Dict) -> None:
    write_header = not path.exists()
    all_fields = fields + [k for k in row if k not in fields]
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
