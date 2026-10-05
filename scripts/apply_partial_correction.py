#!/usr/bin/env python3
"""
CLI: apply entropy-ranked partial correction to one queried volume, and
write out the mixed label map, provenance map, and correction/provenance
stats logs.

Inputs are plain .npy volumes so this script has no imaging-library
dependency; convert from NIfTI with your own tooling if needed
(e.g. `nibabel.load(path).get_fdata()`).

Usage:
    python scripts/apply_partial_correction.py \\
        --entropy_map entropy.npy --prediction pred.npy --reference ref.npy \\
        --confidence confidence.npy --budget 0.20 --patch_size 32 --alpha 0.0 \\
        --out_dir round1_volXYZ/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cscs_al_partial.partial_correction import (
    rank_patches_by_entropy, apply_partial_correction, build_weight_map, summarize_provenance,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entropy_map", required=True, help=".npy (D, H, W) voxelwise entropy of the current prediction")
    parser.add_argument("--prediction", required=True, help=".npy (D, H, W) int label map, current model prediction")
    parser.add_argument("--reference", required=True, help=".npy (D, H, W) int label map, reference/ground truth")
    parser.add_argument("--confidence", required=True, help=".npy (D, H, W) float max-softmax confidence")
    parser.add_argument("--budget", type=float, default=0.20, help="Fraction of patches to correct (patch-count based, see docs)")
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.0, help="MODEL_UNCHECKED weight ceiling (0.0 = E2, >0 = E2-OPT)")
    parser.add_argument("--tau_low", type=float, default=0.70)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    entropy_map = np.load(args.entropy_map)
    prediction = np.load(args.prediction)
    reference = np.load(args.reference)
    confidence_map = np.load(args.confidence)

    patches = rank_patches_by_entropy(entropy_map, patch_size=args.patch_size)
    mixed_label, provenance_map, stats = apply_partial_correction(
        prediction=prediction, reference=reference, patch_ranking=patches, budget=args.budget,
    )
    weight_map = build_weight_map(provenance_map, confidence_map, alpha=args.alpha, tau_low=args.tau_low)
    prov_summary = summarize_provenance(provenance_map, weight_map)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "mixed_label.npy", mixed_label)
    np.save(out_dir / "provenance_map.npy", provenance_map)
    np.save(out_dir / "weight_map.npy", weight_map)
    (out_dir / "correction_stats.json").write_text(json.dumps(stats, indent=2))
    (out_dir / "provenance_summary.json").write_text(json.dumps(prov_summary, indent=2))

    print(json.dumps({"correction_stats": stats, "provenance_summary": prov_summary}, indent=2))
    print(f"\nWrote mixed_label.npy, provenance_map.npy, weight_map.npy to {out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
