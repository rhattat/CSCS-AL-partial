#!/usr/bin/env python3
"""
CLI: score and select the next batch of volumes for one iterative round
(r >= 1) of CSCS-AL.

Assumes a task model has already been trained on the currently labeled set
and used to produce softmax probability NPZs (key 'probabilities', shape
(C, D, H, W)) for every volume still in the residual pool -- this script
does not run inference itself. Training/inference are outside this
repository's scope (see README.md's "What is not included" section).

Usage:
    python scripts/run_iterative_round.py \\
        --dataset example_dataset --config_dir configs/ \\
        --labeled_ids_file labeled.txt --predictions_dir preds/round1 \\
        --dcr 0.42 --k 3 --out selection_round1.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cscs_al_partial.io import load_config, load_pool_data
from cscs_al_partial.acquisition import compute_useg, compute_iterative_scores, compute_gamma_fixed_dcr


def _read_id_list(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_dir", default="configs")
    parser.add_argument("--labeled_ids_file", required=True, help="Newline-delimited IDs already labeled")
    parser.add_argument("--predictions_dir", required=True, help="Dir with {volume_id}.npz softmax files for the residual pool")
    parser.add_argument("--dcr", type=float, required=True, help="DCR from the cold-start result (fixed for the whole run)")
    parser.add_argument("--k", type=int, default=None, help="Volumes to query this round (default: config budgets.round_k)")
    parser.add_argument("--dilation_voxels", type=int, default=3)
    parser.add_argument("--min_fg_voxels", type=int, default=100)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    config = load_config(args.dataset, config_dir=args.config_dir)
    pool = load_pool_data(config)

    labeled_ids = set(_read_id_list(args.labeled_ids_file))
    id_to_idx = {vid: i for i, vid in enumerate(pool["volume_ids"])}
    residual_ids = [vid for vid in pool["volume_ids"] if vid not in labeled_ids]
    if not residual_ids:
        raise SystemExit("No residual pool left -- every volume is already labeled.")

    T_residual = np.array([pool["T_ssl"][id_to_idx[v]] for v in residual_ids], dtype=np.float32)

    preds_dir = Path(args.predictions_dir)
    U_seg = []
    for vid in residual_ids:
        npz_path = preds_dir / f"{vid}.npz"
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing softmax NPZ for residual volume '{vid}': {npz_path}")
        U_seg.append(compute_useg(
            npz_path, dilation_voxels=args.dilation_voxels, min_fg_voxels=args.min_fg_voxels,
        )["useg"])
    U_seg = np.array(U_seg, dtype=np.float32)

    gamma_r = compute_gamma_fixed_dcr(
        n_labeled=len(labeled_ids), n_pool=len(pool["volume_ids"]), dcr=args.dcr,
        clip_min=config.get("cscs", {}).get("clip_min", 0.3),
        clip_max=config.get("cscs", {}).get("clip_max", 0.7),
    )
    scores = compute_iterative_scores(T_residual, U_seg, gamma=gamma_r)

    k = args.k if args.k is not None else config["budgets"]["round_k"]
    k_eff = min(k, len(residual_ids))
    top_k = np.argsort(scores)[::-1][:k_eff]
    query_ids = [residual_ids[i] for i in top_k]

    out = {
        "gamma_r": gamma_r,
        "queried_ids": query_ids,
        "scores_by_id": {residual_ids[i]: float(scores[i]) for i in range(len(residual_ids))},
    }
    print(json.dumps({"gamma_r": gamma_r, "queried_ids": query_ids}, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
