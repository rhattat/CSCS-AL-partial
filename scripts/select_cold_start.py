#!/usr/bin/env python3
"""
CLI: run cold-start (round 0) selection for a dataset.

This script only covers the algorithmic core shipped in this repository
(see README.md's "What is not included" section) -- it selects K0 volume
IDs from a pool and writes them out; it does not train a task model.

Usage:
    python scripts/select_cold_start.py \\
        --dataset example_dataset --config_dir configs/ \\
        --strategy cscs_curriculum --seed 42 --out selection_round0.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cscs_al_partial.io import load_config, load_pool_data
from cscs_al_partial.strategies import get_cold_start_class


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset config name (configs/{dataset}.yaml)")
    parser.add_argument("--config_dir", default="configs", help="Directory containing default/paths/strategies/{dataset}.yaml")
    parser.add_argument("--strategy", default="cscs_curriculum", help="Strategy name from configs/strategies.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None, help="Output JSON path (default: stdout only)")
    args = parser.parse_args()

    config = load_config(args.dataset, config_dir=args.config_dir)
    pool = load_pool_data(config)

    cold_start_cls = get_cold_start_class(args.strategy)
    selector = cold_start_cls(config=config)

    k = config["budgets"]["cold_start_k"]
    result = selector.select(
        pool_ids=pool["volume_ids"],
        embeddings=pool["embeddings"],
        U_ssl=pool["U_ssl"],
        T_ssl=pool["T_ssl"],
        k=k,
        seed=args.seed,
    )

    out = {
        "selected_ids": result.selected_ids,
        "gamma": result.gamma,
        "dcr": result.dcr,
        "metadata": {k: v for k, v in result.metadata.items() if k != "scores"},
    }
    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
