"""
Minimal, self-contained walkthrough of CSCS-AL on synthetic data.

No GPU, no real dataset required -- everything here is randomly generated
just to exercise the API end to end: cold-start selection, one iterative
round, and partial correction of a single "queried" volume.

Run:
    python examples/minimal_example.py
"""
import numpy as np

from cscs_al_partial.cold_start import CSCSColdStart
from cscs_al_partial.acquisition import (
    compute_iterative_scores, compute_gamma_fixed_dcr, ConvergenceChecker,
)
from cscs_al_partial.partial_correction import (
    rank_patches_by_entropy, apply_partial_correction, build_weight_map, summarize_provenance,
)
from cscs_al_partial.evaluation import dice_coefficient, hd95


def main():
    rng = np.random.RandomState(0)

    # ---- 1. Cold-start selection on a synthetic pool ----
    N, d = 40, 32
    pool_ids = [f"vol_{i:03d}" for i in range(N)]
    embeddings = rng.normal(size=(N, d)).astype(np.float32)
    U_ssl = rng.uniform(0, 1, size=N)
    T_ssl = rng.uniform(0, 1, size=N)

    cold_start = CSCSColdStart(config={"cscs": {"clip_min": 0.3, "clip_max": 0.7}})
    result = cold_start.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=42)
    print(f"[cold-start] selected {result.selected_ids}")
    print(f"[cold-start] DCR={result.dcr:.3f}  gamma0={result.gamma:.3f}")

    labeled_ids = set(result.selected_ids)
    residual_ids = [v for v in pool_ids if v not in labeled_ids]

    # ---- 2. One iterative round (r=1) ----
    # In a real run, U_seg comes from `compute_useg(npz_path)["useg"]` on the
    # current task model's softmax predictions for each residual volume;
    # here we just fake plausible values.
    U_seg = rng.uniform(0, 1, size=len(residual_ids))
    T_residual = rng.uniform(0, 1, size=len(residual_ids))

    gamma_1 = compute_gamma_fixed_dcr(n_labeled=len(labeled_ids), n_pool=N, dcr=result.dcr)
    scores = compute_iterative_scores(T_residual, U_seg, gamma=gamma_1)
    K_r = 3
    top_k = np.argsort(scores)[::-1][:K_r]
    query_ids = [residual_ids[i] for i in top_k]
    print(f"\n[round 1] gamma_1={gamma_1:.3f}")
    print(f"[round 1] queried {query_ids}")

    # ---- 3. Partial correction of the first queried volume ----
    D, H, W = 24, 64, 64
    entropy_map = rng.uniform(0, 2, size=(D, H, W)).astype(np.float32)
    prediction = rng.randint(0, 3, size=(D, H, W)).astype(np.int16)
    reference = prediction.copy()
    reference[10:14, 20:40, 20:40] = 2  # pretend the model missed a region

    patches = rank_patches_by_entropy(entropy_map, patch_size=8)
    mixed_label, provenance_map, corr_stats = apply_partial_correction(
        prediction=prediction, reference=reference, patch_ranking=patches, budget=0.20,
    )
    confidence_map = rng.uniform(0.5, 1.0, size=(D, H, W)).astype(np.float32)
    weight_map = build_weight_map(provenance_map, confidence_map, alpha=0.0)  # E2 (alpha=0)
    prov_summary = summarize_provenance(provenance_map, weight_map)

    print(f"\n[partial correction] {corr_stats['n_patches_corrected']}/"
          f"{corr_stats['n_patches_total']} patches corrected "
          f"({corr_stats['pct_volume_corrected']:.1f}% of the volume)")
    print(f"[partial correction] provenance breakdown: {prov_summary}")

    # ---- 4. Evaluate the mixed label against the reference ----
    for lab in (1, 2):
        d = dice_coefficient(mixed_label == lab, reference == lab)
        h = hd95(mixed_label == lab, reference == lab, spacing=(2.0, 1.0, 1.0))
        print(f"[eval] class {lab}: Dice={d:.3f}  HD95={h:.3f}")

    # ---- 5. Convergence check across (fake) rounds ----
    checker = ConvergenceChecker(delta_pp=0.5, patience=2, max_rounds=9)
    for round_dice in [72.0, 78.5, 82.1, 82.4, 82.6]:
        stop = checker.update(round_dice, pool_remaining=len(residual_ids) - K_r, round_budget=K_r)
        if stop:
            print(f"\n[convergence] stopped: {checker.reason}")
            break


if __name__ == "__main__":
    main()
