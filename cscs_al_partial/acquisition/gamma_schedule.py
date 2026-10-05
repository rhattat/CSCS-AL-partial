"""
Adaptive pacing (gamma_r) for iterative acquisition.

Two variants are provided:

- compute_gamma_fixed_dcr: the reference schedule used throughout the
  paper's main results (E1/E2/E2-OPT). DCR is computed once at cold-start
  (Spearman between SSL reconstruction difficulty and embedding typicality
  on the full pool) and held fixed for the rest of the run.

      gamma_r = clip(0.5 + DCR/4 * c_r,  0.3, 0.7)
      c_r     = |L_r| / (sqrt(N) + |L_r|)

- compute_gamma_dynamic: an ablation variant that recomputes DCR each round
  from the residual pool (segmentation uncertainty vs. typicality, rather
  than SSL difficulty vs. typicality) and uses a wider amplitude and clip.

      DCR_r   = Spearman(U_seg_r, T_r)          on the residual pool P_r
      gamma_r = clip(0.5 + DCR_r/divisor * c_r, clip_min, clip_max)

  In the paper's ablations this dynamic variant was not more stable than
  the fixed-DCR schedule -- late-round DCR_r estimates get noisy as the
  residual pool shrinks. It is included here for completeness / further
  experimentation, not as the recommended default.
"""

import math
import numpy as np
from scipy.stats import spearmanr


def compute_gamma_fixed_dcr(
    n_labeled: int,
    n_pool: int,
    dcr: float,
    clip_min: float = 0.3,
    clip_max: float = 0.7,
) -> float:
    """
    gamma_r = clip(0.5 + DCR/4 * alpha_eff/(1+alpha_eff), clip_min, clip_max)
    where alpha_eff = |L_r| / sqrt(N).

    DCR is fixed for the whole run (computed once at cold-start).
    """
    if n_pool <= 0:
        return 0.5
    alpha_eff = n_labeled / math.sqrt(n_pool)
    budget_factor = alpha_eff / (1.0 + alpha_eff)
    gamma = 0.5 + (dcr / 4.0) * budget_factor
    return max(clip_min, min(clip_max, gamma))


def compute_dcr(u_scores: np.ndarray, t_scores: np.ndarray) -> float:
    """
    Spearman rank correlation between uncertainty and typicality.

    Returns 0.0 when the pool has fewer than 4 volumes -- the estimate is
    too noisy below this threshold to safely drive the gamma schedule.
    Returns 0.0 (not NaN) when all scores are identical (degenerate pool).

    Args:
        u_scores: (N,) uncertainty scores for residual pool volumes
        t_scores: (N,) typicality scores for the same volumes

    Returns:
        DCR in [-1, 1]
    """
    n = len(u_scores)
    if n < 4:
        return 0.0
    rho, _ = spearmanr(u_scores, t_scores)
    if np.isnan(rho):
        return 0.0
    return float(rho)


def compute_gamma_dynamic(
    dcr_r: float,
    n_labeled: int,
    n_total: int,
    divisor: float = 2.0,
    clip_min: float = 0.1,
    clip_max: float = 0.9,
) -> float:
    """
    Per-round dynamic-DCR gamma schedule (ablation variant).

        alpha_r = n_labeled / sqrt(n_total)
        c_r     = alpha_r / (1 + alpha_r)          in (0, 1)
        gamma_r = clip(0.5 + DCR_r/divisor * c_r,  clip_min, clip_max)

    Args:
        dcr_r:     round-level DCR in [-1, 1] (see compute_dcr)
        n_labeled: current |L_r| (volumes labeled so far)
        n_total:   full pool size N (constant throughout the run)
        divisor:   controls maximum shift magnitude (default 2.0)
        clip_min:  lower bound for gamma (default 0.1)
        clip_max:  upper bound for gamma (default 0.9)

    Returns:
        gamma_r in [clip_min, clip_max]
    """
    if n_total <= 0:
        return 0.5
    alpha_r = n_labeled / math.sqrt(n_total)
    c_r = alpha_r / (1.0 + alpha_r)
    gamma_r = 0.5 + (dcr_r / divisor) * c_r
    return float(np.clip(gamma_r, clip_min, clip_max))
