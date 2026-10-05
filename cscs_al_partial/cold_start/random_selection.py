"""
Random Cold-Start
==================
Uniform random sampling from the unlabeled pool.
Baseline for all comparisons.
"""

import numpy as np
from typing import List

from .base import ColdStartSelector, ColdStartResult


class RandomColdStart(ColdStartSelector):

    def select(
        self,
        pool_ids: List[str],
        embeddings: np.ndarray,
        U_ssl: np.ndarray,
        T_ssl: np.ndarray,
        k: int,
        seed: int,
    ) -> ColdStartResult:
        rng = np.random.RandomState(seed)
        k_eff = min(k, len(pool_ids))
        indices = rng.choice(len(pool_ids), size=k_eff, replace=False)
        selected = [pool_ids[i] for i in sorted(indices)]

        return ColdStartResult(
            selected_ids=selected,
            gamma=None,
            dcr=None,
            metadata={'method': 'random', 'seed': seed},
        )
