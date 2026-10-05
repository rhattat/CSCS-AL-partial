"""
ProbCover Cold-Start (baseline)
=================================
Greedy ball-covering that maximizes probability mass covered
(Yehuda et al., 2022), used here as a representativeness-only baseline
against which CSCS's typicality-uncertainty composite is compared.
"""

import numpy as np
from typing import List
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

from .base import ColdStartSelector, ColdStartResult


class ProbCoverColdStart(ColdStartSelector):

    def select(
        self,
        pool_ids: List[str],
        embeddings: np.ndarray,
        U_ssl: np.ndarray,
        T_ssl: np.ndarray,
        k: int,
        seed: int,
    ) -> ColdStartResult:
        N = len(pool_ids)
        k_eff = min(k, N)

        emb_scaled = StandardScaler().fit_transform(embeddings)
        dists = euclidean_distances(emb_scaled)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        # Auto delta via binary search on the covering radius
        lo = np.percentile(nn_dists, 10)
        hi = np.percentile(nn_dists, 90)
        for _ in range(20):
            mid = (lo + hi) / 2
            avg_cov = (dists <= mid).sum(axis=1).mean()
            if avg_cov * k_eff > N * 1.5:
                hi = mid
            else:
                lo = mid
        delta = (lo + hi) / 2

        # Greedy covering
        covered = np.zeros(N, dtype=bool)
        selected = []
        for _ in range(k_eff):
            coverage = np.array([
                np.sum((dists[i] <= delta) & ~covered) if i not in selected else -1
                for i in range(N)
            ])
            best = int(np.argmax(coverage))
            selected.append(best)
            covered[dists[best] <= delta] = True
            if covered.all():
                break

        # Fill remainder with farthest-point sampling if coverage saturates early
        if len(selected) < k_eff:
            min_d = np.full(N, np.inf)
            for s in selected:
                d = np.linalg.norm(emb_scaled - emb_scaled[s], axis=1)
                min_d = np.minimum(min_d, d)
            for s in selected:
                min_d[s] = -1
            while len(selected) < k_eff:
                best = int(np.argmax(min_d))
                selected.append(best)
                min_d[best] = -1
                d = np.linalg.norm(emb_scaled - emb_scaled[best], axis=1)
                min_d = np.minimum(min_d, d)

        selected_ids = [pool_ids[i] for i in selected[:k_eff]]

        return ColdStartResult(
            selected_ids=selected_ids,
            gamma=None,
            dcr=None,
            metadata={'method': 'probcover', 'delta': float(delta), 'seed': seed},
        )
