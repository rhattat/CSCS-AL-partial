"""
CSCS Cold-Start
================
Curriculum-Stratified Cold-Start Selection.

Pipeline:
  1. Compute DCR = Spearman(U_ssl, T) on the full pool
  2. k-means++ clustering with k = K0 on L2-normalized embeddings
  3. gamma = clip(0.5 + DCR/4 * alpha_eff/(1+alpha_eff), 0.3, 0.7)
  4. Per cluster: select argmax S(x) = T_local^(1-gamma) * U_local^gamma
"""

import math
import numpy as np
from typing import List
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .base import ColdStartSelector, ColdStartResult


class CSCSColdStart(ColdStartSelector):

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

        clip_min = self.config.get('cscs', {}).get('clip_min', 0.3)
        clip_max = self.config.get('cscs', {}).get('clip_max', 0.7)
        smin = self.config.get('cscs', {}).get('smin_cluster', 3)

        # --- DCR ---
        dcr_pre = self.config.get('dcr_precomputed', None)
        if dcr_pre is not None:
            dcr = float(dcr_pre)
        else:
            dcr, _ = spearmanr(U_ssl, T_ssl)
            dcr = float(dcr)

        # --- gamma (cold-start pacing) ---
        alpha_eff = k_eff / math.sqrt(N)
        gamma = 0.5 + (dcr / 4.0) * (alpha_eff / (1.0 + alpha_eff))
        gamma = max(clip_min, min(clip_max, gamma))

        # --- Clustering ---
        emb_scaled = StandardScaler().fit_transform(embeddings)
        km = KMeans(n_clusters=k_eff, init='k-means++', n_init=10, random_state=seed)
        labels = km.fit_predict(emb_scaled)

        # --- Per-cluster selection via composite score ---
        eps = 0.01
        selected_indices = []

        for cid in range(k_eff):
            members = np.where(labels == cid)[0]
            if len(members) == 0:
                continue
            if len(members) == 1:
                selected_indices.append(int(members[0]))
                continue

            # Small clusters are still scored individually rather than
            # merged -- merging is left as a config-time decision, not
            # hard-coded here.
            if len(members) < smin:
                pass

            # Percentile-rank normalization within the cluster
            U_local = U_ssl[members]
            T_local = T_ssl[members]

            n_m = len(members)
            U_pct = np.clip(
                np.argsort(np.argsort(U_local)).astype(float) / max(n_m - 1, 1),
                eps, 1.0
            )
            T_pct = np.clip(
                np.argsort(np.argsort(T_local)).astype(float) / max(n_m - 1, 1),
                eps, 1.0
            )

            S = (T_pct ** (1.0 - gamma)) * (U_pct ** gamma)
            best_local = np.argmax(S)
            selected_indices.append(int(members[best_local]))

        # Fallback: fill any remaining slots from a global ranking
        # (only triggered if a cluster ends up empty after fit_predict).
        if len(selected_indices) < k_eff:
            already = set(selected_indices)
            U_pct_g = np.clip(
                np.argsort(np.argsort(U_ssl)).astype(float) / max(N - 1, 1),
                eps, 1.0
            )
            T_pct_g = np.clip(
                np.argsort(np.argsort(T_ssl)).astype(float) / max(N - 1, 1),
                eps, 1.0
            )
            S_global = (T_pct_g ** (1.0 - gamma)) * (U_pct_g ** gamma)

            ranked = np.argsort(S_global)[::-1]
            for idx in ranked:
                if len(selected_indices) >= k_eff:
                    break
                if int(idx) not in already:
                    selected_indices.append(int(idx))
                    already.add(int(idx))

        selected_ids = [pool_ids[i] for i in selected_indices[:k_eff]]

        scores = {}
        for i in selected_indices[:k_eff]:
            scores[pool_ids[i]] = float(
                (T_ssl[i] ** (1.0 - gamma)) * (U_ssl[i] ** gamma)
            )

        return ColdStartResult(
            selected_ids=selected_ids,
            gamma=gamma,
            dcr=dcr,
            metadata={
                'method': 'cscs',
                'alpha_eff': alpha_eff,
                'N': N,
                'K': k_eff,
                'seed': seed,
                'cluster_sizes': np.bincount(labels).tolist(),
                'scores': scores,
            },
        )
