"""
Embeddings I/O
===============
Loads pre-computed SSL embeddings from individual .npy files, and
typicality T(x) / uncertainty U(x) proxies either from a companion CSV
or computed directly from the embeddings as a fallback.

Expected directory structure:
    {embeddings_dir}/
        volume_001.npy    (d-dim vector, one file per volume)
        volume_002.npy
        ...
    {ssl_features_dir}/
        {dataset}_ssl_features_train.csv   (columns: volume_id, uncertainty, typicality)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def normalize_id(vol_id: str, suffixes: List[str]) -> str:
    """Strip known filename suffixes (e.g. '_0000') from a volume ID."""
    vol_id = str(vol_id).strip()
    for suf in suffixes:
        vol_id = vol_id.replace(suf, '')
    return vol_id


def load_embeddings_npy(
    embeddings_dir: str,
    volume_ids: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings from individual .npy files.

    Args:
        embeddings_dir: directory containing {volume_id}.npy files
        volume_ids: if provided, only load these IDs (and preserve order)

    Returns:
        embeddings: (N, d) array
        loaded_ids: list of volume IDs that were successfully loaded
    """
    emb_dir = Path(embeddings_dir)
    if not emb_dir.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {emb_dir}")

    available = {fpath.stem: fpath for fpath in sorted(emb_dir.glob("*.npy"))}
    logger.info(f"Found {len(available)} embedding files in {emb_dir}")

    if volume_ids is not None:
        load_order = [vid for vid in volume_ids if vid in available]
        missing = [vid for vid in volume_ids if vid not in available]
        if missing:
            logger.warning(f"{len(missing)} volumes missing embeddings: {missing[:5]}...")
    else:
        load_order = sorted(available.keys())

    if not load_order:
        raise ValueError(f"No embeddings found for requested volumes in {emb_dir}")

    emb_list, loaded_ids = [], []
    for vid in load_order:
        emb = np.load(str(available[vid]))
        if emb.ndim > 1:
            emb = emb.flatten()
        emb_list.append(emb)
        loaded_ids.append(vid)

    embeddings = np.stack(emb_list)
    logger.info(f"Loaded embeddings: shape={embeddings.shape}")
    return embeddings, loaded_ids


def load_ssl_proxy_csv(
    ssl_features_dir: str,
    dataset_name: str,
    volume_ids: Optional[List[str]] = None,
) -> Optional[pd.DataFrame]:
    """
    Load a precomputed SSL proxy CSV with U and T scores, if one exists.

    Looks for `{ssl_features_dir}/{dataset}_ssl_features_train.csv` (and a
    couple of alternate filenames), with columns identifying the volume ID,
    uncertainty and typicality (case-insensitive, several common aliases).

    Returns:
        DataFrame with columns [volume_id, U, T], or None if no CSV is found
        or it doesn't contain recognizable U/T columns.
    """
    features_dir = Path(ssl_features_dir)

    patterns = [
        f"{dataset_name.lower()}_ssl_features_train.csv",
        f"{dataset_name.lower()}_ssl_features.csv",
        "ssl_features_train.csv",
        "ssl_features.csv",
    ]
    csv_path = next((features_dir / p for p in patterns if (features_dir / p).exists()), None)
    if csv_path is None:
        csvs = list(features_dir.glob("*ssl*features*.csv"))
        csv_path = csvs[0] if csvs else None
    if csv_path is None:
        logger.info(f"No SSL proxy CSV found in {features_dir}")
        return None

    logger.info(f"Loading SSL proxy from {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    id_col = next((c for c in ['volume_id', 'volume', 'id'] if c in df.columns), df.columns[0])
    u_col = next((c for c in ['uncertainty', 'uncertainty_normalized', 'u'] if c in df.columns), None)
    t_col = next((c for c in ['typicality', 'typicality_normalized', 't'] if c in df.columns), None)

    if u_col is None or t_col is None:
        logger.warning(f"Cannot find U/T columns in {csv_path}. Columns: {list(df.columns)}")
        return None

    result = pd.DataFrame({
        'volume_id': df[id_col].astype(str),
        'U': df[u_col].astype(float),
        'T': df[t_col].astype(float),
    })
    if volume_ids is not None:
        result = result[result['volume_id'].isin(set(volume_ids))].reset_index(drop=True)
    return result


def load_train_val_ids(
    train_file: str,
    val_file: str,
    suffixes: List[str],
) -> Tuple[List[str], List[str]]:
    """Load newline-delimited train/val volume ID lists ('#' = comment)."""

    def _read_ids(fpath):
        ids = []
        with open(fpath, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    ids.append(normalize_id(line, suffixes))
        seen, out = set(), []
        for x in ids:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    train_ids = _read_ids(train_file)
    val_ids = _read_ids(val_file)
    logger.info(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}")
    return train_ids, val_ids


def load_pool_data(config: dict) -> Dict:
    """
    Load everything needed to run cold-start / iterative acquisition:
    embeddings, U_ssl, T_ssl (min-max normalized), and train/val ID splits.

    Args:
        config: merged config dict from `cscs_al_partial.io.config.load_config`

    Returns:
        dict with keys: train_ids, val_ids, embeddings, volume_ids, U_ssl, T_ssl
    """
    suffixes = config['dataset']['id_suffixes']
    train_ids, val_ids = load_train_val_ids(config['train_file'], config['val_file'], suffixes)
    embeddings, emb_ids = load_embeddings_npy(config['embeddings_dir'], volume_ids=train_ids)

    proxy_df = load_ssl_proxy_csv(config['ssl_features_dir'], config['dataset']['name'], volume_ids=emb_ids)
    if proxy_df is not None and len(proxy_df) == len(emb_ids):
        proxy_df = proxy_df.set_index('volume_id').loc[emb_ids].reset_index()
        U_ssl, T_ssl = proxy_df['U'].values, proxy_df['T'].values
        logger.info("Loaded U and T from SSL proxy CSV")
    else:
        logger.info("Computing U and T from embeddings (no proxy CSV)")
        U_ssl, T_ssl = compute_proxies_from_embeddings(embeddings, k=config['ssl']['knn_k'])

    U_ssl = _minmax(U_ssl)
    T_ssl = _minmax(T_ssl)

    return {
        'train_ids': train_ids,
        'val_ids': val_ids,
        'embeddings': embeddings,
        'volume_ids': emb_ids,
        'U_ssl': U_ssl,
        'T_ssl': T_ssl,
    }


def compute_proxies_from_embeddings(
    embeddings: np.ndarray,
    k: int = 20,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fallback U/T computed directly from embeddings, for datasets without a
    precomputed SSL proxy CSV.

    U = distance of the L2-normalized embedding to the pool centroid
        (further from the centroid = treated as more "uncertain").
    T = inverse mean distance to the k nearest neighbors (density proxy).

    This is O(N^2) in the pool size -- fine for typical AL pool sizes
    (hundreds of volumes), not intended for very large pools.
    """
    N = len(embeddings)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    emb_norm = embeddings / np.clip(norms, 1e-8, None)

    centroid = emb_norm.mean(axis=0)
    U = np.linalg.norm(emb_norm - centroid, axis=1)

    k_eff = min(k, N - 1)
    dists = np.linalg.norm(emb_norm[:, None] - emb_norm[None, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    knn_dists = np.sort(dists, axis=1)[:, :k_eff]
    avg_knn_dist = knn_dists.mean(axis=1)
    T = 1.0 / np.clip(avg_knn_dist, 1e-10, None)

    return U, T


def _minmax(x: np.ndarray) -> np.ndarray:
    xmin, xmax = x.min(), x.max()
    if xmax - xmin < 1e-10:
        return np.ones_like(x) * 0.5
    return (x - xmin) / (xmax - xmin)
