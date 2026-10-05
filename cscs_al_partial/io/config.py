"""
Config Loader
==============
Loads and merges YAML configuration files into a single dict: shared
defaults, a per-dataset config, a strategy registry, and a paths file
(all real filesystem paths -- keep this file outside version control or
fill it with placeholders, see `configs/paths.example.yaml`).

Usage:
    config = load_config(dataset="my_dataset", config_dir="configs/")
    # config contains: paths, training/convergence/cscs/ssl/uncertainty
    # params, dataset params, budgets, strategies
"""

import yaml
from pathlib import Path
from typing import Any, Dict, Optional


def _load_yaml(path: Path) -> dict:
    """Load a YAML file. Raises if missing -- config files should exist."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, 'r') as f:
        return yaml.safe_load(f) or {}


def load_config(dataset: str, config_dir: str) -> Dict[str, Any]:
    """
    Load and merge all configs for a given dataset.

    Expects `config_dir` to contain:
        default.yaml     -- shared training/convergence/cscs/ssl params
        paths.yaml        -- filesystem paths (dataset-specific, gitignored)
        strategies.yaml   -- strategy name -> cold-start/acquisition wiring
        {dataset}.yaml    -- dataset-specific fields (labels, budgets, ...)

    Returns a dict with keys:
        paths, training, convergence, cscs, ssl, uncertainty, seeds,
        retraining_modes, dataset, budgets, dcr_precomputed, strategies,
        source_images, source_labels, train_file, val_file,
        embeddings_dir, ssl_features_dir
    """
    config_dir = Path(config_dir)

    paths_cfg = _load_yaml(config_dir / "paths.yaml")
    default_cfg = _load_yaml(config_dir / "default.yaml")
    dataset_cfg = _load_yaml(config_dir / f"{dataset}.yaml")
    strategies_cfg = _load_yaml(config_dir / "strategies.yaml")

    config: Dict[str, Any] = {'paths': paths_cfg}

    for key in ['training', 'convergence', 'cscs', 'ssl', 'uncertainty',
                'seeds', 'retraining_modes']:
        if key in default_cfg:
            config[key] = default_cfg[key]

    config['dataset'] = dataset_cfg.get('dataset', {})
    config['budgets'] = dataset_cfg.get('budgets', {})
    config['dcr_precomputed'] = dataset_cfg.get('dcr_precomputed', None)
    config['strategies'] = strategies_cfg.get('strategies', {})

    ds_key = dataset.lower()
    sources = paths_cfg.get('sources', {}).get(ds_key, {})
    config['source_images'] = sources.get('images', '')
    config['source_labels'] = sources.get('labels', '')
    config['train_file'] = sources.get('train_file', '')
    config['val_file'] = sources.get('val_file', '')
    config['embeddings_dir'] = paths_cfg.get('embeddings', {}).get(ds_key, '')
    config['ssl_features_dir'] = paths_cfg.get('ssl_features', {}).get(ds_key, '')

    return config


def get_strategy_config(config: dict, strategy_name: str) -> dict:
    """Get the config entry for a specific strategy name."""
    strategies = config.get('strategies', {})
    if strategy_name not in strategies:
        available = list(strategies.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    return strategies[strategy_name]
