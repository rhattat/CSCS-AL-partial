"""
Strategy Registry
==================
Maps a strategy name to a cold-start selector class. No logic here -- just
wiring.

This release ships one iterative acquisition path (the function-based
`cscs_al_partial.acquisition` module: `compute_iterative_scores` +
`compute_gamma_fixed_dcr` / `compute_gamma_dynamic`), which is what the
"cscs_curriculum" strategy below refers to. It is not itself a class in this
registry -- swap in your own round-level policy while keeping the cold-start
class fixed, or vice versa, by calling the pieces in `cold_start` and
`acquisition` directly instead of going through a registry entry.

Usage:
    cs_cls = get_cold_start_class("cscs_curriculum")
    cold_start = cs_cls(config)
    result = cold_start.select(pool_ids, embeddings, U_ssl, T_ssl, k=k, seed=seed)
    # then drive rounds r >= 1 with cscs_al_partial.acquisition directly
"""

from typing import Dict, List, Type

from cscs_al_partial.cold_start.base import ColdStartSelector
from cscs_al_partial.cold_start.random_selection import RandomColdStart
from cscs_al_partial.cold_start.probcover import ProbCoverColdStart
from cscs_al_partial.cold_start.cscs import CSCSColdStart

# name -> cold-start class. The name also documents which round-level
# acquisition it was paired with in the paper (see docstring above);
# this package only ships the CSCS-AL iterative-scoring acquisition path,
# so every strategy that isn't "cscs_curriculum" is a cold-start-only
# ablation baseline useful for comparison, not a full alternative pipeline.
_REGISTRY: Dict[str, Type[ColdStartSelector]] = {
    "randomCS":        RandomColdStart,
    "probcover":       ProbCoverColdStart,
    "cscs_curriculum": CSCSColdStart,
}


def get_cold_start_class(strategy_name: str) -> Type[ColdStartSelector]:
    """
    Get the ColdStartSelector class for a named strategy.

    Args:
        strategy_name: one of the keys in `list_strategies()`

    Returns:
        ColdStartSelector subclass
    """
    if strategy_name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise ValueError(f"Unknown strategy '{strategy_name}'. Available: {available}")
    return _REGISTRY[strategy_name]


def list_strategies() -> List[str]:
    """List all available strategy names."""
    return list(_REGISTRY.keys())
