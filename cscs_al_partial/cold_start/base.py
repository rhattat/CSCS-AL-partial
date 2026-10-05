"""
Cold-Start Selector — Abstract Base Class
==========================================
All cold-start methods must implement `select()`.

The ColdStartResult carries metadata (DCR, gamma, etc.) that downstream
acquisition policies can use (e.g. the CSCS-AL iterative scorer needs the
DCR computed at cold-start).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np


@dataclass
class ColdStartResult:
    """Result of a cold-start selection."""
    selected_ids: List[str]           # K_0 selected volume IDs
    gamma: Optional[float] = None     # gamma from CSCS (None for random/probcover)
    dcr: Optional[float] = None       # DCR from CSCS (None for random/probcover)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ColdStartSelector(ABC):
    """
    Abstract base class for cold-start selection strategies.

    All selectors receive:
        - pool_ids: list of candidate volume IDs
        - embeddings: (N, d) array aligned with pool_ids
        - U_ssl: (N,) SSL uncertainty scores aligned with pool_ids
        - T_ssl: (N,) SSL typicality scores aligned with pool_ids
        - k: number of volumes to select
        - seed: random seed for reproducibility

    Not all selectors use all inputs (e.g. random ignores embeddings/U/T).
    """

    def __init__(self, config: dict):
        """
        Args:
            config: merged config dict with keys like cscs.*, ssl.*, etc.
        """
        self.config = config

    @abstractmethod
    def select(
        self,
        pool_ids: List[str],
        embeddings: np.ndarray,
        U_ssl: np.ndarray,
        T_ssl: np.ndarray,
        k: int,
        seed: int,
    ) -> ColdStartResult:
        """
        Select K volumes from the unlabeled pool.

        Args:
            pool_ids: (N,) list of volume IDs
            embeddings: (N, d) SSL embeddings
            U_ssl: (N,) uncertainty scores in [0, 1]
            T_ssl: (N,) typicality scores in [0, 1]
            k: budget K_0
            seed: random seed

        Returns:
            ColdStartResult with selected IDs and metadata
        """
        ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
