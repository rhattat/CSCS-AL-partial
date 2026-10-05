import numpy as np
import pytest

from cscs_al_partial.cold_start import CSCSColdStart, RandomColdStart, ColdStartResult


def _synthetic_pool(N=30, d=8, seed=0):
    rng = np.random.RandomState(seed)
    pool_ids = [f"v{i}" for i in range(N)]
    embeddings = rng.normal(size=(N, d)).astype(np.float32)
    U_ssl = rng.uniform(0, 1, size=N)
    T_ssl = rng.uniform(0, 1, size=N)
    return pool_ids, embeddings, U_ssl, T_ssl


def test_cscs_cold_start_selects_exactly_k_unique_ids():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = CSCSColdStart(config={"cscs": {"clip_min": 0.3, "clip_max": 0.7}})
    result = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=42)
    assert isinstance(result, ColdStartResult)
    assert len(result.selected_ids) == 6
    assert len(set(result.selected_ids)) == 6
    assert all(sid in pool_ids for sid in result.selected_ids)


def test_cscs_cold_start_gamma_within_clip_bounds():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = CSCSColdStart(config={"cscs": {"clip_min": 0.3, "clip_max": 0.7}})
    result = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=42)
    assert 0.3 <= result.gamma <= 0.7
    assert -1.0 <= result.dcr <= 1.0


def test_cscs_cold_start_deterministic_given_seed():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = CSCSColdStart(config={"cscs": {}})
    r1 = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=5, seed=7)
    r2 = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=5, seed=7)
    assert r1.selected_ids == r2.selected_ids
    assert r1.gamma == pytest.approx(r2.gamma)


def test_cscs_cold_start_respects_precomputed_dcr_override():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = CSCSColdStart(config={"cscs": {}, "dcr_precomputed": 0.9})
    result = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=5, seed=1)
    assert result.dcr == pytest.approx(0.9)


def test_cscs_cold_start_k_larger_than_pool_is_clamped():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool(N=5)
    selector = CSCSColdStart(config={"cscs": {}})
    result = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=100, seed=1)
    assert len(result.selected_ids) == 5


def test_random_cold_start_selects_k_unique_ids():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = RandomColdStart(config={})
    result = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=0)
    assert len(result.selected_ids) == 6
    assert len(set(result.selected_ids)) == 6


def test_random_cold_start_deterministic_given_seed():
    pool_ids, embeddings, U_ssl, T_ssl = _synthetic_pool()
    selector = RandomColdStart(config={})
    r1 = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=3)
    r2 = selector.select(pool_ids, embeddings, U_ssl, T_ssl, k=6, seed=3)
    assert r1.selected_ids == r2.selected_ids
