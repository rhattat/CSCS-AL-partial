import numpy as np
import pytest

from cscs_al_partial.acquisition import (
    rank_normalize, compute_iterative_scores,
    compute_gamma_fixed_dcr, compute_gamma_dynamic, compute_dcr,
    compute_prediction_agreement, ConvergenceChecker,
)


def test_rank_normalize_range_and_order():
    scores = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    out = rank_normalize(scores)
    assert out.min() >= 1e-6
    assert out.max() <= 1.0
    # rank order must be preserved
    assert np.argsort(out).tolist() == np.argsort(scores).tolist()


def test_rank_normalize_ties_use_average_rank():
    scores = np.array([1.0, 1.0, 2.0])
    out = rank_normalize(scores)
    assert out[0] == pytest.approx(out[1])


def test_rank_normalize_empty():
    assert rank_normalize(np.array([])).shape == (0,)


def test_iterative_scores_geometric_mean_of_ranks():
    T = np.array([1.0, 2.0, 3.0])
    U = np.array([3.0, 2.0, 1.0])
    scores = compute_iterative_scores(T, U, gamma=0.5)
    T_r = rank_normalize(T)
    U_r = rank_normalize(U)
    expected = np.sqrt(T_r * U_r)
    np.testing.assert_allclose(scores, expected, atol=1e-5)


def test_iterative_scores_gamma_extremes():
    T = np.array([1.0, 2.0, 3.0])
    U = np.array([3.0, 2.0, 1.0])
    scores_t_only = compute_iterative_scores(T, U, gamma=0.0)
    scores_u_only = compute_iterative_scores(T, U, gamma=1.0)
    np.testing.assert_allclose(scores_t_only, rank_normalize(T), atol=1e-5)
    np.testing.assert_allclose(scores_u_only, rank_normalize(U), atol=1e-5)


@pytest.mark.parametrize("dcr", [-1.0, -0.5, 0.0, 0.5, 1.0])
def test_gamma_fixed_dcr_neutral_at_round_zero(dcr):
    # |L_r|=0 => alpha_eff=0 => gamma must be exactly 0.5 regardless of DCR
    gamma = compute_gamma_fixed_dcr(n_labeled=0, n_pool=100, dcr=dcr)
    assert gamma == pytest.approx(0.5)


def test_gamma_fixed_dcr_stays_within_clip_bounds():
    for dcr in np.linspace(-1, 1, 9):
        for n_labeled in [1, 5, 20, 100]:
            gamma = compute_gamma_fixed_dcr(n_labeled, n_pool=100, dcr=dcr)
            assert 0.3 <= gamma <= 0.7


def test_gamma_fixed_dcr_positive_dcr_increases_with_progress():
    g_early = compute_gamma_fixed_dcr(n_labeled=1, n_pool=100, dcr=0.8)
    g_late = compute_gamma_fixed_dcr(n_labeled=90, n_pool=100, dcr=0.8)
    assert g_late > g_early


def test_gamma_dynamic_matches_documented_limits():
    # DCR_r=0 -> gamma=0.5 regardless of budget
    assert compute_gamma_dynamic(0.0, n_labeled=50, n_total=100) == pytest.approx(0.5)


def test_compute_dcr_returns_zero_below_min_pool_size():
    assert compute_dcr(np.array([1.0, 2.0]), np.array([2.0, 1.0])) == 0.0


def test_compute_dcr_matches_spearman_on_larger_pool():
    rng = np.random.RandomState(0)
    u = rng.uniform(size=20)
    t = -u + rng.normal(scale=0.01, size=20)  # strong negative correlation
    dcr = compute_dcr(u, t)
    assert dcr < -0.9


# ══════════════════════════════════════════════════════════════════════
# LABEL-FREE STOPPING (prediction agreement + ConvergenceChecker proxy modes)
# ══════════════════════════════════════════════════════════════════════

def _write_fake_npz(path, seg, n_classes=3):
    """Build a one-hot softmax NPZ whose argmax equals `seg` exactly."""
    onehot = np.eye(n_classes, dtype=np.float32)[seg]  # (D, H, W, C)
    softmax = np.moveaxis(onehot, -1, 0)  # (C, D, H, W)
    np.savez(str(path), probabilities=softmax)


def test_prediction_agreement_identical_predictions_is_100(tmp_path):
    seg = np.zeros((4, 4, 4), dtype=np.int64)
    seg[1:3, 1:3, 1:3] = 1
    prev_dir = tmp_path / "prev"; prev_dir.mkdir()
    curr_dir = tmp_path / "curr"; curr_dir.mkdir()
    _write_fake_npz(prev_dir / "v1.npz", seg)
    _write_fake_npz(curr_dir / "v1.npz", seg)
    agreement = compute_prediction_agreement(prev_dir, curr_dir, ["v1"])
    assert agreement == pytest.approx(100.0)


def test_prediction_agreement_disjoint_foreground_is_low(tmp_path):
    seg_a = np.zeros((4, 4, 4), dtype=np.int64)
    seg_a[0, 0, 0] = 1
    seg_b = np.zeros((4, 4, 4), dtype=np.int64)
    seg_b[3, 3, 3] = 1
    prev_dir = tmp_path / "prev"; prev_dir.mkdir()
    curr_dir = tmp_path / "curr"; curr_dir.mkdir()
    _write_fake_npz(prev_dir / "v1.npz", seg_a)
    _write_fake_npz(curr_dir / "v1.npz", seg_b)
    agreement = compute_prediction_agreement(prev_dir, curr_dir, ["v1"])
    assert agreement == pytest.approx(0.0)


def test_prediction_agreement_missing_common_ids_returns_none(tmp_path):
    prev_dir = tmp_path / "prev"; prev_dir.mkdir()
    curr_dir = tmp_path / "curr"; curr_dir.mkdir()
    assert compute_prediction_agreement(prev_dir, curr_dir, ["ghost"]) is None


def test_convergence_checker_supervised_mode_unchanged():
    # default mode="supervised" behaves exactly as before update_proxy existed
    checker = ConvergenceChecker(delta_pp=0.5, patience=2, max_rounds=9)
    assert checker.update(80.0) is False
    assert checker.update(80.1) is False  # delta=0.1 < 0.5 -> plateau_count=1
    assert checker.update(80.2) is True   # plateau_count=2 >= patience -> stop
    assert "Plateau" in checker.reason


def test_convergence_checker_update_proxy_requires_matching_mode():
    checker = ConvergenceChecker(mode="supervised")
    with pytest.raises(ValueError):
        checker.update_proxy(prediction_agreement=99.0)


def test_convergence_checker_stability_mode_stops_on_high_agreement():
    checker = ConvergenceChecker(patience=2, max_rounds=9, mode="stability", stability_delta_pp=0.5)
    assert checker.update_proxy(prediction_agreement=99.8) is False  # plateau_count=1
    assert checker.update_proxy(prediction_agreement=99.9) is True   # plateau_count=2 -> stop
    assert "agreement" in checker.reason


def test_convergence_checker_entropy_mode_stops_on_stable_uncertainty():
    checker = ConvergenceChecker(patience=2, max_rounds=9, mode="entropy", entropy_delta_pct=2.0)
    assert checker.update_proxy(pool_uncertainty=0.50) is False  # first sample, no delta yet
    assert checker.update_proxy(pool_uncertainty=0.505) is False  # ~1% change -> plateau_count=1
    assert checker.update_proxy(pool_uncertainty=0.507) is True   # ~0.4% change -> plateau_count=2 -> stop
    assert "uncertainty" in checker.reason


def test_convergence_checker_proxy_mode_round_zero_has_no_signal():
    # round 0: no task-model predictions yet -> signal=None must not raise
    # and must not count as a plateau round
    checker = ConvergenceChecker(patience=2, max_rounds=9, mode="stability")
    assert checker.update_proxy(pool_remaining=50, round_budget=5) is False
    assert checker._plateau_count == 0


def test_convergence_checker_proxy_mode_respects_pool_exhaustion():
    checker = ConvergenceChecker(patience=2, max_rounds=9, mode="entropy")
    assert checker.update_proxy(pool_remaining=1, round_budget=2, pool_uncertainty=0.5) is True
    assert "exhausted" in checker.reason
