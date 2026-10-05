import numpy as np
import pytest

from cscs_al_partial.partial_correction import (
    rank_patches_by_entropy,
    filter_patches_with_foreground,
    apply_partial_correction,
    build_weight_map,
    make_provenance_map_from_masks,
    summarize_provenance,
    HUMAN_FULL,
    HUMAN_CORRECTED,
    MODEL_UNCHECKED,
    EXCLUDED,
)


def test_rank_patches_sorted_descending():
    entropy_map = np.zeros((16, 16, 16), dtype=np.float32)
    entropy_map[0:8, 0:8, 0:8] = 5.0  # high-entropy patch
    entropy_map[8:16, 8:16, 8:16] = 1.0
    patches = rank_patches_by_entropy(entropy_map, patch_size=8)
    scores = [p["score"] for p in patches]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(5.0)


def test_rank_patches_covers_whole_volume_with_uneven_size():
    entropy_map = np.zeros((10, 10, 10), dtype=np.float32)
    patches = rank_patches_by_entropy(entropy_map, patch_size=8)
    total_voxels = sum(p["n_voxels"] for p in patches)
    assert total_voxels == entropy_map.size


def test_filter_patches_with_foreground_splits_correctly():
    entropy_map = np.zeros((8, 8, 8), dtype=np.float32)
    patches = rank_patches_by_entropy(entropy_map, patch_size=4)
    softmax = np.zeros((2, 8, 8, 8), dtype=np.float32)
    softmax[0] = 1.0  # background everywhere by default
    softmax[1, 0:4, 0:4, 0:4] = 2.0  # first patch predicted foreground
    softmax[0, 0:4, 0:4, 0:4] = 0.0
    fg, bg = filter_patches_with_foreground(patches, softmax, min_fg_voxels=10)
    assert len(fg) == 1
    assert len(bg) == 7
    assert fg[0]["n_fg_voxels"] == 64


def test_apply_partial_correction_respects_patch_count_budget():
    D, H, W = 8, 8, 8
    prediction = np.zeros((D, H, W), dtype=np.int16)
    reference = np.ones((D, H, W), dtype=np.int16)
    entropy_map = np.random.RandomState(0).uniform(size=(D, H, W)).astype(np.float32)
    patches = rank_patches_by_entropy(entropy_map, patch_size=4)  # 8 patches total
    mixed, prov, stats = apply_partial_correction(prediction, reference, patches, budget=0.25)
    assert stats["n_patches_total"] == 8
    assert stats["n_patches_corrected"] == 2  # round(0.25 * 8)
    assert (prov == HUMAN_CORRECTED).sum() == stats["n_voxels_corrected"]
    assert (prov == MODEL_UNCHECKED).sum() == prediction.size - stats["n_voxels_corrected"]
    # corrected region matches reference, uncorrected region matches prediction
    assert np.all(mixed[prov == HUMAN_CORRECTED] == 1)
    assert np.all(mixed[prov == MODEL_UNCHECKED] == 0)


def test_apply_partial_correction_minimum_one_patch():
    D, H, W = 4, 4, 4
    prediction = np.zeros((D, H, W), dtype=np.int16)
    reference = np.ones((D, H, W), dtype=np.int16)
    patches = rank_patches_by_entropy(np.zeros((D, H, W), dtype=np.float32), patch_size=4)
    _, _, stats = apply_partial_correction(prediction, reference, patches, budget=0.01)
    assert stats["n_patches_corrected"] == 1


def test_make_provenance_map_priority_excluded_over_human_full():
    shape = (2, 2, 2)
    human_full = np.ones(shape, dtype=bool)
    corrected = np.zeros(shape, dtype=bool)
    excluded = np.zeros(shape, dtype=bool)
    excluded[0, 0, 0] = True
    pmap = make_provenance_map_from_masks(shape, human_full, corrected, excluded)
    assert pmap[0, 0, 0] == EXCLUDED
    assert pmap[1, 1, 1] == HUMAN_FULL


def test_build_weight_map_alpha_zero_ignores_model_unchecked():
    provenance_map = np.array([HUMAN_FULL, HUMAN_CORRECTED, MODEL_UNCHECKED, EXCLUDED])
    confidence_map = np.array([0.9, 0.9, 0.99, 0.99])
    w = build_weight_map(provenance_map, confidence_map, alpha=0.0)
    np.testing.assert_allclose(w, [1.0, 1.0, 0.0, 0.0])


def test_build_weight_map_alpha_positive_ramps_with_confidence():
    provenance_map = np.array([MODEL_UNCHECKED, MODEL_UNCHECKED, MODEL_UNCHECKED])
    confidence_map = np.array([0.5, 0.70, 1.0])  # below tau_low, at tau_low, max
    w = build_weight_map(provenance_map, confidence_map, alpha=0.2, tau_low=0.70)
    assert w[0] == pytest.approx(0.0)
    assert w[1] == pytest.approx(0.0)
    assert w[2] == pytest.approx(0.2)


def test_summarize_provenance_counts_and_percentages():
    provenance_map = np.array([HUMAN_FULL, MODEL_UNCHECKED, MODEL_UNCHECKED, EXCLUDED])
    weight_map = np.array([1.0, 0.0, 0.0, 0.0])
    summary = summarize_provenance(provenance_map, weight_map)
    assert summary["n_voxels_human_full"] == 1
    assert summary["n_voxels_model_unchecked"] == 2
    assert summary["n_voxels_excluded"] == 1
    assert summary["pct_weight_zero"] == pytest.approx(75.0)
    assert summary["pct_weight_one"] == pytest.approx(25.0)
