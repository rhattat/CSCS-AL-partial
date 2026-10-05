import numpy as np
import pytest

from cscs_al_partial.evaluation import dice_coefficient, hd95, evaluate_volume


def test_dice_perfect_overlap():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[2:6, 2:6, 2:6] = True
    assert dice_coefficient(mask, mask) == pytest.approx(1.0)


def test_dice_no_overlap():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    a[0:2, 0:2, 0:2] = True
    b[8:10, 8:10, 8:10] = True
    assert dice_coefficient(a, b) == pytest.approx(0.0)


def test_dice_both_empty_is_one():
    empty = np.zeros((5, 5, 5), dtype=bool)
    assert dice_coefficient(empty, empty) == pytest.approx(1.0)


def test_dice_partial_overlap_known_value():
    a = np.zeros((4, 4, 1), dtype=bool)
    b = np.zeros((4, 4, 1), dtype=bool)
    a[0:2, :, :] = True  # 8 voxels
    b[1:3, :, :] = True  # 8 voxels, overlap = 4 voxels (row 1)
    # Dice = 2*4 / (8+8) = 0.5
    assert dice_coefficient(a, b) == pytest.approx(0.5)


def test_hd95_empty_mask_returns_zero():
    a = np.zeros((10, 10, 10), dtype=bool)
    b = np.zeros((10, 10, 10), dtype=bool)
    b[5, 5, 5] = True
    assert hd95(a, b, spacing=(1.0, 1.0, 1.0)) == 0.0
    assert hd95(b, a, spacing=(1.0, 1.0, 1.0)) == 0.0


def test_hd95_identical_masks_is_zero():
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[3:7, 3:7, 3:7] = True
    assert hd95(mask, mask, spacing=(1.0, 1.0, 1.0)) == pytest.approx(0.0, abs=1e-6)


def test_hd95_respects_anisotropic_spacing_axis_order():
    # Two single-voxel "surfaces" offset only along axis 0.
    a = np.zeros((10, 3, 3), dtype=bool)
    b = np.zeros((10, 3, 3), dtype=bool)
    a[2, 1, 1] = True
    b[5, 1, 1] = True  # 3 voxels apart along axis 0
    # spacing along axis 0 is 4.0mm -> distance should scale with it, not axis 1/2 spacing
    d_axis0_stretched = hd95(a, b, spacing=(4.0, 1.0, 1.0))
    d_isotropic = hd95(a, b, spacing=(1.0, 1.0, 1.0))
    assert d_axis0_stretched == pytest.approx(4.0 * d_isotropic, rel=1e-6)


def test_evaluate_volume_returns_expected_keys():
    pred = np.zeros((6, 6, 6), dtype=np.int16)
    gt = np.zeros((6, 6, 6), dtype=np.int16)
    pred[0:3, 0:3, 0:3] = 1
    gt[0:3, 0:3, 0:3] = 1
    pred[3:6, 3:6, 3:6] = 2
    gt[3:6, 3:6, 3:6] = 2
    out = evaluate_volume(pred, gt, spacing=(1.0, 1.0, 1.0), labels=[0, 1, 2])
    assert set(["dice_0", "hd95_0", "dice_1", "hd95_1", "dice_2", "hd95_2",
                "dice_mean_fg", "hd95_mean_fg"]).issubset(out.keys())
    assert out["dice_1"] == pytest.approx(1.0)
    assert out["dice_2"] == pytest.approx(1.0)
    assert out["dice_mean_fg"] == pytest.approx(1.0)
