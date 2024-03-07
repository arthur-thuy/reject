#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Testing script for uncertainty."""
# =============================================================================
# NOTE: run with `poetry run pytest tests/`

import numpy as np
import pytest

from reject.uncertainty import compute_confidence, compute_uncertainty


@pytest.fixture
def y_stack():
    return np.array(
        [
            [[0.28, 0.72], [0.15, 0.85], [0.2, 0.8]],
            [[0.12, 0.88], [0.22, 0.78], [0.08, 0.92]],
        ]
    )


@pytest.fixture
def y_pred_label():
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0])


class TestComputeConfidence:
    def test_unit(self, y_stack):
        actual = compute_confidence(y_stack)
        expected_y_mean = np.array(
            [
                [(0.28 + 0.15 + 0.2) / 3, (0.72 + 0.85 + 0.8) / 3],
                [(0.12 + 0.22 + 0.08) / 3, (0.88 + 0.78 + 0.92) / 3],
            ]
        )
        expected = np.max(expected_y_mean, axis=-1)
        np.testing.assert_allclose(
            actual, expected, err_msg="confidence computed incorrectly!"
        )

    def test_error(self, y_pred_label):
        with pytest.raises(ValueError):
            compute_confidence(y_pred_label)


class TestComputeUncertainty:
    @pytest.mark.parametrize(
        "y_pred, unc_tuple",
        [
            (
                np.array([[[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]]]),
                (np.array([0.0]), np.array([0.0]), np.array([0.0])),
            ),
            (
                np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]]),
                (np.array([1.0]), np.array([0.0]), np.array([1.0])),
            ),
            (
                np.array([[[0.5, 0.5], [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]]]),
                (np.array([1.0]), np.array([1.0]), np.array([0.0])),
            ),
        ],
    )
    def test_unit(self, y_pred, unc_tuple):
        actual_unc_all = compute_uncertainty(y_pred, unc_type=None)
        print(actual_unc_all)
        assert actual_unc_all["TU"] == pytest.approx(unc_tuple[0]), (
            f"Total uncertainty should be {float(unc_tuple[0][0])},"
            f" is {float(actual_unc_all['TU'][0])}."
        )
        assert actual_unc_all["AU"] == pytest.approx(unc_tuple[1]), (
            f"Aleatoric uncertainty should be {float(unc_tuple[1][0])},"
            f" is {float(actual_unc_all['AU'][0])}."
        )
        assert actual_unc_all["EU"] == pytest.approx(unc_tuple[2]), (
            f"Epistemic uncertainty should be {float(unc_tuple[2][0])},"
            f" is {float(actual_unc_all['EU'][0])}."
        )

    def test_error(self):
        with pytest.raises(ValueError):
            compute_uncertainty(np.zeros((5)), unc_type=None)
        with pytest.raises(ValueError):
            compute_uncertainty(np.zeros((5, 128, 10)), unc_type="wrong")
