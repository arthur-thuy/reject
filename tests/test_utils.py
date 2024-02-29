#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Testing script for utils."""
# =============================================================================
# NOTE: run with `poetry run pytest tests/`

import pytest
import numpy as np

from reject.utils import generate_synthetic_output, aggregate_preds, compute_correct


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


class TestGenerateSyntheticOutput:
    @pytest.mark.parametrize(
        "num_samples, num_observations",
        [
            (5, 100),
            (10, 200),
        ],
    )
    def test_unit(self, num_samples, num_observations):
        y_pred_all, y_true_all = generate_synthetic_output(
            num_samples, num_observations
        )
        assert y_true_all.shape == (2 * num_observations,)
        assert y_pred_all.shape == (2 * num_observations, num_samples, 10)


class TestAggregatePreds:
    @pytest.mark.parametrize(
        "y_stack",
        [
            np.array(
                [
                    [[0.28, 0.72], [0.15, 0.85], [0.2, 0.8]],
                    [[0.12, 0.88], [0.22, 0.78], [0.08, 0.92]],
                ]
            ),
            np.array([[0.28, 0.72], [0.12, 0.88]]),
        ],
    )
    def test_unit(self, y_stack):
        y_stack, y_mean, y_label = aggregate_preds(y_stack)
        assert y_stack.ndim == 3
        assert y_mean.ndim == 2
        assert y_label.ndim == 1

    def test_error(self, y_pred_label):
        with pytest.raises(ValueError):
            aggregate_preds(y_pred_label)


class TestComputeCorrect:
    @pytest.mark.parametrize(
        "y_pred_label, y_true_label, is_correct",
        [
            (np.array([0.0, 0.0, 1.0, 0.0, 0.0]), np.array([0.0, 1.0, 1.0, 0.0, 1.0]), np.array([True, False, True, True, False])),
            (np.array([0.0, 0.0, 1.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0, 0.0, 1.0]), np.array([False, False, True, True, False])),
        ],
    )
    def test_unit(self, y_pred_label, y_true_label, is_correct):
        is_correct_actual = compute_correct(y_pred_label, y_true_label)
        assert np.array_equal(is_correct, is_correct_actual)

    def test_error(self):
        with pytest.raises(ValueError):
            compute_correct(np.zeros((3,)), np.zeros((4,)))
            
        with pytest.raises(ValueError):
            compute_correct(np.zeros((3, 3)), np.zeros((3,)))   

        with pytest.raises(ValueError):
             compute_correct(np.zeros((3,)), np.zeros((3,3,3,3)))
