#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed March 6 2024
# =============================================================================
"""Testing script for diversity."""
# =============================================================================
# NOTE: run with `poetry run pytest tests/`

import warnings
import numpy as np
import pytest

from reject.diversity import (
    compute_pairwise_diversity,
    _dq_divide,
    diversity_quality_score,
    diversity_score,
    UndefinedMetricWarning,
    _input_array,
    _warn_dq,
    _check_zero_division,
    _diversity_quality_score_base,
)


class TestDqDivide:
    @pytest.mark.parametrize(
        "numerator,denominator,zero_division,expected",
        [
            (0.0, 1.0, "warn", 0.0),
            (1.0, 0.0, "warn", 0.0),
            (0.0, 0.0, "warn", 0.0),
            (1.0, 1.0, "warn", 1.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 0.0),
            (1.0, 0.0, np.nan, np.nan),
            (0.0, 0.0, np.nan, np.nan),
        ],
    )
    def test_output(self, numerator, denominator, zero_division, expected):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            numerator = _input_array(numerator)
            denominator = _input_array(denominator)
            expected = _input_array(expected)
            actual = _dq_divide(numerator, denominator, zero_division=zero_division)
            np.testing.assert_allclose(
                actual, expected, err_msg="DQ divide computed incorrectly!"
            )

    @pytest.mark.parametrize(
        "numerator,denominator,warning",
        [
            (0.0, 1.0, False),
            (1.0, 0.0, True),
            (0.0, 0.0, True),
            (1.0, 1.0, False),
        ],
    )
    def test_warning(self, numerator, denominator, warning):
        numerator = _input_array(numerator)
        denominator = _input_array(denominator)
        if warning:
            with pytest.warns(UndefinedMetricWarning):
                _dq_divide(numerator, denominator, zero_division="warn")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                _dq_divide(numerator, denominator, zero_division="warn")


class TestWarnDQ:
    def test_warning(self):
        with pytest.warns(UndefinedMetricWarning):
            _warn_dq(result_size=1)


class TestCheckZeroDivision:
    @pytest.mark.parametrize(
        "zero_division,expected",
        [
            (0.0, 0.0),
            (np.nan, np.nan),
            ("warn", 0.0),
        ],
    )
    def test_unit(self, zero_division, expected):
        np.testing.assert_allclose(_check_zero_division(zero_division), expected)


class TestInputArray:
    @pytest.mark.parametrize(
        "a,expected",
        [
            ([1, 2, 3], np.array([1, 2, 3], dtype=np.float64)),
            (np.array([1, 2, 3]), np.array([1, 2, 3])),
            (1, np.array([1], dtype=np.float64)),
        ],
    )
    def test_unit(self, a, expected):
        np.testing.assert_allclose(_input_array(a), expected)


class TestDiversityQualityScoreBase:
    @pytest.mark.parametrize(
        "diversity_id,diversity_ood",
        [(0.1, 0.1), (0.9, 0.9), (0.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
    )
    def test_unit(self, diversity_id, diversity_ood):
        with_function = _diversity_quality_score_base(
            diversity_id=diversity_id, diversity_ood=diversity_ood, beta_ood=1.0
        )
        manual = (
            2
            * ((1 - diversity_id) * diversity_ood)
            / ((1 - diversity_id) + diversity_ood)
        )
        np.testing.assert_allclose(with_function, manual)

    @pytest.mark.parametrize(
        "diversity_id,diversity_ood,beta_ood, expected",
        [
            (0.1, 0.9, 0.0, 0.1),
            (0.1, 0.9, np.inf, 0.9),
        ],
    )
    def test_extreme(self, diversity_id, diversity_ood, beta_ood, expected):
        output = _diversity_quality_score_base(
            diversity_id=diversity_id, diversity_ood=diversity_ood, beta_ood=beta_ood
        )
        np.testing.assert_allclose(output, expected)

    @pytest.mark.parametrize("keepdims,expected_float", [(True, False), (False, True)])
    def test_keepdims(self, keepdims, expected_float):
        output = _diversity_quality_score_base(
            diversity_id=0.1, diversity_ood=0.9, keepdims=keepdims
        )
        assert isinstance(output, float if expected_float else np.ndarray)


class TestComputePairwiseDiversity:
    @pytest.mark.parametrize(
        "other_member_label,base_member_label,expected",
        [
            (np.array([0, 0, 0]), np.array([1, 1, 1]), 1.0),
            (np.array([0, 0, 0]), np.array([0, 0, 0]), 0.0),
            (np.array([0, 0, 0]), np.array([0, 1, 0]), 1 / 3),
        ],
    )
    def test_diversity(self, other_member_label, base_member_label, expected):
        output_div = compute_pairwise_diversity(
            other_member_label=other_member_label, base_member_label=base_member_label
        )
        np.testing.assert_allclose(output_div, expected)


class TestDiversityScore:
    @pytest.mark.parametrize(
        "average,keepdims,expected_type,expected_shape",
        [
            (True, True, np.ndarray, (1,)),
            (False, False, np.ndarray, (5,)),
            (True, False, float, None),
            (False, True, np.ndarray, (5,)),
        ],
    )
    def test_shape(self, average, keepdims, expected_type, expected_shape):
        example_pred = np.zeros((10, 5, 3))
        output = diversity_score(
            y_pred=example_pred, average=average, keepdims=keepdims
        )
        assert isinstance(output, expected_type)
        if not isinstance(output, float):
            assert output.shape == expected_shape


class TestDiversityQualityScore:
    def test_warning(self):
        with pytest.raises(ValueError):
            diversity_quality_score(
                y_pred_id=np.zeros((10, 5, 3)),
                y_pred_ood=np.zeros((10, 5, 3)),
                zero_division="wrong",
            )
