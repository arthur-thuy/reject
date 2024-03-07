#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed March 6 2024
# =============================================================================
"""Module for diversity."""
# =============================================================================

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from reject.utils import aggregate_preds


def compute_pairwise_diversity(
    other_member_label: NDArray,
    base_member_label: NDArray,
) -> float:
    """Calculate diversity between two members of an ensemble.

    Parameters
    ----------
    other_member_label : NDArray
        Predicted labels of other ensemble member.
    base_member_label : NDArray
        Predicted labels of base ensemble member.

    Returns
    -------
    float
        Diversity score.
    """
    diversity = 1 - np.mean(base_member_label == other_member_label)
    return diversity


def _dq_divide(numerator, denominator, zero_division="warn"):
    """Perform division and handles divide-by-zero.

    On zero-division, sets the corresponding result elements equal to
    0 or np.nan (according to ``zero_division``). Plus, if
    ``zero_division != "warn"`` raises a warning.
    """
    mask = denominator == 0.0
    denominator = denominator.copy()
    denominator[mask] = 1  # avoid infs/nans
    result = numerator / denominator

    if not np.any(mask):
        return result

    # set those with 0 denominator to `zero_division`, and 0 when "warn"
    zero_division_value = _check_zero_division(zero_division)
    result[mask] = zero_division_value

    if zero_division != "warn":
        return result

    # build appropriate warning
    _warn_dq(len(result))

    return result


def _warn_dq(result_size):
    """Warns about ill-defined DQ-score."""
    msg = (
        "DQ-score ill-defined and being set to 0.0 {0} ID diversity of 1.0 and OOD"
        " diversity of 0.0. Use `zero_division` parameter to control"
        " this behavior.".format("due to" if result_size == 1 else "in samples with")
    )
    warnings.warn(msg, UndefinedMetricWarning, stacklevel=2)


def _check_zero_division(zero_division):
    """Return replacement value for zero division."""
    if isinstance(zero_division, str) and zero_division == "warn":
        return np.float64(0.0)
    elif isinstance(zero_division, (int, float)) and zero_division == 0:
        return np.float64(zero_division)
    else:  # np.isnan(zero_division)
        return np.nan


class UndefinedMetricWarning(UserWarning):
    """Warning used when the metric is invalid."""


def _input_array(a: ArrayLike) -> NDArray:
    """Convert input to numpy array.

    Parameters
    ----------
    a : ArrayLike
        Input array to convert

    Returns
    -------
    NDArray
        Numpy array
    """
    if not isinstance(a, np.ndarray):
        a = np.array(a, dtype=np.float64)
    a = np.atleast_1d(a)
    return a


def _diversity_quality_score_base(
    diversity_id: ArrayLike,
    diversity_ood: ArrayLike,
    beta_ood: float = 1.0,
    zero_division: Any = "warn",
    keepdims: bool = False,
) -> NDArray:
    """Compute Diversity Quality score based on diversity scores.

    Parameters
    ----------
    diversity_id : ArrayLike
        Diversity score on the in-distribution (ID) set
    diversity_ood : ArrayLike
        Diversity score on the out-of-distribution (OOD) set
    beta_ood : float, optional
        OOD score is considered `beta_ood` times as imortant as ID score, by default 1.0
    zero_division : str, optional
        How to handle division by zero {"warn", 0.0, np.nan}, by default "warn"
    keepdims : bool, optional
        If True, the output will keep the same dimensions as the input, by default False

    Returns
    -------
    NDArray
        Diversity-quality score
    """
    diversity_id = _input_array(diversity_id)
    diversity_ood = _input_array(diversity_ood)

    if np.isposinf(beta_ood):
        score = diversity_ood
    elif beta_ood == 0:
        score = diversity_id
    else:
        score = _dq_divide(
            (1.0 + beta_ood**2) * ((1.0 - diversity_id) * diversity_ood),
            (beta_ood**2 * (1.0 - diversity_id) + diversity_ood),
            zero_division=zero_division,
        )
    if score.size == 1 and not keepdims:
        score = score.item()
    return score


def diversity_score(
    y_pred: ArrayLike,
    average: bool = False,
    keepdims: bool = False,
) -> ArrayLike:
    """Compute diversity score in some prediction.

    Parameters
    ----------
    y_pred : ArrayLike
        Output predictions
    average : bool, optional
        Whether to average over the ensemble members, by default False
    keepdims : bool, optional
        Whether to keep the output dimensions, by default False

    Returns
    -------
    ArrayLike
        Diversity score
    """
    y_pred = _input_array(y_pred)
    y_pred_stack, _, y_pred_label = aggregate_preds(y_pred)

    # compute averaged model and compare to all
    base_member_label = y_pred_label
    other_members_label = np.argmax(y_pred_stack, axis=-1)

    # calculate diversity for all columns
    diversity = np.apply_along_axis(
        compute_pairwise_diversity,
        axis=0,
        arr=other_members_label,
        base_member_label=base_member_label,
    )
    if average:
        diversity = np.mean(diversity, keepdims=keepdims)
    return diversity


def diversity_quality_score(
    y_pred_id: ArrayLike,
    y_pred_ood: ArrayLike,
    beta_ood: float = 1.0,
    average: bool = False,
    zero_division: str = "warn",
    keepdims: bool = False,
) -> ArrayLike:
    """Compute Diversity Quality score based on output predictions.

    Parameters
    ----------
    y_pred_id : ArrayLike
        Predictions on the ID set
    y_pred_ood : ArrayLike
        Predictions on the OOD set
    beta_ood : float, optional
        OOD score is considered `beta_ood` times as imortant as ID score, by default 1.0
    average : bool, optional
        Whether to average over the ensemble members, by default False
    zero_division : str, optional
        How to handle division by zero {"warn", 0.0, np.nan}, by default "warn"
    keepdims : bool, optional
        Whether to keep the output dimensions, by default False

    Returns
    -------
    ArrayLike
        Diversity Quality score
    """
    if zero_division not in ["warn", 0.0, np.nan]:
        raise ValueError(
            "Invalid zero_division value. Expected one of {{'warn', 0.0, np.nan}},"
            " got {0}".format(zero_division)
        )
    y_pred_id = _input_array(y_pred_id)
    y_pred_ood = _input_array(y_pred_ood)

    # get diversity scores
    diversity_id = diversity_score(y_pred_id, average=average, keepdims=keepdims)
    diversity_ood = diversity_score(y_pred_ood, average=average, keepdims=keepdims)

    # compute diversity quality score
    score = _diversity_quality_score_base(
        diversity_id=diversity_id,
        diversity_ood=diversity_ood,
        beta_ood=beta_ood,
        zero_division=zero_division,
        keepdims=keepdims,
    )
    return score
