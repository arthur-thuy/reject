#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Module for uncertainty."""
# =============================================================================

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import entropy

from reject.utils import aggregate_preds
from reject.constant import ENTROPY_UNC_LIST


def compute_uncertainty(
    y_pred: ArrayLike, unc_type: Optional[str] = None
) -> Union[NDArray, tuple[NDArray, NDArray, NDArray]]:
    """Calculate total uncertainty (TU), aleatoric uncertainty (AU) and epistemic uncertainty (EU).
    Parameters
    ----------
    y_pred : ArrayLike
        Array of predictions. Shape (n_observations, n_classes)\
              or (n_observations, n_samples, n_classes).
    unc_type : Optional[str], optional
        Type of uncertainty to compute (either TU, AU, or EU), by default None
    Returns
    -------
    Union[NDArray, tuple[NDArray, NDArray, NDArray]]
        Array of one uncertainty type, or all three uncertainty types.
    Raises
    ------
    ValueError
        If unc_type is invalid.
    """
    # checks
    if (unc_type is not None) and (unc_type not in ENTROPY_UNC_LIST):
        raise ValueError("`type` must be `None` or one of TU, AU, EU.")
    if not y_pred.ndim in [2, 3]:
        raise ValueError(f"`y_stack` should have rank 2 or 3, has rank {y_pred.ndim}")

    # get rank 3 stack
    y_stack, _, _ = aggregate_preds(y_pred)

    # total: (observations, samples, classes) => (observations, classes) => (observations,)
    unc_total = entropy(np.mean(y_stack, axis=-2), base=2, axis=-1)
    # aleatoric: (observations, samples, classes) => (observations, samples) => (observations,)
    unc_aleatoric = np.mean(entropy(y_stack, base=2, axis=-1), axis=-1)
    # epistemic: (observations,)
    unc_epistemic = np.subtract(unc_total, unc_aleatoric)
    unc_all = {"TU": unc_total, "AU": unc_aleatoric, "EU": unc_epistemic}
    if unc_type is not None:
        return unc_all[unc_type]
    else:
        return unc_all


def compute_confidence(y_pred) -> NDArray:
    """Compute confidence.

    Parameters
    ----------
    y_pred : ArrayLike
        Array of predictions. Shape (n_observations, n_classes)\
              or (n_observations, n_samples, n_classes).

    Returns
    -------
    conf : NDArray
        Array of confidence values.
    """
    # checks
    if not y_pred.ndim in [2, 3]:
        raise ValueError(f"`y_stack` should have rank 2 or 3, has rank {y_pred.ndim}")
    _, y_mean, _ = aggregate_preds(y_pred)
    conf = np.max(y_mean, axis=-1)
    return conf
