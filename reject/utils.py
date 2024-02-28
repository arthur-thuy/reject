#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Module for utils."""
# =============================================================================

from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_correct(y_true: ArrayLike, y_pred: ArrayLike) -> NDArray:
    """Compute correct predictions.

    Parameters
    ----------
    y_true : ArrayLike
        Array of true labels. Shape (n_observations,).
    y_pred : ArrayLike
        Array of predictions. Shape (n_observations, n_classes)\
              or (n_observations, n_samples, n_classes).

    Returns
    -------
    NDArray
        Array of correct predictions. Shape (n_observations,).

    Raises
    ------
    ValueError
        If shape of `y_pred` or `y_true` is invalid.
    """
    # checks
    if not y_pred.ndim in [1, 2, 3]:
        raise ValueError(
            f"`y_pred` should have rank 1, 2, or 3, has rank {y_pred.ndim}"
        )
    if not y_true.ndim == 1:
        raise ValueError(f"`y_true` should have rank 1, has rank {y_true.ndim}")
    if y_pred.ndim == 1:
        y_label = y_pred
    else:
        _, _, y_label = aggregate_preds(y_pred)
    is_correct = np.equal(y_true, y_label)
    return is_correct


def aggregate_preds(y_pred: ArrayLike) -> tuple[NDArray, NDArray, NDArray]:
    """Aggregate predictions to get stack, mean, and label.
    Parameters
    ----------
    y_pred : ArrayLike
        Array of predictions. Shape (n_observations, n_classes)\
              or (n_observations, n_samples, n_classes).
    Returns
    -------
    tuple[NDArray, NDArray, NDArray]
        Stack (rank 2 or 3), mean (rank 2), and label (rank 1) of predictions.
    """
    # checks
    if not y_pred.ndim in [2, 3]:
        raise ValueError(f"`y_pred` should have rank 2 or 3, has rank {y_pred.ndim}")
    # only take mean if multiple samples
    if y_pred.ndim == 3:
        y_stack = y_pred
        y_mean = np.mean(y_pred, axis=-2)
    elif y_pred.ndim == 2:
        y_stack = np.expand_dims(y_pred, axis=-2)
        y_mean = y_pred
    y_label = np.argmax(y_mean, axis=-1)
    return y_stack, y_mean, y_label
