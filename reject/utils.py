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
from scipy.special import softmax


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
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Number of observations in `y_true` and `y_pred` should match,\
                got {y_true.shape[0]} and {y_pred.shape[0]}"
        )
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


def generate_synthetic_output(
    num_samples: int, num_observations: int
) -> tuple[NDArray, NDArray]:
    """Generate synthetic NN output for showcasing functions.

    Parameters
    ----------
    num_samples : int
        Number of samples to draw per observation.
    num_observations : int
        Number of observations.

    Returns
    -------
    tuple[NDArray, NDArray]
        Tuple of synthetic predictions and true labels.
    """
    NUM_CLASSES = 10
    # example logit output
    logit_ary = [0.01, 0.01, 0.01, 0.4, 0.01, 0.01, 0.03, 0.01, 0.40, 0.11]
    assert np.isclose(np.sum(logit_ary), 1.0)

    # OOD
    y_pred_ood = np.empty((num_observations, num_samples, NUM_CLASSES))
    for i in range(num_observations):
        for j in range(num_samples):
            roll_idx = np.random.choice(
                10, 1, p=[0.11, 0.01, 0.01, 0.27, 0.01, 0.11, 0.02, 0.01, 0.20, 0.25]
            )
            y_pred_ood[i, j] = np.random.multinomial(
                10, np.roll(logit_ary, roll_idx), size=1
            )
    y_pred_ood = softmax(y_pred_ood, axis=-1)
    assert y_pred_ood.shape == (num_observations, num_samples, NUM_CLASSES)

    # ID
    id_ary = [0.01, 0.01, 0.01, 0.27, 0.01, 0.01, 0.02, 0.01, 0.40, 0.25]
    assert np.isclose(np.sum(id_ary), 1.0)
    y_pred_id = np.random.multinomial(10, id_ary, size=(num_observations, num_samples))
    y_pred_id = softmax(y_pred_id, axis=-1)
    assert y_pred_id.shape == (num_observations, num_samples, NUM_CLASSES)

    # concatenate preds
    y_pred_all = np.concatenate((y_pred_ood, y_pred_id), axis=0)
    assert y_pred_all.shape == (2 * num_observations, num_samples, NUM_CLASSES)

    # true labels
    y_true_id = np.full((num_observations), 8)
    y_true_ood = np.full((num_observations), 999)
    y_true_all = np.concatenate((y_true_ood, y_true_id), axis=0)
    assert y_true_all.shape == (2 * num_observations,)

    return y_pred_all, y_true_all
