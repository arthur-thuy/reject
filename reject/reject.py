#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Module for rejection."""
# =============================================================================

from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike, NDArray
from tabulate import tabulate

from reject.uncertainty import compute_uncertainty, compute_confidence
from reject.utils import compute_correct
from reject.constant import (
    ALL_UNC_LIST,
    METRICS_DICT,
    ENTROPY_UNC_LIST,
    GENERAL_UNC_LIST,
    UNCERTAINTIES_DICT,
)


def confusion_matrix(
    correct: ArrayLike,
    unc_ary: ArrayLike,
    threshold: float,
    relative: bool = True,
    show: bool = False,
    seed: int = 44,
) -> tuple[tuple[int, int, int, int], NDArray]:
    """Compute confusion matrix with 2 axes: (i) correct/incorrect, (ii) rejected/non-rejected.

    Parameters
    ----------
    correct : ArrayLike
        1D array of correct/incorrect indicators.
    unc_ary : ArrayLike
        1D array of uncertainty values, largest value rejected first.
    threshold : float
        Rejection threshold.
    relative : bool, optional
        Use relative rejection, otherwise absolute rejection, by default True
    show : bool, optional
        Print confusion matrix to console, by default False
    seed: int, optional
        Seed value for random rejection, by default 42

    Returns
    -------
    n_cor_rej : int
        Number of correct observations that are rejected.
    n_cor_nonrej : int
        Number of correct observations that are not rejected.
    n_incor_rej : int
        Number of incorrect observations that are rejected.
    n_incor_nonrej : int
        Number of incorrect observations that are not rejected.
    pred_reject : ndarray
        Array of True/False indicators to reject predictions.
    """
    # input checks
    if threshold < 0:
        raise ValueError("Threshold must be non-negative.")
    if relative and threshold > 1:
        raise ValueError("Threshold must be less than or equal to 1.")

    # axis 0: correct or incorrect
    idx_correct = np.where(correct == 1.0)[0]
    idx_incorrect = np.where(correct == 0.0)[0]

    # axis 1: rejected or non-rejected
    if relative:
        # relative rejection
        n_preds_rej = int(threshold * correct.size)
        # use uncertainty array
        # sort by unc_ary, then by random numbers random_draws
        # -> if values equal e.g. 1.0 -> rejected randomly
        np.random.seed(seed=seed)
        random_draws = np.random.random(correct.size)
        idx = np.lexsort((random_draws, unc_ary))
        idx = np.flip(idx, axis=0)
        idx_rej = idx[:n_preds_rej]
        idx_nonrej = idx[n_preds_rej:]
        pred_reject = np.where(np.isin(np.arange(correct.size), idx_rej), True, False)
    else:
        # absolute rejection
        pred_reject = np.where(unc_ary >= threshold, True, False)
        idx_rej = np.where(pred_reject == True)[0]
        idx_nonrej = np.where(pred_reject == False)[0]

    # intersections
    idx_cor_rej = np.intersect1d(idx_correct, idx_rej)
    idx_cor_nonrej = np.intersect1d(idx_correct, idx_nonrej)
    idx_incor_rej = np.intersect1d(idx_incorrect, idx_rej)
    idx_incor_nonrej = np.intersect1d(idx_incorrect, idx_nonrej)
    n_cor_rej = idx_cor_rej.shape[0]
    n_cor_nonrej = idx_cor_nonrej.shape[0]
    n_incor_rej = idx_incor_rej.shape[0]
    n_incor_nonrej = idx_incor_nonrej.shape[0]
    if show:
        print(  # TODO: use logging?
            tabulate(
                [
                    ["", "Non-rejected", "Rejected"],
                    ["Correct", n_cor_nonrej, n_cor_rej],
                    ["Incorrect", n_incor_nonrej, n_incor_rej],
                ],
                headers="firstrow",
            )
        )
    return (n_cor_rej, n_cor_nonrej, n_incor_rej, n_incor_nonrej), pred_reject


def compute_metrics(
    threshold: float,
    correct: ArrayLike,
    unc_ary: ArrayLike,
    relative: bool = True,
    return_bool: bool = True,
    show: bool = True,
    seed: int = 44,
) -> Union[tuple[float, float, float], tuple[tuple[float, float, float], NDArray]]:
    """Compute 3 rejection metrics using relative or absolute threshold:
    - non-rejeced accuracy (NRA)
    - classification quality (CQ)
    - rejection quality (RQ)

    Parameters
    ----------
    threshold : float
        Rejection threshold.
    correct : ArrayLike
        1D array of correct/incorrect indicator.
    unc_ary : ndarray
        1D array of uncertainty values, largest value rejected first.
    relative : bool, optional
        Use relative rejection, otherwise absolute rejection, by default True
    return_bool : bool, optional
        Return boolean array of rejected predictions, by default True
    show : bool, optional
        Print confusion matrix to console, by default True
    seed: int, optional
        Seed value for random rejection, by default 42

    Returns
    -------
    nonrej_acc : float
        Non-rejeced accuracy (NRA).
    class_quality : float
        Classification quality (CQ).
    rej_quality : float
        Rejection quality (RQ).
    pred_reject : ndarray
        Array of True/False indicators to reject predictions.
    Notes
    -----
    - rejection quality is undefined when `n_cor_rej=0`
        - if any observation is rejected: RQ = positive infinite
        - if no sample is rejected: RQ = 1
        - see: `Condessa et al. (2017) <https://doi.org/10.1016/j.patcog.2016.10.011>`_
    """
    (n_cor_rej, n_cor_nonrej, n_incor_rej, n_incor_nonrej), pred_reject = (
        confusion_matrix(
            correct=correct,
            unc_ary=unc_ary,
            threshold=threshold,
            show=show,
            relative=relative,
            seed=seed,
        )
    )

    # non-rejected accuracy
    try:
        nonrej_acc = n_cor_nonrej / (n_incor_nonrej + n_cor_nonrej)
    except ZeroDivisionError:
        nonrej_acc = np.inf  # invalid
    # classification quality
    try:
        class_quality = (n_cor_nonrej + n_incor_rej) / (
            n_cor_rej + n_cor_nonrej + n_incor_rej + n_incor_nonrej
        )
    except ZeroDivisionError:
        class_quality = np.inf  # invalid
    # rejection quality
    try:
        rej_quality = (n_incor_rej / n_cor_rej) / (
            (n_incor_rej + n_incor_nonrej) / (n_cor_rej + n_cor_nonrej)
        )
    except ZeroDivisionError:
        if (n_incor_rej + n_cor_rej) > 0:
            rej_quality = np.inf
        else:
            rej_quality = 1.0

    if show:
        data = [[nonrej_acc, class_quality, rej_quality]]
        print(  # TODO: use logging instead of print
            "\n"
            + tabulate(
                data,
                headers=[
                    "Non-rejected accuracy",
                    "Classification quality",
                    "Rejection quality",
                ],
                floatfmt=".4f",
            )
        )
    if return_bool:
        return (nonrej_acc, class_quality, rej_quality), pred_reject
    else:
        return (nonrej_acc, class_quality, rej_quality)


class ClassificationRejector:
    def __init__(
        self,
        y_true: ArrayLike,
        y_pred: ArrayLike,
        seed: int = 42,
    ):
        """Classification with rejection.

        Parameters
        ----------
        y_true : ArrayLike
            Array of true labels. Shape (n_observations,).
        y_pred : ArrayLike
            Array of predictions. Shape (n_observations, n_classes)\
                    or (n_observations, n_samples, n_classes).
        seed : int, optional
            Seed value for random rejection, by default 42
        """
        self.y_true = y_true
        self.y_pred = y_pred
        self.seed = seed
        self.num_classes = y_pred.shape[-1]
        self.max_entropy = np.log2(self.num_classes)

        # calculate uncertainty and correctness
        self._uncertainty = compute_uncertainty(y_pred)
        self.confidence = compute_confidence(y_pred)
        self.correct = compute_correct(y_true, y_pred)

    def uncertainty(
        self, unc_type: Optional[str] = None
    ) -> Union[NDArray, dict[NDArray, NDArray, NDArray]]:
        """Get uncertainty or confidence values.

        Parameters
        ----------
        unc_type : Optional[str], optional
            Uncertainty type to return. If None, dict of TU, AU, and EU is returned. By default None

        Returns
        -------
        Union[NDArray, dict[NDArray, NDArray, NDArray]]
            Array of one uncertainty type, or all three uncertainty types.

        Raises
        ------
        ValueError
            If `unc_type` is invalid.
        """
        # checks
        if unc_type is not None and unc_type not in ALL_UNC_LIST:
            raise ValueError(
                "Invalid uncertainty type. Expected one of: TU, AU, EU, confidence or None"
            )

        if unc_type == "confidence":
            return self.confidence
        elif unc_type is not None:
            return self._uncertainty[unc_type]
        else:
            return self._uncertainty

    def plot_uncertainty(
        self,
        unc_type: Optional[str] = None,
        bins: int = 15,
    ) -> plt.Figure:
        """Plot uncertainty values.

        Parameters
        ----------
        unc_type : Optional[str], optional
            Uncertainty type to return. If None, dict of TU, AU, and EU is returned. By default None

        Returns
        -------
        plt.Figure
            Figure object.

        Raises
        ------
        ValueError
            If `unc_type` is invalid.
        """
        # checks
        if unc_type is not None and unc_type not in ALL_UNC_LIST:
            raise ValueError(
                "Invalid uncertainty type. Expected one of: TU, AU, EU, confidence or None"
            )
        if unc_type == "confidence":
            xlim = (0 - 0.05, 1 + 0.05)
        else:
            max_entropy = np.log2(self.num_classes)  # TODO: use as attribute
            xlim = (0 - 0.05 * max_entropy, max_entropy + 0.05 * max_entropy)

        # draw plot
        if unc_type is not None:
            unc_enumerate = [unc_type]
            fig, axes = plt.subplots(ncols=1, figsize=(4.5, 3))
            axes = [axes]
        else:
            unc_enumerate = ENTROPY_UNC_LIST
            fig, axes = plt.subplots(ncols=3, figsize=(16, 3))

        for i, unc_type in enumerate(unc_enumerate):
            axes[i].hist(self.uncertainty(unc_type), bins=bins)
            axes[i].grid(linestyle="dashed")
            axes[i].set(xlabel=UNCERTAINTIES_DICT[unc_type], ylabel="Frequency")
            axes[i].set_xlim(xlim)
        return fig

    def reject(
        self, threshold: float, unc_type: str, relative: bool = True, show: bool = False
    ) -> tuple[tuple[float, float, float], NDArray]:
        """Reject with a single threshold.

        Parameters
        ----------
        threshold : float
            Rejection threshold.
        unc_type : str
            Uncertainty type to use for rejection order.
        relative : bool, optional
            Reject relative to the amount of observations, otherwise compare to the uncertainty value. By default True
        show : bool, optional
            Print confusion matrix and metrics, by default False

        Returns
        -------
        tuple[float, float, float]
            Non-rejected accuracy, classification quality, and rejection quality.
        """
        # checks
        if unc_type not in ALL_UNC_LIST:
            raise ValueError(
                "Invalid uncertainty type. Expected one of: TU, AU, EU, confidence"
            )

        unc_ary = (
            self.confidence if unc_type == "confidence" else self._uncertainty[unc_type]
        )
        return compute_metrics(
            threshold=threshold,
            correct=self.correct,
            unc_ary=unc_ary,
            relative=relative,
            show=show,
            seed=self.seed,
        )

    def plot_reject(
        self,
        unc_type: Optional[str] = None,
        metric: Optional[str] = None,
        relative: bool = True,
        space_start: float = 0.001,
        space_stop: float = 0.99,
        space_bins: int = 100,
        filename: Optional[str] = None,
        **save_args
    ) -> plt.Figure:
        """Plot one or multiple rejection metrics for a range of thresholds, based on one or more uncertainty types.\
        There should be at least one of `unc_type` or `metric` specified.

        Parameters
        ----------
        unc_type : Optional[str], optional
            Uncertainty type to use for rejection order. If None, 3 panels with TU, AU, and EU are plotted. By default None
        metric : Optional[str], optional
            Rejection metrics to compute. If None, 3 panels with NRA, CQ, and RQ are plotted. By default None
        relative : bool, optional
            Reject relative to the amount of observations, otherwise compare to the uncertainty value. By default True
        space_start : float, optional
            Threshold value to start figure at, by default 0.001
        space_stop : float, optional
            Threshold value to stop figure at, by default 0.99
        space_bins : int, optional
            Number of evaluation points in the line plot, by default 100
        filename : Optional[str], optional
            Filename to save figure. If None, no figure saved. By default None

        Returns
        -------
        plt.Figure
            Figure object.

        Raises
        ------
        ValueError
            If `unc_type` and `metric` are both None.
        ValueError
            If `unc_type` is invalid.
        """
        # checks
        if unc_type is None and metric is None:
            raise ValueError(
                "`unc_type` and `metric` cannot be both None, at least one must be specified."
            )
        unc_types = ["TU", "AU", "EU", "confidence"]
        if unc_type is not None and unc_type not in unc_types:
            raise ValueError(
                "Invalid uncertainty type. Expected one of: %s" % unc_types
            )

        if metric is not None:
            fig = self.__plot_1_3_uncertainty_panels(
                metric=metric,
                unc_type=unc_type,
                relative=relative,
                space_start=space_start,
                space_stop=space_stop,
                space_bins=space_bins,
                filename=filename,
                **save_args
            )
        elif metric is None:
            fig = self.__plot_3_metric_panels(
                unc_type=unc_type,
                relative=relative,
                space_start=space_start,
                space_stop=space_stop,
                space_bins=space_bins,
                filename=filename,
                **save_args
            )
        return fig

    def __plot_1_3_uncertainty_panels(
        self,
        metric: str,
        unc_type: Optional[str] = None,
        relative: bool = True,
        space_start: float = 0.001,
        space_stop: float = 0.99,
        space_bins: int = 100,
        filename: Optional[str] = None,
        **save_args
    ) -> plt.Figure:
        """Plot one or 3 panels with uncertainty types, for a specific metric.

        Parameters
        ----------
        metric : str
            Rejection metrics to compute.
        unc_type : Optional[str], optional
            Uncertainty type to use for rejection order. If None, 3 panels with TU, AU, and EU are plotted. By default None
        relative : bool, optional
            Reject relative to the amount of observations, otherwise compare to the uncertainty value. By default True
        space_start : float, optional
            Threshold value to start figure at, by default 0.001
        space_stop : float, optional
            Threshold value to stop figure at, by default 0.99
        space_bins : int, optional
            Number of evaluation points in the line plot, by default 100
        filename : Optional[str], optional
            Filename to save figure. If None, no figure saved. By default None

        Returns
        -------
        plt.Figure
            Figure object.
        """
        # draw plot
        if unc_type is not None:
            unc_enumerate = [unc_type]
            fig, axes = plt.subplots(ncols=1, figsize=(4.5, 3))
            axes = [axes]
        else:
            unc_enumerate = ENTROPY_UNC_LIST
            fig, axes = plt.subplots(ncols=3, figsize=(16, 3))

        for i, unc_type in enumerate(unc_enumerate):
            unc_ary = (
                self.confidence
                if unc_type == "confidence"
                else self._uncertainty[unc_type]
            )
            if unc_type == "confidence":
                # largest value is most uncertain
                unc_ary = 1.0 - unc_ary
            self.__plot_base_panel(
                correct=self.correct,
                unc_ary=unc_ary,
                metric=metric,
                unc_type="confidence" if unc_type == "confidence" else "entropy",
                relative=relative,
                space_start=space_start,
                space_stop=space_stop,
                space_bins=space_bins,
                ax=axes[i],
            )
            axes[i].grid(linestyle="dashed")
            if relative:
                axes[i].set(xlabel="Relative threshold", ylabel=METRICS_DICT[metric])
            else:
                axes[i].set(xlabel="Absolute threshold", ylabel=METRICS_DICT[metric])

        if filename is not None:
            fig.tight_layout()
            fig.savefig(filename, **save_args)
        return fig

    def __plot_3_metric_panels(
        self,
        unc_type: str,
        relative: bool = True,
        space_start: float = 0.001,
        space_stop: float = 0.99,
        space_bins: int = 100,
        filename: Optional[str] = None,
        **save_args
    ) -> plt.Figure:
        """Plot 3 panels with rejection metrics, for a specific uncertainty type.

        Parameters
        ----------
        unc_type : str
            Uncertainty type to use for rejection order.
        relative : bool, optional
            Reject relative to the amount of observations, otherwise compare to the uncertainty value. By default True
        space_start : float, optional
            Threshold value to start figure at, by default 0.001
        space_stop : float, optional
            Threshold value to stop figure at, by default 0.99
        space_bins : int, optional
            Number of evaluation points in the line plot, by default 100
        filename : Optional[str], optional
            Filename to save figure. If None, no figure saved. By default None

        Returns
        -------
        plt.Figure
            Figure object.
        """
        unc_ary = (
            self.confidence if unc_type == "confidence" else self._uncertainty[unc_type]
        )
        if unc_type == "confidence":
            # largest value is most uncertain
            unc_ary = 1.0 - unc_ary

        # draw plot
        fig, axes = plt.subplots(ncols=3, figsize=(16, 3))
        for i, label in enumerate(METRICS_DICT.keys()):
            self.__plot_base_panel(
                correct=self.correct,
                unc_ary=unc_ary,
                metric=label,
                unc_type="confidence" if unc_type == "confidence" else "entropy",
                relative=relative,
                space_start=space_start,
                space_stop=space_stop,
                space_bins=space_bins,
                ax=axes[i],
            )
            axes[i].grid(linestyle="dashed")
            if relative:
                axes[i].set(xlabel="Relative threshold", ylabel=METRICS_DICT[label])
            else:
                axes[i].set(xlabel="Absolute threshold", ylabel=METRICS_DICT[label])

        if filename is not None:
            fig.tight_layout()
            fig.savefig(filename, **save_args)
        return fig

    def __plot_base_panel(
        self,
        correct: ArrayLike,
        unc_ary: ArrayLike,
        metric: str,
        unc_type: str,
        relative: bool = True,
        space_start: float = 0.001,
        space_stop: float = 0.99,
        space_bins: int = 100,
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """Plot single panel with some rejection metric and uncertainty type.

        Parameters
        ----------
        correct : ArrayLike
            Array of correct predictions. Shape (n_observations,).
        unc_ary : ArrayLike
            Array of uncertainty values, largest value rejected first.
        metric : str
            Rejection metric to compute.
        unc_type : str
            Uncertainty type to use for rejection order.
        relative : bool, optional
            Reject relative to the amount of observations, otherwise compare to the uncertainty value. By default True
        space_start : float, optional
            Threshold value to start figure at, by default 0.001
        space_stop : float, optional
            Threshold value to stop figure at, by default 0.99
        space_bins : int, optional
            Number of evaluation points in the line plot, by default 100
        ax : Optional[plt.Axes], optional
            Ax to plot on. If None, new ax is created. By default None

        Returns
        -------
        plt.Axes
            Axes object.

        Raises
        ------
        ValueError
            If `unc_type` is invalid.
        """
        # checks
        if unc_type not in GENERAL_UNC_LIST:
            raise ValueError(
                "Invalid uncertainty type. Expected one of: %s" % GENERAL_UNC_LIST
            )

        if relative:
            treshold_ary = np.linspace(
                start=space_start, stop=space_stop, num=space_bins
            )
            reject_ary = treshold_ary
            plot_ary = treshold_ary
        elif not relative and unc_type == "confidence":
            treshold_ary = np.linspace(
                start=(1 - space_start),
                stop=(1 - space_stop),
                num=space_bins,
            )
            reject_ary = treshold_ary
            plot_ary = np.flip(treshold_ary, axis=0)
        elif not relative and unc_type == "entropy":
            max_entropy = np.log2(self.num_classes)
            treshold_ary = np.linspace(
                start=(1 - space_start) * max_entropy,
                stop=(1 - space_stop) * max_entropy,
                num=space_bins,
            )
            reject_ary = treshold_ary
            plot_ary = treshold_ary

        compute_metrics_rej_v = np.vectorize(
            compute_metrics,
            excluded=["correct", "unc_ary", "show", "relative", "seed"],
        )
        nonrej_acc, class_quality, rej_quality = compute_metrics_rej_v(
            reject_ary,
            correct=correct,
            unc_ary=unc_ary,
            show=False,
            return_bool=False,
            relative=relative,
            seed=self.seed,
        )

        # plot on existing axis or new axis
        if ax is None:
            ax = plt.gca()
        if metric == "NRA":
            ax.plot(plot_ary, nonrej_acc)
        elif metric == "CQ":
            ax.plot(plot_ary, class_quality)
        elif metric == "RQ":
            ax.plot(plot_ary, rej_quality)

        if not relative and unc_type == "entropy":
            # invert x-axis, largest uncertainty values on the left
            ax.invert_xaxis()
            ax.set_xlim(max_entropy + 0.05 * max_entropy, 0 - 0.05 * max_entropy)
        return ax
