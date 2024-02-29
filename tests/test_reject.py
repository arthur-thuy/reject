#!/usr/bin/env python3
# =============================================================================
# Created By  : Arthur Thuy
# Created Date: Wed February 28 2024
# =============================================================================
"""Testing script for rejection."""
# =============================================================================
# NOTE: run with `poetry run pytest tests/`

import pytest
import numpy as np
import matplotlib.pyplot as plt
from reject.reject import ClassificationRejector, confusion_matrix, compute_metrics


@pytest.fixture
def y_true_label():
    return np.array([0.0, 1.0, 1.0, 0.0, 1.0])


@pytest.fixture
def y_pred_label():
    return np.array([0.0, 0.0, 1.0, 0.0, 0.0])


@pytest.fixture
def correct():
    return np.array([1.0, 0.0, 1.0, 1.0, 0.0])


@pytest.fixture
def y_pred_stack():
    return np.array([[0.8, 0.2], [0.6, 0.4], [0.3, 0.7], [0.3, 0.7], [0.1, 0.9]])


@pytest.fixture
def unc_ary():
    return np.array([0.2, 0.8, 0.4, 0.6, 0.5])


class TestConfusionMatrix:
    @pytest.mark.parametrize(
        "relative, threshold, matrix_rej",
        [
            (False, 0.45, (1, 2, 2, 0)),
            (False, 0.1, (3, 0, 2, 0)),
            (False, 0.9, (0, 3, 0, 2)),
            (True, 0.45, (1, 2, 1, 1)),
            (True, 0.1, (0, 3, 0, 2)),
            (True, 0.9, (2, 1, 2, 0)),
        ],
    )
    def test_unit(self, correct, unc_ary, threshold, relative, matrix_rej):
        (
            actual_n_cor_rej,
            actual_n_cor_nonrej,
            actual_n_incor_rej,
            actual_n_incor_nonrej,
        ), _ = confusion_matrix(correct, unc_ary, threshold, relative=relative)
        (
            expected_n_cor_rej,
            expected_n_cor_nonrej,
            expected_n_incor_rej,
            expected_n_incor_nonrej,
        ) = matrix_rej
        assert (
            actual_n_cor_rej == expected_n_cor_rej
        ), f"`n_cor_rej` should be {expected_n_cor_rej}, is {actual_n_cor_rej}."
        assert (
            actual_n_cor_nonrej == expected_n_cor_nonrej
        ), f"`n_cor_nonrej` should be {expected_n_cor_nonrej}, is {actual_n_cor_nonrej}."
        assert (
            actual_n_incor_rej == expected_n_incor_rej
        ), f"`n_incor_rej` should be {expected_n_incor_rej}, is {actual_n_incor_rej}."
        assert (
            actual_n_incor_nonrej == expected_n_incor_nonrej
        ), f"`n_incor_nonrej` should be {expected_n_incor_nonrej}, is {actual_n_incor_nonrej}."

    def test_error(self, correct, unc_ary):
        with pytest.raises(ValueError):
            confusion_matrix(correct, unc_ary, 1.5, relative=True)
        with pytest.raises(ValueError):
            confusion_matrix(correct, unc_ary, -0.5, relative=False)


class TestComputeMetricsRej:
    @pytest.mark.parametrize(
        "relative, use_idx, threshold, metrics_rej",
        [
            (False, False, 0.45, (1.0, 0.8, 3.0)),
            (False, False, 0.1, (np.inf, 0.4, 1.0)),
            (False, False, 0.9, (0.6, 0.6, 1.0)),
            (True, False, 0.45, (2 / 3, 0.6, 1.5)),
            (True, False, 0.1, (0.6, 0.6, 1.0)),
            (True, False, 0.9, (1.0, 0.6, 1.5)),
        ],
    )
    def test_unit(self, correct, unc_ary, threshold, relative, metrics_rej, use_idx):
        actual_nonrej_acc, actual_class_quality, actual_rej_quality = compute_metrics(
            threshold, correct, unc_ary, relative=relative, return_bool=False
        )
        expected_nonrej_acc, expected_class_quality, expected_rej_quality = metrics_rej
        np.testing.assert_allclose(
            actual_nonrej_acc,
            expected_nonrej_acc,
            err_msg=f"`nonrej_acc` should be {expected_nonrej_acc}, is {actual_nonrej_acc}.",
        )
        np.testing.assert_allclose(
            actual_class_quality,
            expected_class_quality,
            err_msg=f"`class_quality` should be {expected_class_quality}, is {actual_class_quality}.",
        )
        np.testing.assert_allclose(
            actual_rej_quality,
            expected_rej_quality,
            err_msg=f"`rej_quality` should be {expected_rej_quality}, is {actual_rej_quality}.",
        )

    def test_unit_edge_nra(self):
        correct = np.array([1.0, 1.0, 1.0, 0.0])
        unc_ary = np.array([0.2, 0.15, 0.3, 0.7])
        threshold = 0.1
        nra, _, _ = compute_metrics(
            threshold, correct, unc_ary, relative=False, return_bool=False
        )
        assert (
            nra == np.inf
        ), f"NRA should be inf if all observations are rejected, is {nra}."

    def test_unit_edge_cq(self):
        correct = np.array([])
        unc_ary = np.array([])
        threshold = 0.1
        _, cq, _ = compute_metrics(
            threshold, correct, unc_ary, relative=False, return_bool=False
        )
        assert cq == np.inf, f"CQ should be inf if there are no observations, is {cq}."

    def test_unit_edge_rq1(self):
        correct = np.array([1.0, 1.0, 1.0, 1.0])
        unc_ary = np.array([0.2, 0.15, 0.3, 0.7])
        threshold = 0.1
        _, _, rq = compute_metrics(
            threshold, correct, unc_ary, relative=False, return_bool=False
        )
        assert (
            rq == np.inf
        ), f"RQ should be inf if `n_cor_rej` = 0 and any sample is rejected, is {rq}."

    def test_unit_edge_rq2(self):
        correct = np.array([1.0, 1.0, 1.0, 1.0])
        unc_ary = np.array([0.2, 0.15, 0.3, 0.7])
        threshold = 0.9
        _, _, rq = compute_metrics(
            threshold, correct, unc_ary, relative=False, return_bool=False
        )
        assert (
            rq == 1.0
        ), f"RQ should be 1.0 if `n_cor_rej` = 0 and no samples are rejected, is {rq}."


class TestClassificationRejector:
    def test_unit(self):
        pass

    @pytest.mark.parametrize(
        "y_true, y_pred, num_classes",
        [
            (np.zeros((5,)), np.zeros((5, 128, 10)), 10),
            (np.zeros((5,)), np.zeros((5, 10)), 10),
            (np.zeros((12,)), np.zeros((12, 128, 5)), 5),
        ],
    )
    def test_constructor(self, y_true, y_pred, num_classes):
        rej = ClassificationRejector(y_true, y_pred)
        assert (
            rej.num_classes == num_classes
        ), f"Number of classes should be {num_classes}, is {rej.num_classes}."
        assert rej.max_entropy == np.log2(
            rej.num_classes
        ), f"Max entropy should be {np.log2(num_classes) = :.2f}, is {rej.max_entropy:.2f}."

    @pytest.mark.parametrize(
        "unc_type, return_type",
        [
            ("TU", np.ndarray),
            ("AU", np.ndarray),
            ("EU", np.ndarray),
            ("confidence", np.ndarray),
            (None, dict),
        ],
    )
    def test_uncertainty(self, unc_type, return_type):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        assert isinstance(rej.uncertainty(unc_type), return_type)

    def test_uncertainty_error(self):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        with pytest.raises(ValueError):
            rej.uncertainty("wrong")

    @pytest.mark.parametrize(
        "unc_type",
        ["TU", "AU", "EU", "confidence"],
    )
    def test_plot_uncertainty(self, unc_type, y_true_label, y_pred_stack):
        rej = ClassificationRejector(y_true=y_true_label, y_pred=y_pred_stack)
        fig = rej.plot_uncertainty(unc_type)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_uncertainty_error(self):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        with pytest.raises(ValueError):
            rej.plot_uncertainty("wrong")

    def test_reject(self):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        output = rej.reject(0.5, "TU")
        assert isinstance(output, tuple)
        assert isinstance(output[0], tuple)
        assert isinstance(output[1], np.ndarray)

    def test_reject_error(self):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        with pytest.raises(ValueError):
            rej.reject(0.5, "wrong")

    @pytest.mark.parametrize(
        "unc_type, metric, relative",
        [
            ("TU", "NRA", False),
            ("TU", "CQ", False),
            ("TU", "RQ", False),
            ("AU", "NRA", False),
            ("AU", "CQ", False),
            ("AU", "RQ", False),
            ("EU", "NRA", False),
            ("EU", "CQ", False),
            ("EU", "RQ", False),
            ("confidence", "NRA", False),
            ("confidence", "CQ", False),
            ("confidence", "RQ", False),
            ("TU", "NRA", True),
            ("TU", "CQ", True),
            ("TU", "RQ", True),
            ("AU", "NRA", True),
            ("AU", "CQ", True),
            ("AU", "RQ", True),
            ("EU", "NRA", True),
            ("EU", "CQ", True),
            ("EU", "RQ", True),
            ("confidence", "NRA", True),
            ("confidence", "CQ", True),
            ("confidence", "RQ", True),
        ],
    )
    def test_plot_reject(self, y_true_label, y_pred_stack, unc_type, metric, relative):
        rej = ClassificationRejector(y_true=y_true_label, y_pred=y_pred_stack)
        fig = rej.plot_reject(unc_type=unc_type, metric=metric, relative=relative)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_plot_reject_error(self):
        rej = ClassificationRejector(
            y_true=np.zeros((5,)), y_pred=np.zeros((5, 128, 10))
        )
        with pytest.raises(ValueError):
            rej.plot_reject("wrong")
        with pytest.raises(ValueError):
            rej.plot_reject(None, None)
