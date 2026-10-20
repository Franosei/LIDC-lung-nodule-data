import numpy as np

from lidc_bias.metrics import _calibration_parameters, classification_metrics


def test_classification_metrics_for_perfect_predictions():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["auc"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0
    assert metrics["prevalence"] == 0.5
    assert 0.0 <= metrics["brier_score"] < 0.1


def test_calibration_fit_is_finite_when_classes_overlap():
    intercept, slope = _calibration_parameters(
        np.array([0, 1, 0, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert np.isfinite(intercept)
    assert np.isfinite(slope)


def test_calibration_fit_fails_safely_under_complete_separation():
    intercept, slope = _calibration_parameters(
        np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])
    )
    assert np.isnan(intercept)
    assert np.isnan(slope)
