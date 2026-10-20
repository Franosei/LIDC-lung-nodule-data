import numpy as np

from lidc_bias.rq6 import fit_latent_classes


def test_latent_class_fit_orders_classes_by_expected_rating():
    counts = np.array(
        [
            [3, 1, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [2, 2, 0, 0, 0],
            [0, 0, 0, 1, 3],
            [0, 0, 0, 0, 4],
            [0, 0, 1, 1, 2],
        ],
        dtype=float,
    )
    fit = fit_latent_classes(
        counts,
        classes=2,
        starts=10,
        seed=17,
        maximum_iterations=1000,
        tolerance=1e-10,
    )
    expected_ratings = fit.rating_probabilities @ np.arange(1, 6)
    assert fit.posterior.shape == (6, 2)
    assert np.allclose(fit.posterior.sum(axis=1), 1.0)
    assert expected_ratings[0] < expected_ratings[1]
    assert np.isclose(fit.mixing.sum(), 1.0)
