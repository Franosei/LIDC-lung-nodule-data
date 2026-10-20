import numpy as np
import pandas as pd

from lidc_bias.rq1 import disagreement_metrics


def test_disagreement_metrics_use_all_within_nodule_pairs():
    rows = []
    ratings = {"n1": [1, 1, 1], "n2": [1, 3, 5], "n3": [4, 4, 4]}
    for patient_index, (nodule_id, values) in enumerate(ratings.items()):
        for reader_index, value in enumerate(values, start=1):
            rows.append(
                {
                    "patient_id": f"p{patient_index}",
                    "nodule_id": nodule_id,
                    "reader_index": reader_index,
                    "malignancy": value,
                }
            )
    metrics = disagreement_metrics(pd.DataFrame(rows))

    assert np.isclose(metrics["pairwise_exact_agreement"], 6 / 9)
    assert np.isclose(metrics["pairwise_binary_agreement"], 6 / 7)
    assert np.isclose(metrics["all_readers_agree"], 2 / 3)
    assert np.isclose(metrics["benign_malignant_discordance"], 1 / 3)
    assert np.isfinite(metrics["ordinal_alpha"])
    assert np.isfinite(metrics["binary_alpha"])
