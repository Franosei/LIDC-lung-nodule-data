import numpy as np
import pandas as pd

from lidc_bias.tables import build_nodule_table


def _annotation(nodule_id, patient_id, rating, volume, surface, reader):
    return {
        "nodule_id": nodule_id,
        "patient_id": patient_id,
        "study_instance_uid": f"study-{patient_id}",
        "source_series_instance_uid": f"series-{patient_id}",
        "tracking_identifier": nodule_id,
        "volume_mm3": volume,
        "diameter_mm": 10.0 + reader,
        "surface_area_mm2": surface,
        "compactness": 0.7 + reader / 100,
        "subtlety": 3,
        "internal_structure": 1,
        "calcification": 6,
        "sphericity": 4,
        "margin": 3,
        "lobulation": 2,
        "spiculation": 2,
        "texture": 5,
        "malignancy": rating,
    }


def test_build_nodule_table_averages_reader_features_and_preserves_disagreement():
    annotations = pd.DataFrame(
        [
            _annotation("n1", "p1", 1, 100.0, 50.0, 1),
            _annotation("n1", "p1", 5, 400.0, 80.0, 2),
            _annotation("n2", "p2", 3, 200.0, 60.0, 1),
            _annotation("n2", "p2", 3, 200.0, 60.0, 2),
        ]
    )
    nodules = build_nodule_table(annotations).set_index("nodule_id")

    assert nodules.loc["n1", "reader_count"] == 2
    assert nodules.loc["n1", "diameter_mm"] == 11.5
    assert nodules.loc["n1", "malignancy_median"] == 3.0
    assert bool(nodules.loc["n1", "benign_malignant_discordance"])
    assert not bool(nodules.loc["n1", "all_readers_agree"])
    assert np.isclose(nodules.loc["n1", "log_volume"], np.mean(np.log([100.0, 400.0])))
    assert bool(nodules.loc["n2", "all_readers_agree"])
