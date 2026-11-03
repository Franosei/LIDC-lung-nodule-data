import pandas as pd

from lidc_bias.config import LabelCaseEffectsConfig, LogisticConfig
from lidc_bias.modelling import cross_validated_predictions, make_patient_partitions
from lidc_bias.tables import ALL_MODEL_FEATURES


def _config():
    return LabelCaseEffectsConfig(
        minimum_readers=3,
        maximum_readers=4,
        cross_validation_folds=2,
        cross_validation_repetitions=2,
        bootstrap_repetitions=10,
        confidence_level=0.95,
        random_reader_seed_offset=1,
        partition_seed_offset=2,
        training_reference="median_exclude_3",
        model="combined_logistic",
        primary_references=("median_3_benign", "median_3_malignant"),
        sensitivity_references=("any_reader_4_or_5", "random_reader"),
        logistic_regression=LogisticConfig(1.0, 1000, None),
    )


def test_recorded_partitions_are_reused_and_predictions_cover_every_nodule():
    rows = []
    for index in range(20):
        label = index % 2
        row = {
            "nodule_id": f"n{index}",
            "patient_id": f"p{index}",
            "reference": float(label),
            "partition_stratum": label,
        }
        for feature_index, feature in enumerate(ALL_MODEL_FEATURES):
            row[feature] = int(label + feature_index % 2)
        rows.append(row)
    frame = pd.DataFrame(rows)
    partitions = make_patient_partitions(frame, "partition_stratum", 2, 2, seed=20)
    predictions = cross_validated_predictions(
        frame,
        "reference",
        "combined_logistic",
        _config(),
        master_seed=31,
        partitions=partitions,
    )

    assert len(predictions) == 40
    assert not predictions.duplicated(["nodule_id", "repetition"]).any()
    assert predictions["probability"].between(0, 1).all()
    assert set(predictions.groupby("repetition")["fold"].nunique()) == {2}
    expected = partitions.sort_values(["nodule_id", "repetition"])["fold"].tolist()
    observed = predictions.sort_values(["nodule_id", "repetition"])["fold"].tolist()
    assert observed == expected
