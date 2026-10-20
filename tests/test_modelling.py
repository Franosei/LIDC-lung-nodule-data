import pandas as pd

from lidc_bias.config import LogisticConfig, RandomForestConfig, Rq2Config
from lidc_bias.modelling import cross_validated_predictions
from lidc_bias.tables import ALL_MODEL_FEATURES


def _config():
    return Rq2Config(
        minimum_readers=3,
        maximum_readers=4,
        cross_validation_folds=2,
        cross_validation_repetitions=2,
        bootstrap_repetitions=10,
        confidence_level=0.95,
        random_reader_seed_offset=1,
        fixed_prediction_reference="median_exclude_3",
        fixed_prediction_model="combined_logistic",
        excluded_case_reference="any_reader_4_or_5",
        logistic_regression=LogisticConfig(1.0, 1000, None),
        random_forest=RandomForestConfig(10, 1, "sqrt", None, 1),
    )


def test_cross_validated_predictions_cover_each_nodule_once_per_repetition():
    rows = []
    for index in range(20):
        label = index % 2
        row = {
            "nodule_id": f"n{index}",
            "patient_id": f"p{index}",
            "reference": float(label),
        }
        for feature_index, feature in enumerate(ALL_MODEL_FEATURES):
            row[feature] = label + feature_index / 100 + index / 1000
        rows.append(row)
    predictions = cross_validated_predictions(
        pd.DataFrame(rows), "reference", "combined_logistic", _config(), master_seed=31
    )

    assert len(predictions) == 40
    assert not predictions.duplicated(["nodule_id", "repetition"]).any()
    assert predictions["probability"].between(0, 1).all()
    assert set(predictions.groupby("repetition")["fold"].nunique()) == {2}
