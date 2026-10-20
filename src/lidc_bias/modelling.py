"""Leakage-resistant model construction and grouped repeated cross-validation."""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import Rq2Config
from .errors import AnalysisError
from .tables import GEOMETRY_FEATURES, SEMANTIC_FEATURES

MODEL_FEATURES = {
    "geometry_logistic": GEOMETRY_FEATURES,
    "semantic_logistic": SEMANTIC_FEATURES,
    "combined_logistic": GEOMETRY_FEATURES + SEMANTIC_FEATURES,
    "combined_random_forest": GEOMETRY_FEATURES + SEMANTIC_FEATURES,
}


def _class_weight(value: str | None) -> str | None:
    return None if value in {None, "none", "None"} else value


def build_model(model_name: str, config: Rq2Config, seed: int) -> Pipeline:
    """Construct one fully encapsulated preprocessing and classification pipeline."""
    if model_name not in MODEL_FEATURES:
        raise AnalysisError(f"Unsupported model: {model_name}")
    features = MODEL_FEATURES[model_name]
    if model_name.endswith("logistic"):
        preprocessing = ColumnTransformer(
            [
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median")),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    features,
                )
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        estimator = LogisticRegression(
            C=config.logistic_regression.regularisation_c,
            max_iter=config.logistic_regression.maximum_iterations,
            class_weight=_class_weight(config.logistic_regression.class_weight),
            solver="lbfgs",
            random_state=seed,
        )
    else:
        preprocessing = ColumnTransformer(
            [("numeric", SimpleImputer(strategy="median"), features)],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        estimator = RandomForestClassifier(
            n_estimators=config.random_forest.trees,
            min_samples_leaf=config.random_forest.minimum_leaf_size,
            max_features=config.random_forest.maximum_features,
            class_weight=_class_weight(config.random_forest.class_weight),
            n_jobs=config.random_forest.n_jobs,
            random_state=seed,
        )
    return Pipeline([("preprocess", preprocessing), ("classifier", estimator)])


def cross_validated_predictions(
    frame: pd.DataFrame,
    label_column: str,
    model_name: str,
    config: Rq2Config,
    master_seed: int,
) -> pd.DataFrame:
    """Predict every nodule out of sample in each grouped CV repetition."""
    labels = frame[label_column].astype(float)
    strata = labels.fillna(2.0).astype(int)
    groups = frame["patient_id"].astype(str)
    records: list[pd.DataFrame] = []

    for repetition in range(config.cross_validation_repetitions):
        repeat_seed = master_seed + repetition
        splitter = StratifiedGroupKFold(
            n_splits=config.cross_validation_folds,
            shuffle=True,
            random_state=repeat_seed,
        )
        fold_count = 0
        for fold, (train_indices, test_indices) in enumerate(
            splitter.split(frame, strata, groups), start=1
        ):
            fold_count += 1
            train = frame.iloc[train_indices]
            test = frame.iloc[test_indices]
            train_groups = set(train["patient_id"].astype(str))
            test_groups = set(test["patient_id"].astype(str))
            if train_groups & test_groups:
                raise AnalysisError("Patient leakage detected between training and test folds")

            eligible_train = train[label_column].notna()
            training_labels = train.loc[eligible_train, label_column].astype(int)
            if training_labels.nunique() != 2:
                raise AnalysisError(
                    f"Training fold lacks both classes: reference={label_column}, "
                    f"repetition={repetition}, fold={fold}"
                )
            model_seed = master_seed + repetition * 100 + fold
            model = build_model(model_name, config, model_seed)
            model.fit(train.loc[eligible_train], training_labels)
            probabilities = model.predict_proba(test)[:, 1]
            records.append(
                pd.DataFrame(
                    {
                        "nodule_id": test["nodule_id"].to_numpy(),
                        "patient_id": test["patient_id"].to_numpy(),
                        "reference": label_column,
                        "model": model_name,
                        "repetition": repetition,
                        "fold": fold,
                        "label": test[label_column].to_numpy(),
                        "probability": probabilities,
                    }
                )
            )
        if fold_count != config.cross_validation_folds:
            raise AnalysisError("Cross-validation did not produce the configured number of folds")

    predictions = pd.concat(records, ignore_index=True)
    expected = len(frame) * config.cross_validation_repetitions
    if len(predictions) != expected:
        raise AnalysisError(f"Expected {expected} predictions, produced {len(predictions)}")
    duplicates = predictions.duplicated(["nodule_id", "repetition"]).any()
    if duplicates:
        raise AnalysisError("A nodule received more than one prediction in a CV repetition")
    return predictions
