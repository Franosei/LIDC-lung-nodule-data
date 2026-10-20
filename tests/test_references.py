import pandas as pd

from lidc_bias.references import build_reference_labels


def test_reference_definitions_handle_integer_and_half_integer_medians():
    nodules = pd.DataFrame(
        {
            "nodule_id": ["a", "b", "c", "d"],
            "malignancy_median": [2.5, 3.0, 3.5, 4.0],
        }
    )
    annotations = pd.DataFrame(
        {
            "nodule_id": ["a", "a", "b", "b", "c", "c", "d", "d"],
            "annotation_id": [f"x{i}" for i in range(8)],
            "malignancy": [2, 3, 2, 4, 3, 4, 4, 5],
        }
    )
    labels = build_reference_labels(nodules, annotations, seed=17).set_index("nodule_id")

    assert labels.loc["a", "median_exclude_3"] == 0
    assert pd.isna(labels.loc["b", "median_exclude_3"])
    assert labels.loc["c", "median_exclude_3"] == 1
    assert labels.loc["b", "median_3_benign"] == 0
    assert labels.loc["b", "median_3_malignant"] == 1
    assert labels.loc["b", "any_reader_4_or_5"] == 1


def test_random_reader_reference_is_reproducible():
    nodules = pd.DataFrame({"nodule_id": ["a"], "malignancy_median": [3.0]})
    annotations = pd.DataFrame(
        {"nodule_id": ["a", "a"], "annotation_id": ["x1", "x2"], "malignancy": [1, 5]}
    )
    first = build_reference_labels(nodules, annotations, seed=9)
    second = build_reference_labels(nodules, annotations, seed=9)
    pd.testing.assert_frame_equal(first, second)
