import pandas as pd

from lidc_bias.rq3 import assign_readers


def test_reader_assignments_are_distinct_and_reproducible():
    rows = []
    for nodule_id in ("a", "b"):
        for reader in range(4):
            rows.append(
                {
                    "nodule_id": nodule_id,
                    "patient_id": f"p-{nodule_id}",
                    "annotation_id": f"{nodule_id}-{reader}",
                    "volume_mm3": 10.0 + reader,
                    "diameter_mm": 2.0 + reader,
                    "surface_area_mm2": 5.0 + reader,
                    "compactness": 0.5,
                    "subtlety": 1,
                    "internal_structure": 1,
                    "calcification": 1,
                    "sphericity": 1,
                    "margin": 1,
                    "lobulation": 1,
                    "spiculation": 1,
                    "texture": 1,
                    "malignancy": reader + 1,
                }
            )
    annotations = pd.DataFrame(rows)
    first = assign_readers(annotations, {"a", "b"}, seed=12)
    second = assign_readers(annotations, {"a", "b"}, seed=12)
    pd.testing.assert_frame_equal(first, second)
    selected = first[
        ["reader_a_annotation_id", "reader_b_annotation_id", "reader_c_annotation_id"]
    ]
    assert selected.nunique(axis=1).eq(3).all()
