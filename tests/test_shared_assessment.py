import pandas as pd

from lidc_bias.shared_assessment import assign_annotation_roles


def test_annotation_role_assignments_are_distinct_and_reproducible():
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
    nodules = pd.DataFrame(
        {
            "nodule_id": ["a", "b"],
            "log_volume": [1.0, 1.0],
            "diameter_mm": [3.5, 3.5],
            "log_surface_area": [1.0, 1.0],
            "compactness": [0.5, 0.5],
        }
    )
    first = assign_annotation_roles(annotations, nodules, seed=12, assignment_repetition=0)
    second = assign_annotation_roles(annotations, nodules, seed=12, assignment_repetition=0)
    pd.testing.assert_frame_equal(first, second)
    selected = first[["annotation_a_id", "annotation_b_id", "annotation_c_id"]]
    assert selected.nunique(axis=1).eq(3).all()
    assert first["assignment_seed"].eq(12).all()
