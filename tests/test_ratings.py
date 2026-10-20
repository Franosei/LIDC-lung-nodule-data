import pytest

from lidc_bias.errors import SourceDataError
from lidc_bias.ratings import decode_rating


@pytest.mark.parametrize(
    ("concept", "code", "meaning", "expected"),
    [
        ("Subtlety score", "101", "1 out of 5 (Extremely subtle)", 1),
        ("Malignancy", "905", "5 out of 5 (Highly Suspicious for Cancer)", 5),
        ("Internal structure", "C12471", "Soft tissue", 1),
        ("Internal structure", "C73434", "Air", 4),
        ("Calcification", "RID35453", "Popcorn calcification sign", 1),
        ("Calcification", "RID28473", "Absent", 6),
        ("Sphericity", "RID5811", "linear", 1),
        ("Sphericity", "RID5799", "round", 5),
        ("Margin", "RID5709", "Indistinct margin", 1),
        ("Margin", "RID5707", "Circumscribed margin", 5),
        ("Texture", "RID50153", "non-solid pulmonary nodule", 1),
        ("Texture", "RID50151", "solid pulmonary nodule", 5),
    ],
)
def test_decode_rating(concept, code, meaning, expected):
    assert decode_rating(concept, code, meaning) == expected


def test_unknown_rating_fails_closed():
    with pytest.raises(SourceDataError, match="Unsupported coded rating"):
        decode_rating("Texture", "new", "Unexpected category")
