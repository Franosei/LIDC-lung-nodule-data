import pytest

from lidc_bias.diagnosis import _integer_code
from lidc_bias.errors import SourceDataError


def test_whitespace_only_workbook_cell_is_missing():
    assert _integer_code(" ", {0, 1, 2, 3}, "diagnosis") is None


def test_invalid_workbook_code_fails_closed():
    with pytest.raises(SourceDataError, match="Unsupported diagnosis code"):
        _integer_code("benign", {0, 1, 2, 3}, "diagnosis")
