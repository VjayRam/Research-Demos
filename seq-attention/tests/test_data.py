import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))

from data import _parse_isolet_file  # noqa: E402


def test_parse_isolet_file_splits_features_and_zero_indexes_labels():
    text = "1.0,2.0,3.0,1.\n4.0,5.0,6.0,26.\n"
    X, y = _parse_isolet_file(text)
    assert X.shape == (2, 3)
    assert X[0].tolist() == [1.0, 2.0, 3.0]
    assert y.tolist() == [0, 25]


def test_parse_isolet_file_skips_blank_lines():
    text = "1.0,2.0,1.\n\n3.0,4.0,2.\n"
    X, y = _parse_isolet_file(text)
    assert X.shape == (2, 2)
    assert y.tolist() == [0, 1]
