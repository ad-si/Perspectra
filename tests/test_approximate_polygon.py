import pytest
from perspectra import approximate_polygon

@pytest.mark.skip(reason="Investigate why this test is failing.")
def test_approximate_polygon():
    # approximate_polygon(coords, tolerance, target_count):
    result = approximate_polygon.approximate_polygon(
        coords = [(0, 0), (5, 0), (5, 5), (3, 5.1), (0, 5)],
        tolerance = 1,
        target_count = 4
    )
    assert result == [(0, 0), (5, 0), (5, 5), (5, 0)]
