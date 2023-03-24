"""
    Unit tests for kd_nn.py
"""

import pytest
from diff_ICP.kd_nn import KDTree

@pytest.fixture
def points():
    return [(5,4), (2,6), (13,3), (8,7), (3,1)]

def test_nearest_neighbour(points):
    tree = KDTree(points)

    # Test nearest neighbour search for points
    query_point = (9, 4)
    expected_nearest_point1 = (8, 7)
    nearest_point1 = tree.nearest_neighbour(query_point)
    assert nearest_point1 == expected_nearest_point1

    # Now, add a new point and test again
    tree.add_point((10, 2))
    expected_nearest_point2 = (10, 2)
    nearest_point2 = tree.nearest_neighbour(query_point)
    assert nearest_point2 == expected_nearest_point2