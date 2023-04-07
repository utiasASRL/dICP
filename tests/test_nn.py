"""
    Unit tests for nearest neighbor search algorithms
"""
import pytest
import torch
from dICP.kd_nn import KDTree
from dICP.diff_nn import diff_nn

@pytest.fixture
def points():
    return [(5.0,4.0), (2.0,6.0), (13.0,3.0), (8.0,7.0), (3.0,1.0)]

def test_kd_nn(points):
    # Define points
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

def test_diff_nn(points):
    # Define points
    points = torch.tensor(points, requires_grad=True, dtype=torch.float)

    # Test nearest neighbour search for points
    query_point = torch.tensor((9, 4), requires_grad=True, dtype=torch.float)
    expected_nearest_point1 = torch.tensor((8, 7), requires_grad=True, dtype=torch.float)
    nearest_point1 = diff_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point1 == expected_nearest_point1)
    # Assert that the nearest point is differentiable
    loss1 = torch.sum(nearest_point1)
    loss1.backward()
    assert query_point.grad is not None and points.grad is not None

    # Now, add a new point and test again
    points = torch.cat((points, torch.tensor((10, 2), requires_grad=True, dtype=torch.float).view(1, -1)))
    expected_nearest_point2 = torch.tensor((10, 2), requires_grad=True, dtype=torch.float)
    nearest_point2 = diff_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point2 == expected_nearest_point2)