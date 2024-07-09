"""
    Unit tests for nearest neighbor search algorithms
"""
import pytest
import torch
from dICP.nn import nn

@pytest.fixture
def points():
    return [(5.0,4.0,0.0), (2.0,6.0,0.0), (13.0,3.0,0.0), (8.0,7.0,0.0), (3.0,1.0,0.0)]

def test_diff_nn(points):
    # Create nearest neighbour searcher
    diff_nn = nn(differentiable=True)

    # Define points
    points = torch.tensor(points, requires_grad=True, dtype=torch.float)

    # Test nearest neighbour search for points
    query_point = torch.tensor([[9, 4, 0]], requires_grad=True, dtype=torch.float)
    expected_nearest_point1 = torch.tensor((8, 7, 0), requires_grad=True, dtype=torch.float)

    nearest_point1 = diff_nn.find_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point1 == expected_nearest_point1)

    #Confirm gradient exists
    nearest_point1.sum().backward()
    assert query_point.grad is not None and points.grad is not None

    # Confirm gradient is not nan
    assert torch.isnan(query_point.grad).any() == False and torch.isnan(points.grad).any() == False

    # Now, add a new point and test again
    points = torch.cat((points, torch.tensor((10, 2, 0), requires_grad=True, dtype=torch.float).view(1, -1)))
    expected_nearest_point2 = torch.tensor((10, 2, 0), requires_grad=True, dtype=torch.float)
    nearest_point2 = diff_nn.find_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point2 == expected_nearest_point2)
