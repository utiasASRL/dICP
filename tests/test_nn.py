"""
    Unit tests for nearest neighbor search algorithms
"""
import pytest
import torch
from dICP.kd_nn import KDTree
from dICP.diff_nn import diff_nn

@pytest.fixture
def points():
    return [(5.0,4.0,0.0), (2.0,6.0,0.0), (13.0,3.0,0.0), (8.0,7.0,0.0), (3.0,1.0,0.0)]

def test_kd_nn(points):
    # Define points
    tree = KDTree(points)

    # Test nearest neighbour search for points
    query_point = (9, 4, 0)
    expected_nearest_point1 = (8, 7, 0)
    nearest_point1 = tree.nearest_neighbour(query_point)
    assert nearest_point1 == expected_nearest_point1

    # Now, add a new point and test again
    tree.add_point((10, 2, 0))
    expected_nearest_point2 = (10, 2, 0)
    nearest_point2 = tree.nearest_neighbour(query_point)
    assert nearest_point2 == expected_nearest_point2

def test_diff_nn(points):
    # Define points
    points = torch.tensor(points, requires_grad=True, dtype=torch.float)

    # Test nearest neighbour search for points
    query_point = torch.tensor((9, 4, 0), requires_grad=True, dtype=torch.float).reshape(1,3)
    expected_nearest_point1 = torch.tensor((8, 7, 0), requires_grad=True, dtype=torch.float)

    nearest_point1 = diff_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point1 == expected_nearest_point1)

    # Assert that the nearest point is differentiable
    # THIS DOES NOT WORK YET.... I am still figuring it out...
    # query_point.grad will be None despite the fact that the ICP will be
    # differentiable wrt source pointsclouds...
    # There is this stack overflow: https://stackoverflow.com/questions/54969646/how-does-pytorch-backprop-through-argmax
    #loss1 = torch.sum(nearest_point1)
    #loss1.backward()
    #assert query_point.grad is not None and points.grad is not None

    # Now, add a new point and test again
    points = torch.cat((points, torch.tensor((10, 2, 0), requires_grad=True, dtype=torch.float).view(1, -1)))
    expected_nearest_point2 = torch.tensor((10, 2, 0), requires_grad=True, dtype=torch.float)
    nearest_point2 = diff_nn(query_point, points)

    # Assert that the nearest point is correct
    assert torch.all(nearest_point2 == expected_nearest_point2)