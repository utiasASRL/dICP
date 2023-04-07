"""
    Unit tests for ptp_ICP.py
"""
import pytest
import numpy as np
import torch
torch.set_printoptions(precision=8)
from dICP.ICP import ptp_ICP, diff_ptp_ICP, plot_overlay
from dICP.kd_nn import KDTree
from dICP import se3op
import matplotlib.pyplot as plt

@pytest.fixture
def max_iterations():
    return 50

@pytest.fixture
def tolerance():
    return 1e-12

@pytest.fixture
def source():
    # Return only the first 10 points
    return np.load('points_scan.npy')

@pytest.fixture
def target():
    return np.load('points_map.npy')

def test_ptp_ICP(source, target, max_iterations, tolerance):
    """
    Test 2D point-to-point ICP algorithm.
    """

    # Load test scan and map
    source_tree = KDTree(source.tolist())
    target_tree = KDTree(target.tolist())
    
    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    C_true = np.array([[0.9950, -0.0998, 0.0], [0.0998, 0.9950, 0.0], [0.0, 0.0, 1.0]])
    r_true = np.array([[1.0], [1.0], [0.0]])
    T_st_true = np.vstack((np.hstack((C_true, r_true)), np.array([0, 0, 0, 1])))
    T_ts_true = np.linalg.inv(T_st_true)

    # Run ICP
    source_transformed, T_ts_pred = ptp_ICP(source_tree, target_tree, max_iterations, tolerance)

    # Plot results
    #target_tree.plot_overlay(source_tree)
    #target_tree.plot_overlay(source_transformed)

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check that the transformed source is close to target
    assert np.allclose(source_transformed.points, target_tree.points, atol=1e-5)
    assert(True)

def test_3D_ptp_ICP():
    """
    Test 3D point-to-point ICP algorithm.
    """
    # TO DO: NEED TO ADD TEST 3D POINTCLOUDS
    assert(True)

def test_diff_ptp_ICP(source, target, max_iterations, tolerance):
    """
    Test 2D differentiable point-to-point ICP algorithm.
    """

    # Make into tensors
    source = torch.tensor(source, requires_grad=True)
    target = torch.tensor(target, requires_grad=True)

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    C_true = np.array([[0.9950, -0.0998, 0.0], [0.0998, 0.9950, 0.0], [0.0, 0.0, 1.0]])
    r_true = np.array([[1.0], [1.0], [0.0]])
    T_st_true = np.vstack((np.hstack((C_true, r_true)), np.array([0, 0, 0, 1])))
    T_ts_true = np.linalg.inv(T_st_true)

    # Run ICP
    source_transformed, T_ts_pred = diff_ptp_ICP(source, target, max_iterations, tolerance)

    # Plot results
    #target_tree = KDTree(target.tolist())
    #target_tree.plot_overlay(KDTree(source.tolist()))
    #target_tree.plot_overlay(KDTree(source_transformed.tolist()))

    #plot_overlay(source.detach().numpy(), target.detach().numpy())
    #plot_overlay(source_transformed.detach().numpy(), target.detach().numpy())

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check that the transformed source is close to target
    assert np.allclose(source_transformed.detach().numpy(), target.detach().numpy(), atol=1e-5)

    # Check that the gradient is not none
    T_ts_pred.sum().backward()

    assert source.grad is not None and target.grad is not None
