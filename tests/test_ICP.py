"""
    Unit tests for ICP.py
"""
import pytest
import numpy as np
import torch
torch.set_printoptions(precision=8)
from dICP.ICP import ICP
from dICP.visualization import plot_overlay
from dICP.KDTree_ICP import KDTree
from pylgmath import se3op
import matplotlib.pyplot as plt
from pylgmath import Transformation

@pytest.fixture
def max_iterations():
    return 50

@pytest.fixture
def tolerance():
    return 1e-10

@pytest.fixture
def source():
    return np.load('../data/points_scan.npy')

@pytest.fixture
def target():
    return np.load('../data/points_map.npy')

def aatest_pt2pt_kdICP(source, target, max_iterations, tolerance):
    """
    Test non-differentiable point-to-point ICP algorithm.
    """

    # Load test scan and map
    source_tree = KDTree(source[:,:3].tolist())
    target_tree = KDTree(target[:,:3].tolist())
    
    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init = np.eye(4)

    # Run ICP
    pt2pt_ICP = ICP(icp_type='pt2pt', differentiable = False, max_iterations=max_iterations, tolerance=tolerance)
    source_transformed, T_ts_pred = pt2pt_ICP.icp(source_tree, target_tree, T_init)

    # Plot results
    #target_tree.plot_overlay(source_tree)
    #target_tree.plot_overlay(source_transformed)

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred))
    assert(np.linalg.norm(err_T) < tolerance)

def test_pt2pt_dICP(source, target, max_iterations, tolerance):
    """
    Test differentiable point-to-point ICP algorithm.
    """

    # Make into tensors
    source = torch.tensor(source[:,:3], requires_grad=True)
    target = torch.tensor(target[:,:3], requires_grad=True)

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init = torch.eye(4, dtype=source.dtype)
    
    # Set up loss function
    loss_fn = {"name": "huber", "metric": 1.0}
    #loss_fn = {"name": "cauchy", "metric": 0.5}

    # Run ICP
    pt2pt_dICP = ICP(icp_type='pt2pt', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)
    source_transformed, T_ts_pred = pt2pt_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)

    # Plot results
    #plot_overlay(source_transformed.detach().numpy(), target.detach().numpy(), file_name='pt2pt_dICP.png')

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check that the transformed source is close to target
    assert np.allclose(source_transformed[:,:,0].detach().numpy(), target.detach().numpy(), atol=1e-5)

    # Check that the gradient is not none
    T_ts_pred.sum().backward()

    # Confirm gradient exists
    assert source.grad is not None and target.grad is not None
    # Confirm gradient is not nan
    assert torch.isnan(source.grad).any() == False and torch.isnan(target.grad).any() == False

def test_pt2pl_dICP(source, target, max_iterations, tolerance):
    """
    Test differentiable point-to-plane ICP algorithm.
    """

    # Make into tensors
    source = torch.tensor(source[:,:3], requires_grad=True)
    target = torch.tensor(target, requires_grad=True)

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init = torch.eye(4, dtype=source.dtype)

    # Set up loss function
    loss_fn = {"name": "huber", "metric": 10.0}

    # Run ICP
    pt2pl_dICP = ICP(icp_type='pt2pl', differentiable = True, max_iterations=max_iterations, tolerance=tolerance)
    source_transformed, T_ts_pred = pt2pl_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)

    # Plot results
    #plot_overlay(source_transformed[:,:,0].detach().numpy(), target.detach().numpy(), file_name='pt2pl_dICP.png')

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check that the transformed source is close to target
    assert np.allclose(source_transformed[:,:,0].detach().numpy(), target[:,:3].detach().numpy(), atol=1e-5)

    # Check that the gradient is not none
    T_ts_pred.sum().backward()

    assert source.grad is not None and target.grad is not None

def test_pt2pt_ICP(source, target, max_iterations, tolerance):
    """
    Test differentiable point-to-point ICP algorithm.
    """

    # Make into tensors
    source = torch.tensor(source[:,:3], requires_grad=True)
    target = torch.tensor(target[:,:3], requires_grad=True)

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init = torch.eye(4, dtype=source.dtype)
    
    # Set up loss function
    loss_fn = {"name": "huber", "metric": 10.0}

    # Run ICP
    pt2pt_ICP = ICP(icp_type='pt2pt', differentiable=False, max_iterations=max_iterations, tolerance=tolerance)
    source_transformed, T_ts_pred = pt2pt_ICP.icp(source, target, T_init, trim_dist=5.0, loss_fn = loss_fn, dim=2)

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_true @ np.linalg.inv(T_ts_pred.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check that the transformed source is close to target
    assert np.allclose(source_transformed[:,:,0].detach().numpy(), target.detach().numpy(), atol=1e-5)