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
import time 

@pytest.fixture
def max_iterations():
    return 25

@pytest.fixture
def tolerance():
    return 1e-8

@pytest.fixture
def source():
    return np.load('../data/points_scan.npy')

@pytest.fixture
def target():
    return np.load('../data/points_map.npy')

def test_diff_inputs(source, target, max_iterations, tolerance):
    """
    Test differentiable point-to-point ICP algorithm.
    """

    # Make into tensors
    source_1 = torch.tensor(source[:50,:3], requires_grad=True)
    target_1 = torch.tensor(target[:55,:], requires_grad=True)

    source_2 = torch.tensor(source[:,:3], requires_grad=True)
    target_2 = torch.tensor(target[:,:], requires_grad=True)

    source_3 = torch.tensor(source[:55,:3], requires_grad=True)
    target_3 = torch.tensor(target[:60,:], requires_grad=True)

    source_list = [source_1, source_2, source_3]
    target_list = [target_1, target_2, target_3]

    test_type = source_1.dtype

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init_1 = torch.eye(4, dtype=test_type)
    T_init_2 = torch.eye(4, dtype=test_type)
    T_init_3 = torch.eye(4, dtype=test_type)

    T_init_list = [T_init_1, T_init_2, T_init_3]
    T_init_stack = torch.stack(T_init_list)
    
    # Set up loss function
    loss_fn = {"name": "huber", "metric": 1.0}
    pt2pt_dICP = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)

    # First, test with a single point cloud in loop
    source_transformed_list = []
    T_ts_pred_array = torch.tensor(np.zeros((3,4,4)), dtype=test_type)
    tic = time.time()
    for ii in range(3):
        source = source_list[ii]
        target = target_list[ii]
        T_init = T_init_list[ii]
        
        # Run ICP
        source_transformed, T_ts_pred = pt2pt_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)

        source_transformed_list.append(source_transformed)
        T_ts_pred_array[ii,:,:] = T_ts_pred
    
    toc = time.time()
    time_loop = toc - tic
    # Now pass as batch
    tic = time.time()
    source_transformed_batch, T_ts_pred_batch = pt2pt_dICP.icp(source_list, target_list, T_init_stack, trim_dist=5.0, loss_fn=loss_fn, dim=2)
    toc = time.time()

    time_batch = toc - tic

    print("Time for loop: ", time_loop)
    print("Time for batch: ", time_batch)

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_pred_array.detach().numpy() @ np.linalg.inv(T_ts_pred_batch.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

def test_zero_inputs(source, target, max_iterations, tolerance):
    source_1 = []
    target_1 = torch.tensor(target, requires_grad=True)

    source_2 = []
    target_2 = torch.tensor(target, requires_grad=True)

    source_list = [source_1, source_2]
    target_list = [target_1, target_2]

    test_type = target_2.dtype

    # True 2D transformation is [.1,1,1] (phi,x,y), form T_true 3D transformation
    relative_pos_xi = np.array([1.0,1.0,0, 0,0,.1]).reshape((6,1))
    T_st_true = Transformation(xi_ab=relative_pos_xi).matrix()
    T_ts_true = np.linalg.inv(T_st_true)

    T_init_1 = torch.eye(4, dtype=test_type)
    T_init_2 = torch.eye(4, dtype=test_type)

    T_init_list = [T_init_1, T_init_2]
    T_init_stack = torch.stack(T_init_list)

    loss_fn = None
    pt2pt_dICP = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)

    source_transformed_list = []
    T_ts_pred_array = torch.tensor(np.zeros((2,4,4)), dtype=test_type)
    for ii in range(2):
        source = source_list[ii]
        target = target_list[ii]
        T_init = T_init_list[ii]
        
        # Run ICP
        _, T_ts_pred = pt2pt_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)

        T_ts_pred_array[ii,:,:] = T_ts_pred
        print(ii)

    _, T_ts_pred_batch = pt2pt_dICP.icp(source_list, target_list, T_init_stack, trim_dist=5.0, loss_fn=loss_fn, dim=2)

    print(T_ts_pred_array)
    print(T_ts_pred_batch)

    # Since we have empty source/target, the transformation should return initial guess
    assert(np.linalg.norm(T_ts_pred_array.detach().numpy() - T_init_stack.detach().numpy()) < tolerance)
    assert(np.linalg.norm(T_ts_pred_batch.detach().numpy() - T_init_stack.detach().numpy()) < tolerance)
