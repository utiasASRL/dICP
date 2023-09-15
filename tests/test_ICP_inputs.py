"""
    Unit tests for ICP.py
"""
import pytest
import numpy as np
import torch
torch.set_printoptions(precision=8)
from dICP.ICP import ICP
from dICP.visualization import plot_overlay
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

def test_input_types(source, target, max_iterations, tolerance):
    """
    Test direct tensor or list of tensor input types
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
    T_ts_pred_array = torch.tensor(np.zeros((3,4,4)), dtype=test_type)
    tic = time.time()
    for ii in range(3):
        source = source_list[ii]
        target = target_list[ii]
        T_init = T_init_list[ii]
        
        # Run ICP
        icp_results = pt2pt_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)
        T_ts_pred = icp_results['T']

        T_ts_pred_array[ii,:,:] = T_ts_pred
    
    toc = time.time()
    time_loop = toc - tic
    # Now pass as batch
    tic = time.time()
    icp_results = pt2pt_dICP.icp(source_list, target_list, T_init_stack, trim_dist=5.0, loss_fn=loss_fn, dim=2)
    T_ts_pred_batch = icp_results['T']
    toc = time.time()

    time_batch = toc - tic

    print("Time for loop: ", time_loop)
    print("Time for batch: ", time_batch)

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_pred_array.detach().numpy() @ np.linalg.inv(T_ts_pred_batch.detach().numpy()))

    assert(np.linalg.norm(err_T) < tolerance)

def test_zero_inputs(source, target, max_iterations, tolerance):
    """
    Test different combinations of missing data
    """
    source_1 = torch.tensor(source, requires_grad=True)
    target_1 = []

    source_2 = []
    target_2 = torch.tensor(target, requires_grad=True)

    source_3 = []
    target_3 = []

    source_list = [source_1, source_2, source_3]
    target_list = [target_1, target_2, target_3]

    test_type = source_1.dtype

    T_init_1 = torch.eye(4, dtype=test_type)
    T_init_2 = torch.eye(4, dtype=test_type)
    T_init_3 = torch.eye(4, dtype=test_type)

    T_init_list = [T_init_1, T_init_2, T_init_3]
    T_init_stack = torch.stack(T_init_list)

    loss_fn = None
    pt2pt_dICP = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)

    T_ts_pred_array = torch.tensor(np.zeros((len(source_list),4,4)), dtype=test_type)
    for ii in range(len(source_list)):
        source = source_list[ii]
        target = target_list[ii]
        T_init = T_init_list[ii]
        
        # Run ICP
        icp_results = pt2pt_dICP.icp(source, target, T_init, trim_dist=5.0, loss_fn=loss_fn, dim=2)
        T_ts_pred_array[ii,:,:] = icp_results['T']

    icp_results = pt2pt_dICP.icp(source_list, target_list, T_init_stack, trim_dist=5.0, loss_fn=loss_fn, dim=2)
    T_ts_pred_batch = icp_results['T']
    # Since we have empty source/target, the transformation should return initial guess
    assert(np.linalg.norm(T_ts_pred_array.detach().numpy() - T_init_stack.detach().numpy()) < tolerance)
    assert(np.linalg.norm(T_ts_pred_batch.detach().numpy() - T_init_stack.detach().numpy()) < tolerance)

def test_weight_inputs(source, target, max_iterations, tolerance):
    """
    Test different combinations of passing or not passing weights.
    """

    # Make into tensors
    source_1 = torch.tensor(source[:,:3], requires_grad=True)
    target_1 = torch.tensor(target, requires_grad=True)
    weight_1 = None

    source_2 = torch.tensor(source[:,:3], requires_grad=True)
    target_2 = torch.tensor(target[:,:], requires_grad=True)
    weight_2 = torch.tensor(np.ones((target_2.shape[0])), requires_grad=True)

    # Add 10 random points to source with weights of 0
    source_3 = torch.tensor(np.vstack((source[:,:3], np.random.rand(10,3))), requires_grad=True)
    target_3 = torch.tensor(target[:,:], requires_grad=True)
    weight_3 = torch.tensor(np.hstack((np.ones((target_3.shape[0])), np.zeros((10)))), requires_grad=True)

    source_list = [source_1, source_2, source_3]
    target_list = [target_1, target_2, target_3]
    weight_list = [weight_1, weight_2, weight_3]

    test_type = source_1.dtype

    T_init_1 = torch.eye(4, dtype=test_type)
    T_init_2 = torch.eye(4, dtype=test_type)
    T_init_3 = torch.eye(4, dtype=test_type)

    T_init_list = [T_init_1, T_init_2, T_init_3]
    T_init_stack = torch.stack(T_init_list)
    
    # Set up loss function
    loss_fn = {"name": "huber", "metric": 1.0}
    pt2pt_dICP = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)

    # First, test with a single point cloud in loop
    T_ts_pred_array = torch.tensor(np.zeros((len(source_list),4,4)), dtype=test_type)
    for ii in range(len(source_list)):
        source = source_list[ii]
        target = target_list[ii]
        T_init = T_init_list[ii]
        weight = weight_list[ii]
        
        # Run ICP
        icp_results = pt2pt_dICP.icp(source, target, T_init, weight=weight, trim_dist=5.0, loss_fn=loss_fn, dim=2)
        T_ts_pred_array[ii,:,:] = icp_results['T']

    icp_results = pt2pt_dICP.icp(source_list, target_list, T_init_stack, weight=weight_list, trim_dist=5.0, loss_fn=loss_fn, dim=2)
    T_ts_pred_batch = icp_results['T']
    # Check that passing weight as list returns same result as individual eval
    assert(np.linalg.norm(T_ts_pred_batch.detach().numpy() - T_ts_pred_array.detach().numpy()) < tolerance)
    # Check that all 3 transformations are the same, since the used points should all be the same
    assert(np.linalg.norm(T_ts_pred_array[0,:,:].detach().numpy() - T_ts_pred_array[1,:,:].detach().numpy()) < tolerance)
    assert(np.linalg.norm(T_ts_pred_array[0,:,:].detach().numpy() - T_ts_pred_array[2,:,:].detach().numpy()) < tolerance)

def test_diff_vs_nondiff_types(source, target, max_iterations, tolerance):
    """
    Test that differentiable ICP returns the same result as non-differentiable ICP in easy conditions.
    """

    # Make into tensors
    source_1 = torch.tensor(source[:50,:3], requires_grad=True)
    target_1 = torch.tensor(target[:55,:], requires_grad=True)
    T_init_1 = torch.eye(4, dtype=source_1.dtype)

    # Check huber and pt2pl
    loss_fn = {"name": "huber", "metric": 1.0}
    trim_dist = 5.0
    pt2pt_dICP_diff = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)
    pt2pt_dICP_nondiff = ICP(icp_type='pt2pl', differentiable=False, max_iterations=max_iterations, tolerance=tolerance)

    # First, test with a single point cloud in loop
    icp_results_diff = pt2pt_dICP_diff.icp(source_1, target_1, T_init_1, trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
    icp_results_nondiff = pt2pt_dICP_nondiff.icp(source_1, target_1, T_init_1, trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
    T_ts_diff = icp_results_diff['T']
    T_ts_nondiff = icp_results_nondiff['T']
    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_diff.detach().numpy() @ np.linalg.inv(T_ts_nondiff.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)

    # Check cauchy and pt2pt
    loss_fn = {"name": "cauchy", "metric": 0.5}
    trim_dist = 5.0
    pt2pl_dICP_diff = ICP(icp_type='pt2pl', differentiable=True, max_iterations=max_iterations, tolerance=tolerance)
    pt2pl_dICP_nondiff = ICP(icp_type='pt2pl', differentiable=False, max_iterations=max_iterations, tolerance=tolerance)

    # First, test with a single point cloud in loop
    icp_results_diff = pt2pl_dICP_diff.icp(source_1, target_1, T_init_1, trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
    icp_results_nondiff = pt2pl_dICP_nondiff.icp(source_1, target_1, T_init_1, trim_dist=trim_dist, loss_fn=loss_fn, dim=2)
    T_ts_diff = icp_results_diff['T']
    T_ts_nondiff = icp_results_nondiff['T']

    # Check that the transformation is correct
    err_T = se3op.tran2vec(T_ts_diff.detach().numpy() @ np.linalg.inv(T_ts_nondiff.detach().numpy()))
    assert(np.linalg.norm(err_T) < tolerance)